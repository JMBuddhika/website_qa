"""
Website QA RAG System with Conversational Memory
A Streamlit app that crawls websites and answers questions using RAG (Retrieval Augmented Generation)

Requirements:
    pip install streamlit langchain langchain-community langchain-text-splitters 
                faiss-cpu sentence-transformers beautifulsoup4 lxml requests

Run:
    streamlit run app.py
"""

import os
import re
import time
import urllib.parse
import requests
from uuid import uuid4
from collections import deque
from typing import List, Set, Tuple, Optional
from operator import itemgetter

import streamlit as st
from bs4 import BeautifulSoup

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# ===========================
# Configuration & Constants
# ===========================

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; WebsiteQA-RAG/1.0)"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:1.5b"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
RETRIEVAL_K = 4

# ===========================
# Page Setup
# ===========================

st.set_page_config(
    page_title="Website QA RAG (with Memory)",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Website QA — with Conversational Memory")

# ===========================
# Sidebar Configuration
# ===========================

st.sidebar.header("⚙️ Settings")

# Model settings
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

# Crawl settings
st.sidebar.subheader("🕷️ Crawl Settings")
max_pages = st.sidebar.slider("Max pages to crawl", 1, 200, 25, 1)
max_depth = st.sidebar.slider("Max link depth", 0, 5, 2, 1)
same_domain_only = st.sidebar.checkbox("Stay on same domain", True)
respect_robots = st.sidebar.checkbox("Respect robots.txt", True)
request_delay = st.sidebar.slider("Delay between requests (seconds)", 0.0, 3.0, 0.25, 0.05)

# ===========================
# Session State Initialization
# ===========================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "conversation_stores" not in st.session_state:
    st.session_state.conversation_stores = {}

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "index_info" not in st.session_state:
    st.session_state.index_info = ""


# ===========================
# Caching Functions
# ===========================

@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load the sentence transformer model for embeddings"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def create_llm(model_name: str, temp: float):
    """Initialize the Ollama language model"""
    return ChatOllama(
        model=model_name,
        temperature=float(temp),
        base_url=OLLAMA_BASE_URL
    )


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_webpage(url: str, user_agent: str, timeout: int = 20) -> Tuple[str, Optional[str]]:
    """
    Fetch a webpage and return its content
    Returns: (final_url, html_content or None)
    """
    headers = {"User-Agent": user_agent}

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")

        # Only process HTML pages
        if "text/html" not in content_type:
            return (response.url, None)

        return (response.url, response.text)

    except Exception as e:
        return (url, None)


@st.cache_data(show_spinner=False, ttl=600)
def parse_robots_txt(root_url: str, user_agent: str) -> List[str]:
    """
    Simple robots.txt parser - returns list of disallowed paths
    """
    try:
        parsed_url = urllib.parse.urlparse(root_url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

        _, robots_content = fetch_webpage(robots_url, user_agent, timeout=10)

        if not robots_content:
            return []

        disallowed_paths = []
        currently_applies = False

        for line in robots_content.splitlines():
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            if line.lower().startswith("user-agent:"):
                agent = line.split(":", 1)[1].strip()
                # Apply to wildcard or our specific agent
                currently_applies = (agent == "*") or (agent.lower() in user_agent.lower())

            elif currently_applies and line.lower().startswith("disallow:"):
                path = line.split(":", 1)[1].strip()
                if path:
                    disallowed_paths.append(path)

        return disallowed_paths

    except Exception:
        return []


# ===========================
# Helper Functions
# ===========================

def is_url_disallowed(url: str, disallowed_paths: List[str]) -> bool:
    """Check if a URL matches any disallowed paths from robots.txt"""
    if not disallowed_paths:
        return False

    parsed = urllib.parse.urlparse(url)
    path = parsed.path or "/"

    for disallowed in disallowed_paths:
        if path.startswith(disallowed):
            return True

    return False


def are_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs are from the same domain"""
    parsed1 = urllib.parse.urlparse(url1)
    parsed2 = urllib.parse.urlparse(url2)

    return (parsed1.scheme == parsed2.scheme and
            parsed1.netloc == parsed2.netloc)


def clean_and_normalize_url(base_url: str, link: str) -> Optional[str]:
    """Convert relative links to absolute and normalize URLs"""
    if not link:
        return None

    # Make absolute
    absolute_url = urllib.parse.urljoin(base_url, link)

    parsed = urllib.parse.urlparse(absolute_url)

    # Only accept http/https
    if parsed.scheme not in ("http", "https"):
        return None

    # Remove fragment (#section)
    parsed = parsed._replace(fragment="")

    return urllib.parse.urlunparse(parsed)


def extract_text_from_html(html: str) -> Tuple[str, str]:
    """
    Extract clean text and title from HTML
    Returns: (text_content, page_title)
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove script and style elements
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    # Extract text
    text = soup.get_text(separator="\n")

    # Clean up excessive newlines
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Get page title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    return text.strip(), title


# ===========================
# Web Crawler
# ===========================

def crawl_website(
        start_url: str,
        max_pages: int,
        max_depth: int,
        user_agent: str,
        same_domain_only: bool,
        respect_robots: bool,
        delay_between_requests: float
) -> List[Document]:
    """
    Crawl a website starting from start_url using BFS
    Returns list of Document objects with page content and metadata
    """
    if not start_url:
        return []

    # Normalize the starting URL
    start_url = clean_and_normalize_url(start_url, "") or start_url
    start_domain = urllib.parse.urlparse(start_url).netloc

    # Get robots.txt restrictions if needed
    disallowed = []
    if respect_robots:
        disallowed = parse_robots_txt(start_url, user_agent)

    # Initialize crawl
    visited_urls = set()
    url_queue = deque([(start_url, 0)])  # (url, depth)
    documents = []

    # Breadth-first search
    while url_queue and len(visited_urls) < max_pages:
        current_url, depth = url_queue.popleft()

        # Skip if already visited
        if current_url in visited_urls:
            continue

        # Check robots.txt
        if respect_robots and is_url_disallowed(current_url, disallowed):
            continue

        # Fetch the page
        final_url, html_content = fetch_webpage(current_url, user_agent)
        visited_urls.add(current_url)

        if html_content:
            # Extract text content
            text, title = extract_text_from_html(html_content)

            if text:
                # Create document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": final_url,
                        "title": title or final_url
                    }
                )
                documents.append(doc)

            # Find new links if we haven't reached max depth
            if depth < max_depth:
                soup = BeautifulSoup(html_content, "lxml")

                for anchor in soup.find_all("a", href=True):
                    next_url = clean_and_normalize_url(final_url, anchor["href"])

                    if not next_url or next_url in visited_urls:
                        continue

                    # Check domain restriction
                    if same_domain_only:
                        if urllib.parse.urlparse(next_url).netloc != start_domain:
                            continue

                    url_queue.append((next_url, depth + 1))

        # Be polite - delay between requests
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)

        # Safety check
        if len(visited_urls) >= max_pages:
            break

    return documents


# ===========================
# Vector Store & Retriever
# ===========================

def create_retriever_from_documents(documents: List[Document]):
    """
    Split documents into chunks and create FAISS retriever
    """
    if not documents:
        return None, None

    # Split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K}
    )

    return retriever, vectorstore


def format_retrieved_documents(docs: List[Document]) -> str:
    """Format retrieved documents for the prompt"""
    formatted = []

    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get('title', '')
        content = doc.page_content
        formatted.append(f"[{i}] ({title}) {content}")

    return "\n\n".join(formatted)


# ===========================
# Conversation Management
# ===========================

def get_conversation_history(session_id: str) -> ChatMessageHistory:
    """Get or create conversation history for a session"""
    if session_id not in st.session_state.conversation_stores:
        st.session_state.conversation_stores[session_id] = ChatMessageHistory()

    return st.session_state.conversation_stores[session_id]


# ===========================
# RAG Chain Setup
# ===========================

# Define the prompt template
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context from the website.

If the context doesn't contain enough information to answer the question, say you don't know.
Be concise and cite specific source URLs when helpful."""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# ===========================
# Main UI
# ===========================

# URL input section
url_input = st.text_input(
    "Enter the website URL to crawl",
    value="",
    placeholder="https://example.com"
)

crawl_button = st.button("🔎 Start Crawling & Indexing")

# Reset conversation button
col1, col2 = st.columns([3, 8])
with col1:
    if st.button("🧹 Reset Conversation"):
        st.session_state.conversation_stores[st.session_state.session_id] = ChatMessageHistory()
        st.rerun()

# ===========================
# Crawling & Indexing
# ===========================

if crawl_button and url_input.strip():
    with st.spinner("Crawling website and building index... This may take a while."):

        # Crawl the website
        crawled_docs = crawl_website(
            start_url=url_input.strip(),
            max_pages=max_pages,
            max_depth=max_depth,
            user_agent=DEFAULT_USER_AGENT,
            same_domain_only=same_domain_only,
            respect_robots=respect_robots,
            delay_between_requests=request_delay
        )

        if not crawled_docs:
            st.error("❌ No content found. Try adjusting the crawl settings.")
        else:
            # Build the retriever
            retriever, vectorstore = create_retriever_from_documents(crawled_docs)

            if retriever is None:
                st.error("❌ Failed to build the search index.")
            else:
                # Save to session state
                st.session_state.retriever = retriever
                st.session_state.vectorstore = vectorstore

                # Calculate stats
                total_chars = sum(len(doc.page_content) for doc in crawled_docs)
                st.session_state.index_info = (
                    f"✅ Indexed {len(crawled_docs)} pages "
                    f"(~{total_chars:,} characters)"
                )

                st.success(st.session_state.index_info)

# ===========================
# Chat Interface
# ===========================

if st.session_state.retriever is not None:

    # Build the conversational RAG chain
    llm = create_llm(DEFAULT_MODEL, temperature)

    # Chain components
    retrieve_context = (
            itemgetter("question") |
            st.session_state.retriever |
            format_retrieved_documents
    )

    base_chain = RunnablePassthrough.assign(context=retrieve_context)
    rag_chain = base_chain | RAG_PROMPT | llm | StrOutputParser()

    # Add conversation memory
    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=get_conversation_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    # Display chat section
    st.subheader("💬 Ask Questions")

    if st.session_state.index_info:
        st.caption(f"📦 {st.session_state.index_info}")

    # Show conversation history
    history = get_conversation_history(st.session_state.session_id).messages

    for message in history:
        role = "user" if message.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # Chat input
    user_question = st.chat_input("Ask a question about the website...")

    if user_question:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate answer
        with st.spinner("Thinking..."):
            answer = conversational_rag.invoke(
                {"question": user_question},
                config={
                    "configurable": {
                        "session_id": st.session_state.session_id
                    }
                }
            )

            # Get source documents
            source_docs = st.session_state.retriever.invoke(user_question)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)

            # Show sources
            if source_docs:
                st.markdown("**🔗 Sources**")

                for i, doc in enumerate(source_docs, 1):
                    metadata = doc.metadata or {}
                    title = metadata.get("title") or metadata.get("source") or "Source"
                    source_url = metadata.get("source", "")

                    st.markdown(f"- {i}. [{title}]({source_url})")

else:
    st.info("👆 Enter a URL above and click **Start Crawling** to begin!")