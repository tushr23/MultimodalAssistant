#!/usr/bin/env python3
"""
Professional GenAI Chatbot Frontend
Built by Tushr Verma - Advanced AI Chat Interface
"""
import os
import time
from typing import Dict, Any, List

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Backend configuration
API_BASE = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
CHAT_URL = f"{API_BASE}/v1/chat"
HEALTH_URL = f"{API_BASE}/health"

# Page configuration
st.set_page_config(
    page_title="GenAI Chatbot - Tushr Verma",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/tushr23/MultimodalAssistant',
        'Report a bug': 'https://github.com/tushr23/MultimodalAssistant/issues',
        'About': "# GenAI Chatbot\nBuilt by **Tushr Verma**\n\nA professional AI chat interface powered by HuggingFace models."
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    .author-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.5rem 0;
        display: inline-block;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .model-info {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
        font-size: 0.8rem;
        color: #666;
    }
    /* Fix for white box in sidebar */
    .stSidebar .stTextInput, .stSidebar .stTextArea {
        background-color: transparent !important;
    }
    /* Fix chat input styling */
    .stChatInput {
        background-color: #1e1e1e !important;
    }
    /* Hide any empty elements that might be causing the white box */
    .stSidebar .element-container:empty {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


def make_session() -> requests.Session:
    """Create a session with retry logic."""
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


session = make_session()


@st.cache_data(ttl=10)
def check_health() -> Dict[str, Any]:
    """Quick health check of the backend."""
    try:
        r = session.get(HEALTH_URL, timeout=5)
        return r.json() if r.ok else {"status": "unhealthy"}
    except Exception:
        return {"status": "error"}


# Session state initialization

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_status" not in st.session_state:
    st.session_state.system_status = None


def main():
    """Main application interface with enhanced UI."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 GenAI Chatbot</h1>
        <div class="author-badge">Built by Tushr Verma</div>
    </div>
    """, unsafe_allow_html=True)
    
    # System status in sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Controls")
        
        # Health status check
        health_status = check_health()
        if health_status.get("status") == "healthy":
            st.markdown('<p class="status-healthy">✅ System Online</p>', unsafe_allow_html=True)
            st.markdown(f"**Uptime:** {health_status.get('uptime', 'Unknown')}")
        else:
            st.markdown('<p class="status-error">❌ System Offline</p>', unsafe_allow_html=True)
        
        st.divider()
        
        # Model parameters
        st.markdown("### ⚙️ Model Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Higher values make responses more creative but less focused"
        )
        
        top_p = st.slider(
            "Top-p",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Controls diversity via nucleus sampling"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=1024,  # Match backend validation limit
            value=512,       # Use a safer default value
            step=50,
            help="Maximum length of response (limited to 1024 by backend)"
        )
        
        st.divider()
        
        # Model information
        llm = health_status.get("llm_router", {}) if isinstance(health_status, dict) else {}
        provider = llm.get("provider", "unknown")
        model = llm.get("model", "unknown")
        configured = llm.get("provider_configured", False)
        st.markdown(f"""
        <div class="model-info">
            <strong>🧠 AI Router</strong><br/>
            Provider: <code>{provider}</code><br/>
            Model: <code>{model}</code><br/>
            <small>{'Keys configured' if configured else 'No keys detected - using Ollama (local) then HF fallback'}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
    # Main chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    
                    # Create messages list in the format expected by backend
                    messages_for_api = []
                    for msg in st.session_state.messages:
                        messages_for_api.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    # Add the current user message
                    messages_for_api.append({
                        "role": "user", 
                        "content": prompt
                    })
                    
                    payload = {
                        "messages": messages_for_api,  # Use messages (plural)
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_tokens  # Use max_new_tokens
                    }
                    
                    # Make API request with enhanced error handling
                    response = session.post(CHAT_URL, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        # Extract the assistant's message from the response
                        if result.get("choices") and len(result["choices"]) > 0:
                            full_response = result["choices"][0]["message"]["content"]
                        else:
                            full_response = "No response received from the AI model"
                        elapsed = time.time() - start_time
                        message_placeholder.markdown(full_response)
                        st.caption(f"⚡ Generated in {elapsed:.1f}s")
                    elif response.status_code == 429:
                        full_response = "⚠️ Rate limit exceeded. Please wait a moment and try again."
                        message_placeholder.markdown(full_response)
                    elif response.status_code == 503:
                        full_response = "🔧 AI service temporarily unavailable. Please try again in a few moments."
                        message_placeholder.markdown(full_response)
                    else:
                        full_response = f"❌ Error: {response.status_code} - {response.text}"
                        message_placeholder.markdown(full_response)
                        
            except requests.exceptions.Timeout:
                full_response = "⏱️ Request timed out. The AI might be processing a complex query. Please try again."
                message_placeholder.markdown(full_response)
            except requests.exceptions.ConnectionError:
                full_response = "🔌 Connection error. Please check if the backend service is running."
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"💥 Unexpected error: {str(e)}"
                message_placeholder.markdown(full_response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>⚡ Powered by HuggingFace Models | 🚀 Built with FastAPI & Streamlit</p>
        <p><strong>Created by Tushr Verma</strong> | Professional AI Chat Interface</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
