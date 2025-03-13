# app/streamlit_ui.py
import streamlit as st
import requests
from datetime import datetime

# Configuration
FASTAPI_ENDPOINT = "http://localhost:8000/query"
AVATARS = {"user": "👩", "assistant": "💊"}
TITLE = "MedShop AI"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to MedShop! Ask me about medical inventory or drug information.",
            "timestamp": datetime.now().isoformat()
        }
    ]

# Page configuration
st.set_page_config(page_title=TITLE, page_icon="🏥")
st.title(TITLE)

# Sidebar with API status
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get("http://localhost:8000/health", timeout=3).json()
        st.success(f"API Status: {health['status'].capitalize()}")
        st.caption(f"Environment: {health['environment']}")
        st.caption(f"Database: {'Connected' if health['database_connected'] else 'Disconnected'}")
    except requests.exceptions.RequestException:
        st.error("API Unavailable")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS.get(message["role"])):
        st.markdown(message["content"])
        st.caption(f"_{datetime.fromisoformat(message['timestamp']).strftime('%H:%M')}_")

# Handle user input
if prompt := st.chat_input("Ask about medical inventory..."):
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            response = requests.post(
                FASTAPI_ENDPOINT,
                json={"question": prompt},
                timeout=30
            ).json()
            
            full_response = response.get("result", "No response received")
            
        except requests.exceptions.RequestException as e:
            full_response = f"⚠️ API Error: {str(e)}"
        
        # Display response
        response_placeholder.markdown(full_response)
    
    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "timestamp": datetime.now().isoformat()
    })
