"""Streamlit chat UI for the LangChain Neo4j Agent."""

import os
import uuid

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Neo4j Chat Agent", page_icon="🔗")
st.title("Neo4j Chat Agent")
st.caption("LangChain + Neo4j MCP + Agent Memory")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    user_id = st.text_input("User ID", value="default-user")
    if st.button("New Session"):
        st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        """
        **Features:**
        - Query Neo4j via MCP tools
        - Persistent conversation memory
        - User preference extraction
        """
    )

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something about your Neo4j database..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream(
                    "POST",
                    f"{API_URL}/chat/stream",
                    json={
                        "message": prompt,
                        "session_id": st.session_state.session_id,
                        "user_id": user_id,
                    },
                ) as response:
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            full_response += data
                            placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
        except httpx.ConnectError:
            full_response = "Could not connect to the API server. Make sure it's running."
            placeholder.error(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
