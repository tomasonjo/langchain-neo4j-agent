"""Streamlit chat UI for the LangChain Neo4j Agent."""

import json
import os
import uuid

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Neo4j Chat Agent", page_icon="🔗")
st.title("Neo4j Chat Agent")
st.caption("LangChain + Neo4j MCP + Agent Memory")


def render_tool(name: str, args_str: str, result: str | None = None):
    """Render a completed tool call as a collapsed status block."""
    with st.status(f"🛠️ {name}", state="complete", expanded=False):
        st.markdown("**📤 Input**")
        st.code(args_str, language="json")
        if result is not None:
            st.markdown("**📥 Output**")
            st.code(result, language="text")


def render_assistant_message(content: str, tool_uses: list[dict] | None = None):
    """Render an assistant message with optional tool usage blocks above the text."""
    if tool_uses:
        for tool in tool_uses:
            render_tool(
                tool["name"],
                json.dumps(tool.get("args", {}), indent=2),
                tool.get("result"),
            )
    if content:
        st.markdown(content)


def fetch_users() -> list[str]:
    try:
        resp = httpx.get(f"{API_URL}/users", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def fetch_sessions(user_id: str | None = None) -> list[dict]:
    try:
        params = {}
        if user_id:
            params["user_id"] = user_id
        resp = httpx.get(f"{API_URL}/sessions", params=params, timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def fetch_session_messages(session_id: str) -> list[dict]:
    try:
        resp = httpx.get(f"{API_URL}/sessions/{session_id}/messages", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def stream_response(prompt: str, session_id: str, user_id: str) -> dict:
    """Run the streaming request, show a spinner + streamed text, return the final message dict."""
    tool_uses = []
    full_response = ""
    active_tools = {}  # tc_id -> name

    # Single status line for tool activity + a placeholder for streamed text
    tool_label = st.empty()
    text_placeholder = st.empty()

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{API_URL}/chat/stream",
                json={
                    "message": prompt,
                    "session_id": session_id,
                    "user_id": user_id,
                },
            ) as response:
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")
                    if etype == "done":
                        break
                    elif etype == "error":
                        full_response += f"\n\n**Error:** {event.get('content', 'Unknown error')}"
                        text_placeholder.markdown(full_response)
                        break
                    elif etype == "tool_call":
                        tc_id = event.get("id", "")
                        name = event["name"]
                        args = event.get("args", {})
                        active_tools[tc_id] = name
                        tool_uses.append({"name": name, "args": args, "result": None, "_id": tc_id})
                        tool_label.markdown(f"🛠️ Running **{name}**...")
                    elif etype == "tool_result":
                        tc_id = event.get("tool_call_id", "")
                        content = event.get("content", "")
                        if not isinstance(content, str):
                            content = json.dumps(content, indent=2)
                        active_tools.pop(tc_id, None)
                        result_preview = content[:500] + "..." if len(content) > 500 else content
                        for tu in tool_uses:
                            if tu.get("_id") == tc_id:
                                tu["result"] = result_preview
                                break
                        if active_tools:
                            names = ", ".join(f"**{n}**" for n in active_tools.values())
                            tool_label.markdown(f"🛠️ Running {names}...")
                        else:
                            tool_label.empty()
                    elif etype == "token":
                        full_response += event.get("content", "")
                        text_placeholder.markdown(full_response + "▌")

        tool_label.empty()
        text_placeholder.empty()

    except httpx.ConnectError:
        tool_label.empty()
        text_placeholder.empty()
        full_response = "Could not connect to the API server. Make sure it's running."

    clean_tools = [{"name": t["name"], "args": t["args"], "result": t["result"]} for t in tool_uses] or None
    return {
        "role": "assistant",
        "content": full_response,
        "tool_uses": clean_tools,
    }


# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Settings")

    existing_users = fetch_users()
    NEW_USER_OPTION = "+ New user..."
    options = existing_users + [NEW_USER_OPTION] if existing_users else [NEW_USER_OPTION]
    selected = st.selectbox("User", options, index=0)
    if selected == NEW_USER_OPTION:
        user_id = st.text_input("New User ID", placeholder="Enter a new user ID")
        user_id = user_id.strip() or "default-user"
    else:
        user_id = selected

    st.divider()

    if st.button("New Session", use_container_width=True):
        st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

    st.subheader("Sessions")
    sessions = fetch_sessions(user_id=user_id)

    known_ids = {s["session_id"] for s in sessions}
    if st.session_state.session_id not in known_ids:
        st.button(
            "▶ New conversation",
            key="sess_current_new",
            use_container_width=True,
            disabled=True,
            type="primary",
        )

    if sessions:
        for s in sessions:
            label = s.get("first_message_preview") or s["session_id"]
            if len(label) > 50:
                label = label[:47] + "..."
            count = s.get("message_count", 0)
            ts = s.get("updated_at") or s.get("created_at") or ""
            if ts:
                ts = ts[:16].replace("T", " ")

            is_active = s["session_id"] == st.session_state.session_id
            btn_label = f"{'▶ ' if is_active else ''}{label}"
            help_text = f"{count} msgs · {ts}" if ts else f"{count} msgs"

            if st.button(
                btn_label,
                key=f"sess_{s['session_id']}",
                use_container_width=True,
                help=help_text,
                disabled=is_active,
            ):
                st.session_state.session_id = s["session_id"]
                st.session_state.messages = fetch_session_messages(s["session_id"])
                st.rerun()
    else:
        st.caption("No sessions yet.")

# Display chat history — this renders ALL messages including the latest
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_message(msg["content"], msg.get("tool_uses"))
        else:
            st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something about your Neo4j database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = stream_response(prompt, st.session_state.session_id, user_id)
        st.session_state.messages.append(result)

    # Rerun so the message renders through render_assistant_message — same as history
    st.rerun()
