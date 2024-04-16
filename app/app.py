import streamlit as st
from src.client import ChatClient

st.title('about-me')

client = ChatClient()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar for uploading new text
new_text = st.sidebar.text_area("Add new text to the vector database")
if new_text:
    client.qa._add_new_docs(new_text)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.respond(prompt)
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
