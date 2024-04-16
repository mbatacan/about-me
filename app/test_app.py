import streamlit as st
from src.client import ChatClient


st.title('about me')


def generate_response(input_text):
    chain = ChatClient()
    st.info(chain.respond(input_text)['result'])


new_text = st.sidebar.text_area("Add new text to the vector database")
if new_text:
    chain.qa._add_new_docs(new_text)

with st.form('my_form'):
    text = st.text_area(
        'Enter text:',
        'What excites you about this role?',
    )
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(text)
