import streamlit as st
from src.client import ChatClient


chain = ChatClient()

st.title('about me')
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')


def generate_response(input_text):
    st.info(chain.respond(input_text))


with st.form('my_form'):
    text = st.text_area(
        'Enter text:',
        'What are the three key pieces of advice for learning how to code?',
    )
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)
