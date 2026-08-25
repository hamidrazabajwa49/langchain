from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model=ChatOpenAI(model='gpt-4')

st.header("Research Paper Summarizer")
user_query=st.text_input("Enter your query")

if st.button("Summarize"):
    result=model.invoke(user_query)
    st.write(result.content)
