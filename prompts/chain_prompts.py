from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

template=load_prompt('template.json')
model = ChatOpenAI(model='gpt-4')

input_paper=st.selectbox("Select the Research Paper",["Attention is all you need","BERT","Word2Vec","GPT-3"])

style_input=st.selectbox("Select the Explanation tone",["Beginner Friendly","Technical","Code Oriented", "Detailed Explanation"])

output_length=st.selectbox("Select the length of the output",["Short(1-2 paragraph)","Medium(3-5 paragraph)","long (detailed explaination)"])

if st.button("Summarize"):
    chain = template | model
    prompt=template.invoke({
    'paper_input':input_paper,
    'explain_style':style_input,
    'length':output_length
    })
    result = model.invoke(prompt)
    st.write(result.content)
