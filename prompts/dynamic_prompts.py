from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model=ChatOpenAI(model='gpt-4')

input_paper=st.selectbox("Select the Research Paper",["Attention is all you need","BERT","Word2Vec","GPT-3"])

style_input=st.selectbox("Select the Explanation tone",["Beginner Friendly","Technical","Code Oriented", "Detailed Explanation"])

output_length=st.selectbox("Select the length of the output",["Short(1-2 paragraph)","Medium(3-5 paragraph)","long (detailed explaination)"])

template1 = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following Explanation Style: {explain_style} and Explanation Length: {length}.

1. Mathematical Details:
    - Include relevant mathematical equations if present in the paper.
    - Explain the intuition behind each equation using simple, intuitive terms.
    - If no equations exist, state: "No mathematical equations were found in this paper."

2. Analogies:
    - Use relatable, real-world analogies to simplify complex or technical ideas.

3. Adherence to Length:
    - Strictly follow the given Explanation Length. Do not go over or under it.

4. Fidelity to Source:
    - Base the explanation only on the actual content of the paper.
    - If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing or fabricating.

5. Output Format:
    - Use clear headings, bullet points, and structured formatting (Markdown) for readability.

Ensure the summary is accurate, well-organized, and easy to follow for the specified audience level.
""",
    input_variables=["paper_input", "explain_style", "length"],
)

prompt=template1.invoke({
    'paper_input':input_paper,
    'explain_style':style_input,
    'length':output_length
})

if st.button("Summarize"):
    result=model.invoke(prompt)
    st.write(result.content)
