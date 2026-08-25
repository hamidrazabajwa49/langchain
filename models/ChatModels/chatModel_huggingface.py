from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

LLM = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",
    provider="auto",              
    huggingfacehub_api_token=token,
)

model = ChatHuggingFace(llm=LLM)
result = model.invoke("What is the capital of Pakistan?")
print(result.content)
