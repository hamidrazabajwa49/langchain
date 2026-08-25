from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

result=model.invoke("What is the capital of Pakistan?")

print(f'\n{result.content}')
