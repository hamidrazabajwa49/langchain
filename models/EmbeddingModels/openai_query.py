from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large' , dimensions=32)

result = embedding.embed_query("Karachi is the capital of Pakistan")
print(str(result))


document=[
    "Karachi is the capital of Pakistan.",
    "Paris is the capital of France",
    "Dhaka is the capital of Bangladesh"
]

result1=embedding.embed_documents(document)
print('\n\n')
print(f'{result1}')
