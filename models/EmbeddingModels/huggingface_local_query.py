from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text="Karachi is the capital of Pakistan"

vector=embeddings.embed_query(text)
print(str(vector))

document=[
    "Karachi is the capital of Pakistan.",
    "Paris is the capital of France",
    "Dhaka is the capital of Bangladesh"
]

result1=embedding.embed_documents(document)
print(f'\n{result1}') 
