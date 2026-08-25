from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as numpy

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=500)

documents = [
    "The batsman hit a six over long-on to win the match.",
    "Virat Kohli scored a century in the World Cup final.",
    "The bowler took a hat-trick in the final over.",
    "India won the cricket match by five wickets.",
    "The umpire declared the batsman out for handling the ball.",
    "Rain delayed the start of the second innings.",
    "The fielder took a stunning catch at the boundary.",
    "The team captain won the toss and chose to bat first.",
    "He was bowled out for a duck in the first over.",
    "The stadium erupted as the winning run was scored.",
]

query = "Who scored a hundred in the finals?"

doc_embeddings=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)

scores=cosine_similarity([query_embedding],doc_embeddings)[0]

index , score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print(f'Similarity Score is:{score}')
