from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model='gpt-4' , temperature=1.5)

result=model.invoke("Write five lines on cricket in poetic manner.")

print(result)
print('\n\n')
print(f'{result.content}')
