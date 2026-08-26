from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4')

chat_history={}

while(True):
    user_input=input("Prompt: ")
    if user_input == 'Exit':
        break
    result = model.invoke(user_input)
    chat_history[user_input]=result.content
    print(f'AI:{result.content}')

print(chat_history)
