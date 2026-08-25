from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model=ChatAnthropic(model='claude-3.5-sonnet-20241012',temperature=1.0,max_completion_tokens=50)

result = model.invoke("What is the capital of Pakistan?")

print(f'\n{result.content}')
