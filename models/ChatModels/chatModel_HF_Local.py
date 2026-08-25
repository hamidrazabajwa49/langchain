from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

LLM=HuggingFacePipeline(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=LLM)

model.invoke("What is the capital of Pakistan?")
