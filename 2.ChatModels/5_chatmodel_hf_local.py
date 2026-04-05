from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='google/gemma-4-31B-it',
    task='text_generation',
    pipeline_kwargs = dict(
        temperature=0.5,
        max__new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke('What is the capital of India?')

print(result.content)