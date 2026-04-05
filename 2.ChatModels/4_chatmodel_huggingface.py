from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task='text-generation',
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result.content)
