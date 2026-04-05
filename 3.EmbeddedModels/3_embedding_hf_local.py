from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    'Delhi is capital of India',
    'Mumbai is the capital of Maharashtra',
    'Washington D.C. is the capital of America'
]

vector = embedding.embed_documents(text=documents)

print(str(vector))