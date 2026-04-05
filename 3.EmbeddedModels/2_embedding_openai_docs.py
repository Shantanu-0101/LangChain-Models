from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32) 

documents = [
    'Delhi is capital of India',
    'Mumbai is the capital of Maharashtra',
    'Washington D.C. is the capital of America'
]

result = embedding.embed_documents('Delhi is the capital of India')

print(str(result))