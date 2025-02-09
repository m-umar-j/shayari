from langchain_mistralai import MistralAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "shayari" 

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index)



