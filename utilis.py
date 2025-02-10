import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_together import TogetherEmbeddings
from langchain_mistralai import ChatMistralAI
import cohere
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank


load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
TOGETHER_API=os.getenv("TOGETHER_API")
COHERE_API_KEY=os.getenv("COHERE_API_KEY")

embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    api_key=TOGETHER_API
)

model=ChatMistralAI(
    model="mistral-large-latest",
    api_key=MISTRAL_API_KEY
)
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "together" 

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever()

compressor = CohereRerank(model="rerank-v3.5")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

def retrieve(query):
    compressed_docs = compression_retriever.invoke(query)
    
    return compressed_docs[0]

def llm(query, history):
    query=model.invoke(f"You are a helpful AI assistant. Translate the following query to urdu if the query is in urdu then return the same query. Query: {query}. Remember just translate the query and nothing else.").content
    results=retrieve(query)
    prompt=f""""
    You are a helpful AI Assistant and your job is to help user find relevant shayari.
    The query is {query}
    The matching result after semantic search is {results.page_content}. You may exclude interpretation given in the results. Just respond from the context.
    """

    try:
        response=model.invoke(prompt).content
        history = history or []
        history.append(("You", query))
        history.append(("Assistant", results.page_content))
        return history, history
    except Exception as e:
        return history, history + [("Bot", f"Error in API request: {str(e)}")]

