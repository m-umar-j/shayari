import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_together import TogetherEmbeddings
from langchain_mistralai import ChatMistralAI
import cohere
load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
TOGETHER_API=os.getenv("TOGETHER_API")
COHERE_API_KEY=os.getenv("COHERE_API_KEY")

co = cohere.Client(COHERE_API_KEY)

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

def retrieve(query):
    print(1)
    docs=vector_store.similarity_search(query, k=4)
    print(f"length of docs is {len(docs)}")
    results = co.rerank(query=query, documents=docs, top_n=3, model='rerank-v3.5')
    
    return results

def llm(query, history):
    print(3)
    query=model.invoke(f"You are a helpful AI assistant. Translate the following query to urdu if the query is in urdu then return the same query. Query: {query}. Remember just translate the query and nothing else.").content
    results=retrieve(query)
    prompt=f""""
    You are a helpful AI Assistant and your job is to help user find relevant shayari.
    The query is {query}
    The matching result after semantic search is {results[0].page_content}. You may exclude interpretation given in the results. Just respond from the context.
    """

    try:
        response=model.invoke(prompt).content
        history = history or []
        history.append(("User", query))
        history.append(("Bot", results[0].page_content))
        print(4)
        return history, history
    except Exception as e:
        return history, history + [("Bot", f"Error in API request: {str(e)}")]

