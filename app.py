from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv
import gradio as gr


load_dotenv()
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
model=ChatMistralAI(
    model="mistral-large-latest",
    api_key=MISTRAL_API_KEY
)
def llm(query):
    prompt=f"""
        You are a helpful AI Assistant. Your task is to help user find relevant lines of urdu shayari. Do not generate any new poetry, Just give already existing poetry.
        Also mention who is the poet of the lines. 
        The query of the user is : {query}
        You are supposed to return the answer in urdu.
    """
    return model.invoke(prompt).content

with gr.Blocks() as Iface:
    gr.Markdown("<h1> Find Shayari <h1>")
    
    with gr.Column():        
        query=gr.Textbox(label="Which type of shayari do you want to find?")
        submit_button=gr.Button("Search")
        
    with gr.Column():
        response=gr.Textbox(label="Matching results", scale=1)
        
        
    submit_button.click(
        fn=llm,
        inputs=[query],
        outputs=[response]

    )
Iface.launch()