from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv
import gradio as gr
from utilis import retrieve, llm

load_dotenv()
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
model=ChatMistralAI(
    model="mistral-large-latest",
    api_key=MISTRAL_API_KEY
)


with gr.Blocks(css=".gradio-container {background: url('image.png') !important;}") as Iface:
    gr.Markdown("<h1>Find Shayari</h1>")
    
    with gr.Row():
        with gr.Column(scale=1):
            query = gr.Textbox(label="Which type of shayari do you want to find?")
            submit_button = gr.Button("Search")
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Matching results", height=700)      
    submit_button.click(
        fn=llm,
        inputs=[query, chatbot],
        outputs=[chatbot, chatbot]

    )
Iface.launch()