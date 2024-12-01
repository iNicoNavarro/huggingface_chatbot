from transformers import pipeline
import gradio as gr

model = pipeline(
    "summarization", 
    device=-1
)

def predict(prompt):
    return model(prompt)[0]["summary_text"]

with gr.Blocks() as demo:
    textbox = gr.Textbox(
        placeholder="Enter text block to summarize",
        lines=4
    )
    gr.Interface(fn=predict, inputs=textbox, outputs="text")

    demo.launch()