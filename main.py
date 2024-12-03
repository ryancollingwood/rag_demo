from gpt4all import GPT4All
import gradio as gr
import json
from nomic import AtlasDataset
import requests
import os

# setup Atlas dataset
atlas_dataset_name = "nomic/example-text-dataset-news"
atlas_dataset_id = "4576c3a3-773d-4ba1-bf51-ff60734e4e00"
atlas_df = AtlasDataset(atlas_dataset_name).maps[0].data.df

# setup GPT4All LLM
model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
model = GPT4All(model_name)
system_prompt = "You are a helpful assistant. Use the following context to answer the user's question:"
max_tokens = 1024

# retrieval function
def retrieve(query, top_k=5):
    """Uses the Nomic Atlas API to retrieve data most similar to the query"""
    rag_request_payload = {
        "projection_id": atlas_dataset_id,
        "k": top_k,
        "query": query,
        "selection": { # temporary, selection param will soon be optional for this query endpoint
            "polarity": True,
            "method": "composition",
            "conjunctor": "ALL",
            "filters": [{"method": "search", "query": " ", "field": "text"}]
        }
    }
    
    rag_response = requests.post(
        "https://api-atlas.nomic.ai/v1/query/topk", 
        data=json.dumps(rag_request_payload), 
        headers={
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {os.environ['NOMIC_API_KEY']}"
        }
    )
    
    results = rag_response.json()
    formatted_results = ""
    for idx, data_id in enumerate(results['data'], 1):
        id_ = data_id['id_']
        matching_rows = atlas_df[atlas_df['id_'] == id_]
        for _, row in matching_rows.iterrows():
            formatted_results += f"Result {idx} (Atlas ID: {id_}):\n{row.text}\n\n"
    return formatted_results

# setup Gradio layout

css = """
.container {
    max-width: 1400px;
    margin: auto;
    padding: 20px;
}
.header {
    text-align: center;
    margin-bottom: 40px;
    padding: 20px;
    background-color: #f7f7f7;
    border-radius: 10px;
}
.chat-container {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    background-color: white;
}
.context-container {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    background-color: #f9f9f9;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="header"):
            gr.Markdown(
                "# RAG Demo\n"
                "## Powered by Atlas & GPT4ALL from Nomic\n"
                "This demo combines semantic search using the Atlas API with local LLM generation using GPT4ALL."
            )
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                with gr.Column(elem_classes="chat-container"):
                    gr.Markdown(f"### Chat Session")
                    gr.Markdown(f"LLM loaded: {model_name}")
                    chatbot = gr.Chatbot(
                        show_label=False,
                        container=True,
                        height=300
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            show_label=False,
                            container=False,
                            placeholder="Type your message here...",
                            scale=10
                        )
                        submit = gr.Button("Send", scale=1)
                    with gr.Row():
                        clear = gr.Button("Clear Chat")
                        
            with gr.Column(scale=1):
                with gr.Column(elem_classes="context-container"):
                    gr.Markdown("### Retrieved Context From Atlas")
                    gr.Markdown(f"Atlas data map: {atlas_dataset_name}")
                    context_display = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        lines=20
                    )

    def user(user_message, history: list):
        return "", history + [(user_message, None)]

    def bot(history, context: str):
        formatted_messages = [
            {
                'role': 'system', 
                'content': f"{system_prompt}\n\n{context}"
            }
        ]

        # Convert history to LLM format
        for user_msg, bot_msg in history:
            if user_msg:
                formatted_messages.append({'role': 'user', 'content': user_msg})
            if bot_msg:
                formatted_messages.append({'role': 'assistant', 'content': bot_msg})

        full_prompt = "\n".join([m['content'] for m in formatted_messages])
        
        # Get the last user message that doesn't have a response
        history[-1] = (history[-1][0], "")  # Initialize bot's response
        
        with model.chat_session():
            response = model.generate(
                full_prompt,
                max_tokens=max_tokens,
                streaming=True
            )
            for chunk in response:
                history[-1] = (history[-1][0], history[-1][1] + chunk)
                yield history

    def get_context(history):
        last_user_message = history[-1][0]
        context = retrieve(last_user_message)
        return context

    msg.submit(
        user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        get_context, [chatbot], context_display
    ).then(
        bot, [chatbot, context_display], chatbot
    )

    submit.click(
        user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        get_context, [chatbot], context_display
    ).then(
        bot, [chatbot, context_display], chatbot
    )
    
    clear.click(lambda: (None, ""), None, [chatbot, context_display], queue=False)

demo.launch()