from gpt4all import GPT4All
import gradio as gr
import json
from nomic import AtlasDataset
import requests
import os

# setup GPT4All LLM
model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
model = GPT4All(model_name)
system_prompt = "You are a helpful assistant. Use the following context to answer the user's question:"
max_tokens = 1024

def get_projection_id(org_name, data_name) -> str:
  data_response = requests.get(
      f"https://api-atlas.nomic.ai/v1/project/{org_name}/{data_name}", 
      headers={
          "Content-Type": "application/json", 
          "Authorization": f"Bearer {os.environ['NOMIC_API_KEY']}"
      }
  )
  return data_response.json()["atlas_indices"][0]["projections"][0]["id"]

# retrieval function
# temp: use data DF to convert data ids to data text values 
# (not currently returned from v1/query/topk)
def retrieve_with_state(query, proj_id, df, top_k=5):
    """Uses the Nomic Atlas API to retrieve data most similar to the query"""
    if proj_id is None or df is None:
        return "Please load a dataset first."
            
    rag_request_payload = {
        "projection_id": proj_id,
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
    
    # temp: use data DF to convert data ids to data text values 
    # (not currently returned from v1/query/topk)
    results = rag_response.json()
    formatted_results = ""
    for idx, data_id in enumerate(results['data'], 1):
        id_ = data_id['id_']
        matching_rows = df[df['id_'] == id_]
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

def load_atlas_data(dataset_name):
    """Load Atlas dataset and return the dataframe"""
    try:
        return AtlasDataset(dataset_name).maps[0].data.df
    except Exception as e:
        return None


with gr.Blocks(css=css) as demo:
    # Add state variables to store across sessions
    atlas_df_state = gr.State(None)
    projection_id_state = gr.State(None)
    
    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="header"):
            gr.Markdown("# RAG Demo\n## Powered by Atlas & GPT4ALL from Nomic")
            with gr.Row():
                org_name = gr.Textbox(
                    label="Organization Name",
                    placeholder="e.g. nomic",
                    scale=1
                )
                dataset_name = gr.Textbox(
                    label="Dataset Name",
                    placeholder="e.g. example-text-dataset-news",
                    scale=2
                )
                load_button = gr.Button("Load Dataset", scale=1)
            status_message = gr.Markdown("")

        
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
                    context_display = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        lines=20
                    )

    def load_dataset(org, dataset):
        """Load the dataset and update state variables"""
        if not org or not dataset:
            return None, None, "⚠️ Please enter both organization and dataset names"
        
        full_dataset_name = f"{org}/{dataset}"
        try:
            # Load the Atlas dataset
            df = load_atlas_data(full_dataset_name)
            if df is None:
                return None, None, "❌ Failed to load dataset"
            
            # Get projection ID
            proj_id = get_projection_id(org, dataset)
            
            return df, proj_id, f"✅ Successfully loaded dataset: {full_dataset_name}"
        except Exception as e:
            return None, None, f"❌ Error: {str(e)}"


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

    def get_context(history, proj_id, df):
        if not history:
            return ""
        return retrieve_with_state(history[-1][0], proj_id, df)

    
    load_button.click(
        load_dataset,
        inputs=[org_name, dataset_name],
        outputs=[atlas_df_state, projection_id_state, status_message]
    )

    msg.submit(
        user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        get_context, [chatbot, projection_id_state, atlas_df_state], context_display
    ).then(
        bot, [chatbot, context_display], chatbot
    )

    # Update the submit button click handler similarly
    submit.click(
        user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        get_context, [chatbot, projection_id_state, atlas_df_state], context_display
    ).then(
        bot, [chatbot, context_display], chatbot
    )
    
    clear.click(lambda: (None, ""), None, [chatbot, context_display], queue=False)

demo.launch()