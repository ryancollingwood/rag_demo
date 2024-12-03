# Nomic RAG Demo

Minimal RAG (retrieval-augmented generation) demo in Python using the [Atlas](atlas.nomic.ai) API for semantic search and [GPT4All](nomic.ai/gpt4all) for local LLM generation.

## Running the demo

1. Clone this repository
2. Install the requirements
    ```bash
    pip install -r requirements.txt
    ```
2. Set your Nomic API key (which you can get by creating a free Nomic account [here](https://nomicai-production.us.auth0.com/u/signup?state=hKFo2SAzVjBPaWlUNGZGV2xIcFAta3BUTXVsZmNTV0RTemJsT6Fur3VuaXZlcnNhbC1sb2dpbqN0aWTZIEpjSEVEZVkyakNGaWs3ajUyNm1uemxxNkNUeGc5ZnVko2NpZNkgVkY0MURxZEV5UzJBYXE2NHExSW9PMUVPemRwanBsbnY)):
    ```bash
    export NOMIC_API_KEY='your-nomic-api-key-here'
    ```
3. Run the application:
    ```bash
    python main.py
    ```

