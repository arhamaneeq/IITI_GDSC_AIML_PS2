# GDSC AI&ML Induction Task 2
## Chatbot with RAG + Voice Features
**PROBLEM STATEMENT**
Create an intelligent chatbot that integrates Retrieval-Augmented Generation (RAG) for answering queries and Voice Interaction for a hands-free experience. The chatbot will retrieve relevant information from an external knowledge base and then generate repsonses based on the retrieved context.
**Key Feature Objectives**
- *Retrieval Augmented Generation:* Integrate a search or retrieval model that pulls in relvant documents or data from a knowledge base and then uses a language model to generate human like-responses.
- *Voice Interface:* Implement speech-to-text and text-to-speech for voice interaction, allowing the user to speak to the chatbot and get audio responses.

## File Structure
- `📂chat_sessions`
    - `.gitignore`
- `📂models`
    - `📂llava`
        - `.gitkeep`
    - `.gitkeep`
- `.gitignore`
- `requirements.txt`
- `README.md`
- `config.yaml`
- `raggy.py`
- `raggy_v2.py`
- `llms_chain.py`
- `prompt_templates.py`
- `utils.py`
- `audio_handler.py`
- `image_handler.py`
- `pdf_handler.py`

The main program here is `raggy_v2.py`, which mostly contains the frontend logic. All processing is relegated to other scripts. `llms_chain.py` handles logic for the chat itself, whereas `audio_handler.py`, `image_handler.py`, and `pdf_handler.py` are responsible for converting raw input data from streamlit into formats readble by `llms_chain.py`. `utils.py` handles logic for storage and loading of previous chats via JSON files in the `chat_sessions` folder. We also include a `config.yaml` file which includes information regarding the models used.

### Installation
Clone the repository and then set up a local virtual environment by navigating to the root folder and running
```bash
python -m venv .venv
```
Then activate the virtual environment by running
```bash
./.venv/Scripts/Activate
```
Use `pip` to install most dependencies
```bash
pip install -r requirements.txt
```
You will also need to install `ffmpeg` via `choco`. Open Windows Powershell as an Administrator and run
```bash
choco install ffmpeg -y
```
#### Models
You will need to install the models manually from *HuggingFace*, the models used for my development environment are
- `📂models`
    - `📂llava`
        - `ggml-model-q4_k.gguf`
        - `mmproj-model-f16.gguf`
    - `mistral-7b-instruct-v0.1.Q3_K_M.gguf`
    - `mistral-7b-instruct-v0.1.Q5_K_M.gguf`
### Run
With the virtual environment activated, run the program by writing
```bash
streamlit run raggy_v2.py
```