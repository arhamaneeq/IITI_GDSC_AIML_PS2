# GDSC AI&ML Induction Task 2
## Chatbot with RAG + Voice Features
**PROBLEM STATEMENT**
Create an intelligent chatbot that integrates Retrieval-Augmented Generation (RAG) for answering queries and Voice Interaction for a hands-free experience. The chatbot will retrieve relevant information from an external knowledge base and then generate repsonses based on the retrieved context.
**Key Feature Objectives**
- *Retrieval Augmented Generation:* Integrate a search or retrieval model that pulls in relvant documents or data from a knowledge base and then uses a language model to generate human like-responses.
- *Voice Interface:* Implement speech-to-text and text-to-speech for voice interaction, allowing the user to speak to the chatbot and get audio responses.

## File Structure
- `📂chat_sessions`
- `📂models`
    - `📂llava`
        - `.gitkeep`
    - `.gitkeep`
- `.gitignore`
- `requirements.txt`
- `README.md`
- `raggy.py`
- `raggy_v2.py`
- `llms_chain.py`
- `prompt_templates.py`
- `utils.py`
- `audio_handler.py`
- `image_handler.py`
- `pdf_handler.py`

### Installation
Set up a local virtual environment by navigating to the root folder and running
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
You will also need to install `ffmpeg` via `choco`
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