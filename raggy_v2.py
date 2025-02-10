# IMPORTS
# streamlit stuff
import streamlit as st
from streamlit_mic_recorder import mic_recorder

# my util files
from llms_chain import load_normal_chain
from utils import save_chat_history_json, get_timestamp, load_chat_history_json
from audio_handler import transcribe_audio
from image_handler import handle_image

# langchain stuff
from langchain.memory import StreamlitChatMessageHistory

# file handling stuff
import os
import yaml

# READ CONFIG FILE

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# EXTRACT OUT SOME STUFF

def load_chain(chat_history):
    return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def track_index():
    st.session_state.session_tracker = st.session_state.session_key

def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.new_session_key)
        else:
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.session_key)

# MAIN

def main():
    # DEFINE AND INITIALISE SESSION STATES
    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    # DEFINE INTERFACE
    # Heading
    st.title("Raggy! 🤖")
    st.write("Raggy is here to h-")

    # Declare Chat Container
    chat_container = st.container()

    # Declare Sidebar Stuff
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + os.listdir(config["chat_history_path"])
    # Selectbox Behaviour for Loading Previous Conversations    
    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a Chat Session", chat_sessions, key="session_key", on_change=track_index)
    # Image Upload
    uploaded_image = st.sidebar.file_uploader("Upload an image file: ", type=["jpg", "jpeg", "png"])


    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(config["chat_history_path"] + st.session_state.session_key)
    else:
        st.session_state.history = []


    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    user_input = st.text_input("Ask Raggy anything!", key="user_input", on_change=set_send_input)

    voice_col, send_col = st.columns(2)

    with voice_col:
        voice_recording = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", just_once=True)
    with send_col:
        send_button = st.button("Send", key="send_button", on_click=clear_input_field)


    # SEND DATA TO LLM
    #print(voice_recording)
    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        llm_chain.run(transcribed_audio)

    if (send_button or st.session_state.send_input):
        if (uploaded_image):
            with st.spinner("Processing Image"):
                user_message = "Describe this image in detail please."

                if st.session_state.user_question != "":
                    user_message = st.session_state.user_question
                    st.session_state.user_question = ""

                llm_answer = handle_image(uploaded_image.getvalue(), user_message)  
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_answer["choices"][0]["message"]["content"])  

        if (st.session_state.user_question != ""):
            llm_response = llm_chain.run(st.session_state.user_question)
            # st.chat_message("ai").write(llm_response)
            st.session_state.user_question = ""

    # RETRIEBE DATA FROM LLM & DISPLAY
    if chat_history.messages != []:
        with chat_container:
            st.write("Chat History:")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

    save_chat_history()

if __name__ == "__main__":
    main()