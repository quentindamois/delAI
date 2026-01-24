import requests
import streamlit as st

import os
import sys
import logging

import httpx
from dotenv import load_dotenv
load_dotenv()


# Configure logging to output immediately
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# main.py - modifier ces lignes
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:5000/ask")
logger.info(f"Using LLM ENDPOINT: {LLM_ENDPOINT}")
SUMMARIZATION_ENDPOINT = os.getenv("SUMMARIZATION_ENDPOINT", "http://127.0.0.1:5000/summarize")



def fetch_llm_response(message_text: str, user_name: str, user_id: str) -> str:
    # Increased timeout to 300 seconds for LLM responses (especially for RAG queries)
    response = requests.post(
            LLM_ENDPOINT,
            timeout=300,
            data={
                "user_input": message_text,
                "user_name": user_name,
                "user_id":user_id,
            },
        )
    return response.text

def gen_answer(text):
    user_id = str(999999999)
    display_name = "testUser"

    # Try to extract and store user info from the message
    
    response_text = fetch_llm_response(text, display_name, user_id)

    return response_text




st.title("test interface")

historic_actual_message = list()

prompt = st.chat_input("Enter what you want.")
if prompt:
    historic_actual_message.append(prompt)
    tem = gen_answer(prompt)
    historic_actual_message.append(tem)




ai_message = st.chat_message("ai")
user_message = st.chat_message("user")
tuple_message = (user_message, ai_message)

for i in range(len(historic_actual_message)):
    tuple_message[i % 2].write(historic_actual_message[i])