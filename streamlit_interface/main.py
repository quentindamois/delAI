import requests
import streamlit as st

import os
import sys
import logging

import httpx
from dotenv import load_dotenv
load_dotenv()


from conversation_logger import (
    init_csv,
    log_user_message,
    log_bot_message,
    get_last_interactions
)
from user_context import (
    init_user_file,
    get_user_info,
    format_user_context,
    update_user_info_from_message
)

# Configure logging to output immediately
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

from long_term_memory import retrieve_relevant_memories, format_long_term_memory


# main.py - modifier ces lignes
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:5000/ask")
logger.info(f"Using LLM ENDPOINT: {LLM_ENDPOINT}")
SUMMARIZATION_ENDPOINT = os.getenv("SUMMARIZATION_ENDPOINT", "http://127.0.0.1:5000/summarize")

MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "400"))

def summarize_memory_context(context: str, display_name: str) -> str:
    """
    Use the generative model to summarize memory context if it exceeds MAX_CONTEXT_LENGTH.
    Returns the original context if it's short enough, or a summary otherwise.
    """
    if len(context) <= MAX_CONTEXT_LENGTH:
        return context
    
    logger.info(
        "Context too long (%d chars), generating summary for user %s",
        len(context),
        display_name
    )
    
    try:
        response = requests.post(
                SUMMARIZATION_ENDPOINT,
                data={
                    "context": context,
                    "user_name": display_name,
                },
            )
        response.raise_for_status()
        summary = response.text
        logger.info("Context summarized: %d chars -> %d chars", len(context), len(summary))
        return summary
    except Exception as exc:
        logger.warning(
            "Failed to summarize context (%s), using original",
            exc.__class__.__name__
        )
        return context


def format_interaction_blocks(interactions: list[tuple[str, str]]) -> str:
    blocks = []
    for user_msg, bot_msg in interactions:
        blocks.append(
            f"User message: {user_msg}\n"
            f"Agent response: {bot_msg}"
        )
    return "\n\n".join(blocks)

def format_short_term_memory(interactions: list[tuple[str, str]]) -> str:
    if not interactions:
        return ""

    content = format_interaction_blocks(interactions)

    return (
        "Here are the most recent interactions you had with the user.\n"
        f"{content}\n\n"
    )



def fetch_llm_response(message_text: str, user_id: str, display_name: str) -> str:
    short_term_interactions = get_last_interactions(user_id)
    short_term_context = format_short_term_memory(short_term_interactions)

    long_term_memories = retrieve_relevant_memories(user_id, message_text)
    long_term_context = format_long_term_memory(long_term_memories)

    memory_context = f"{short_term_context}{long_term_context}"

    user_context = format_user_context(get_user_info(user_id))

    # Summarize memory context if it's too long
    memory_context = summarize_memory_context(memory_context, display_name)

    print("Memory context sent to LLM:", memory_context)
    
    # Increased timeout to 300 seconds for LLM responses (especially for RAG queries)
    response = requests.post(
            LLM_ENDPOINT,
            timeout=300,
            data={
                "user_input": message_text,
                "user_name": display_name,
                "memory_context": memory_context,
                "user_context": user_context,
            },
        )
    st.write(response)
    return response.text

def gen_answer(text):
    user_id = str(999999999)
    display_name = "testUser"

    # Try to extract and store user info from the message
    """
    update_user_info_from_message(
        user_id,
        display_name,
        text
    )
    """
    logger.info(f"Message from {display_name}: {text}")
    """
    # Log message utilisateur
    log_user_message(
        user_id,
        text
    )"""
    response_text = fetch_llm_response(text, user_id, display_name)
    """
    # Log réponse du bot
    log_bot_message(
        user_id,
        response_text
    )"""
    return response_text




st.title("test interface")

historic_actual_message = list()

prompt = st.chat_input("Enter what you want.")
if prompt:
    historic_actual_message.append(prompt)
    tem = gen_answer(prompt)
    st.write(tem)
    st.write(type(tem))
    st.write(help(tem))
    historic_actual_message.append(tem)




ai_message = st.chat_message("ai")
user_message = st.chat_message("user")
tuple_message = (user_message, ai_message)

for i in range(len(historic_actual_message)):
    tuple_message[i % 2].write(historic_actual_message[i])