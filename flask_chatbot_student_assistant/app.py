from flask import Flask, request
from llama_cpp import Llama
import re
import threading
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv
import faiss
import pickle
import numpy as np
from group_creator import create_group
from email_sender import (
    load_teachers,
    resolve_teacher_from_text,
    generate_email_draft,
    send_email_to_teacher,
)
load_dotenv()

# Configure logging to both console and file
log_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# File handler
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "app.log"),
    maxBytes=10485760,  # 10MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

"""
llm = Llama.from_pretrained(
	repo_id="hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF",
	filename="llama-3.2-1b-instruct-q8_0.gguf",
)
"""
try:
    llm = Llama(model_path="./models/llama-3.2-1b-instruct-q8_0.gguf", verbose=False)
except:
    llm = Llama(model_path="./models/ibm-granite_granite-3.3-2b-instruct-Q3_K_M.gguf", verbose=False)

# Load embedding model for RAG (same as in long_term_memory.py)
try:
    embedding_model = Llama(
        model_path="./models/nomic-embed-text-v1.5.Q4_K_M.gguf",
        embedding=True,
        verbose=False,
    )
except Exception as e:
    logger.warning(f"Could not load embedding model for RAG: {e}")
    embedding_model = None

# Load FAISS index and RAG database
rag_database_loaded = False
faiss_index = None
all_chunks = []
chunk_metadata = []

try:
    rag_dir = "./rag_database"
    if os.path.exists(os.path.join(rag_dir, "faiss_index.bin")):
        faiss_index = faiss.read_index(os.path.join(rag_dir, "faiss_index.bin"))
        
        with open(os.path.join(rag_dir, "chunks.pkl"), "rb") as f:
            all_chunks = pickle.load(f)
        
        with open(os.path.join(rag_dir, "metadata.pkl"), "rb") as f:
            chunk_metadata = pickle.load(f)
        
        rag_database_loaded = True
        logger.info(f"RAG database loaded successfully. Total chunks: {len(all_chunks)}")
    else:
        logger.warning("RAG database files not found in rag_database directory")
except Exception as e:
    logger.error(f"Error loading RAG database: {e}")
    rag_database_loaded = False

# Lock pour empêcher les accès concurrents aux modèles
DRAFT_LOCK_TIMEOUT = int(os.getenv("DRAFT_LOCK_TIMEOUT", "20"))
# Reentrant lock to allow nested acquisitions within a single request
model_lock = threading.RLock()

# Teacher directory and in-memory cache
TEACHERS_PATH = os.getenv("TEACHERS_PATH", "./data/teachers.json")
teachers, teacher_lookup = load_teachers(TEACHERS_PATH, logger)

# Pending intent state to enable confirmation on the next user message
pending_intent = {
    "intent": None,
    "pretty": None,
    "text": None,
    "user_name": None,
    "teacher": None,
}


def create_group_formation_request(user_input: str, user_name: str) -> dict:
    """Create a request for students to form groups."""
    logger.info(f"[ACTION] Creating group formation request by {user_name}: {user_input[:100]}")
    return create_group(user_input)


def search_rag(query: str, top_k: int = 2) -> list:
    """
    Search the RAG database for relevant documents.
    Returns a list of dicts with 'text' and 'source' fields.
    """
    if not rag_database_loaded or faiss_index is None or embedding_model is None:
        logger.warning("RAG database not available for search")
        return []
    
    try:
        # Get embedding for the query
        query_embedding = embedding_model.embed(query)
        query_emb = np.array([query_embedding], dtype=np.float32)
        
        # Search the FAISS index
        D, I = faiss_index.search(query_emb, top_k)
        
        results = []
        for idx in I[0]:
            if 0 <= idx < len(all_chunks):
                results.append({
                    "text": all_chunks[idx],
                    "source": chunk_metadata[idx].get("source", "Unknown") if idx < len(chunk_metadata) else "Unknown"
                })
        
        logger.info(f"RAG search found {len(results)} relevant documents for query: {query[:100]}")
        return results
    
    except Exception as e:
        logger.error(f"Error searching RAG database: {e}")
        return []


def retrieve_information_request(user_input: str, user_name: str) -> dict:
    """Retrieve information from RAG database."""
    logger.info(f"[ACTION] Retrieving information from RAG for {user_name}: {user_input[:100]}")
    
    # Search the RAG database
    rag_results = search_rag(user_input, top_k=2)
    
    if not rag_results:
        logger.warning(f"No relevant documents found in RAG for query: {user_input[:100]}")
        return {
            "success": False,
            "action": "information_not_found",
            "message": "No relevant documents found in the knowledge base.",
            "results": []
        }
    
    # Format results for the response
    formatted_results = []
    for result in rag_results:
        formatted_results.append({
            "text": result["text"],
            "source": result["source"]
        })
    
    logger.info(f"[SUCCESS] Retrieved {len(formatted_results)} documents from RAG")
    return {
        "success": True,
        "action": "information_retrieved",
        "message": f"Found {len(formatted_results)} relevant documents.",
        "results": formatted_results
    }
keyword_dictionnary = {
    "send_email":{"verb":["send", "sent", "write"], "noun":["teacher", "question", "help"]},
    "make_group":{"verb":["make", "form", "assemble"], "adj":["final project", "project", "presentation"], "noun":["group", "group", "team", "teams", "groups"]},
    "get_information":{
        "verb":["look for", "find", "search"],
        "noun":[
            "school", "campus", "calendar",
            "file", "document", "doc", "docs", "documentation", "pdf", "guide", "policy",
            "assignment", "brief", "instructions", "manual"
        ]
    },
}

convert_morph_letter = lambda a: r"\s+" if a == " "  else rf"[{a.lower()}{a.upper()}]+"

def convert_word_regex(sentence):
    return rf"\b{r''.join(list(map(convert_morph_letter, map(lambda b: b[0], filter(lambda a:  sentence[a[1]] == a[0], zip(sentence, range(len(sentence))))))))}\b"

def create_intent_regex(intent):
    tem = list()
    for type_word in keyword_dictionnary[intent].keys():
        tem_list = list(map(lambda a : convert_word_regex(a), keyword_dictionnary[intent][type_word]))
        tem.append(r"|".join(tem_list))
    keyword_dictionnary[intent] = r".*(:?" + "|".join(list(map(lambda c: rf"(?:{c})", map(lambda b: r"|".join(list(map(lambda a : r"(?:" + a + r")", tem[b:] +  tem[:b]))), range(len(tem)))))) + r").*"

def gen_regex():
    for intent in keyword_dictionnary.keys():
        create_intent_regex(intent)


gen_regex()

def get_intent(text):
    list_detected_intent =  list(filter(lambda a: re.search(keyword_dictionnary[a], text), keyword_dictionnary.keys()))
    logger.info(f"Intent detected: {list_detected_intent}")
    return "Unknown" if len(list_detected_intent) == 0 else list_detected_intent[0]


app = Flask(__name__)
# Configure Flask's logger to also output to console
app.logger.addHandler(console_handler)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

@app.route("/")
def hello_world():
    return "Hello world"

@app.route("/summarize", methods=['POST'])
def summarize_context():
    """Endpoint dedicated to summarizing memory context."""
    context = request.form.get('context', '')
    user_name = request.form.get('user_name', 'User')
    
    logger.info(f"Summarization request from {user_name}: {len(context)} chars")
    
    summarization_prompt = (
        f"Please create a SINGLE paragraph that encapsulates the following conversation context, "
        f"highlighting the user's key preferences, previous questions, and important details in this single paragraph. "
        f"Keep it concise and under 500 characters.\n\n"
        f"Context:\n{context}"
    )
    
    with model_lock:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful summarization assistant. Provide concise, clear summaries."
            },
            {
                "role": "user",
                "content": summarization_prompt
            }
        ]
        
        logger.info(f"Summarizing context for {user_name}")
        answer = llm.create_chat_completion(messages=messages)
    
    summary = answer["choices"][0]["message"]["content"]
    logger.info(f"Summarization complete: {len(context)} chars -> {len(summary)} chars")
    
    return summary

@app.route("/ask", methods=['POST'])
def answer_ask():
    user_input = request.form.get('user_input')
    user_name = request.form.get('user_name', 'User')
    memory_context = request.form.get('memory_context', '')
    user_context = request.form.get('user_context', '')
    
    logger.info(f"Request from {user_name}: {user_input}")
    
    with model_lock:
        global pending_intent

        action_taken = False
        action_result = None
        short_circuit_response = None

        confirm_yes = {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "confirm"}
        confirm_no = {"no", "n", "nope", "cancel", "stop"}

        def normalize_reply(text: str) -> str:
            # strip punctuation and extra spaces for more permissive matching
            if not text:
                return ""
            return re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()

        normalized_input = normalize_reply(user_input)

        # If we had a pending intent from the previous message, act only after explicit confirmation
        if pending_intent["intent"]:
            if normalized_input in confirm_yes:
                logger.info(f"TAKING ACTION AFTER CONFIRMATION: {pending_intent['pretty']}")
                confirmed_intent_pretty = pending_intent["pretty"]
                if pending_intent["intent"] == "send_email":
                    if not pending_intent.get("text"):
                        # Guardrail: never send an empty or missing draft
                        pending_intent = {"intent": None, "pretty": None, "text": None, "user_name": None, "teacher": None}
                        return "I don't have an email draft to send. Please repeat the question and I'll draft an email for you."
                    if not pending_intent.get("teacher"):
                        pending_intent = {"intent": None, "pretty": None, "text": None, "user_name": None, "teacher": None}
                        return "I need the teacher's name before I can send the email. Please try again."
                    action_result = send_email_to_teacher(pending_intent["text"], pending_intent["user_name"], pending_intent.get("teacher"), logger)
                    action_taken = True
                elif pending_intent["intent"] == "make_group":
                    action_result = create_group_formation_request(pending_intent["text"], pending_intent["user_name"])
                    action_taken = True
                elif pending_intent["intent"] == "get_information":
                    action_result = retrieve_information_request(pending_intent["text"], pending_intent["user_name"])
                    action_taken = True
                # Clear pending intent after handling
                pending_intent = {"intent": None, "pretty": None, "text": None, "user_name": None, "teacher": None}
            elif normalized_input in confirm_no:
                logger.info("Confirmation denied. Clearing pending intent.")
                pending_intent = {"intent": None, "pretty": None, "text": None, "user_name": None, "teacher": None}

        # Detect user intent in the current message to set up confirmation for the next turn
        detected_intent = get_intent(user_input.lower())
        intent_val = False
        detected_intent_pretty = None
        pending_candidate = None
        
        # Execute actions based on detected intent
        # Only trigger send_email if both verb AND noun are present
        if detected_intent == "send_email":
            detected_intent_pretty = "send an email"
            intent_val = True
        elif detected_intent == "make_group":
            detected_intent_pretty = "form groups"
            intent_val = True
        elif detected_intent == "get_information":
            detected_intent_pretty = "retrieve information"
            intent_val = True

        # Prepare pending intent payload for the next user confirmation
        if intent_val:
            pending_candidate = {
                "intent": detected_intent,
                "pretty": detected_intent_pretty,
                "text": user_input,
                "user_name": user_name,
                "teacher": None,
            }
            # For send_email, generate the draft immediately and bypass the general LLM reply
            if detected_intent == "send_email" and not action_taken:
                teacher_match, teacher_matches = resolve_teacher_from_text(user_input, teacher_lookup)
                if not teacher_matches:
                    short_circuit_response = (
                        "Please mention the teacher's name so I can send it to the right person."
                    )
                    pending_candidate = None
                    intent_val = False
                elif len(teacher_matches) > 1:
                    names = ", ".join([t.get("name", "(unknown)") for t in teacher_matches if t])
                    short_circuit_response = (
                        f"I found multiple teachers matching that request: {names}. Which one should I email?"
                    )
                    pending_candidate = None
                    intent_val = False
                else:
                    teacher = teacher_match
                    pending_candidate["teacher"] = teacher
                    teacher_label = teacher.get("name", "your teacher")
                    try:
                        draft = generate_email_draft(llm, model_lock, user_input, user_name, teacher, logger, DRAFT_LOCK_TIMEOUT)
                        pending_candidate["text"] = draft
                        short_circuit_response = (
                            "Here is your draft email:\n\n"
                            f"{draft}\n\n"
                            f"Send this to {teacher_label}?"
                        )
                    except TimeoutError:
                        logger.error("Draft generation timed out; not storing pending intent")
                        short_circuit_response = (
                            "Sorry, I'm busy right now and couldn't create the email draft. "
                            "Please try again in a moment."
                        )
                        pending_candidate = None
                        intent_val = False
                    except Exception as e:
                        logger.error(f"Draft generation failed: {e}")
                        short_circuit_response = (
                            "Sorry, I couldn't create the email draft. Please try again."
                        )
                        pending_candidate = None
                        intent_val = False
        
        system_content = [
                    "You are Quorum, a helpful bot assistant for students and teachers. ",
                    "Your main goal is to perform tasks for students based on their requests. You must ask user confirmation before performing a task when an intent is detected. ",
                    "You must confirm when an action has been done successfully or not. ",
                    "You only speak English. ",
                    "Only provide personal details if explicitly listed in the provided user information. ",
                    "You must never invent, assume, or fabricate schedules, plans, events, classes, or activities. ",
                    "If this information is not explicitly provided, you must say you do not have it. ",
                    "Never add examples, guesses, or placeholders. ",
                    "Do not answer with partially related or generic user details. ",
                    "Be friendly and professional. ",
                    "Keep your answers brief."
                   ]
        
        

        developer_content = []

        if not action_taken:
            developer_content.append(user_context)

        if memory_context and not action_taken:
            developer_content.append(memory_context)

        if intent_val:
            developer_content.append(
                    f"\nDETECTED INTENT: {detected_intent_pretty}\n"
                    "When an intent is detected, do not answer the content of the user's message. "
                    "Ask ONLY: 'Do you want to proceed with this action?' "
                    "Do NOT state that any action was executed. Do NOT assume the user confirmed. Do NOT add other text."
                )

        # If an email action was just handled, return a fixed response without LLM
        if action_taken and action_result is not None and action_result.get('action') in ['email_sent', 'email_failed']:
            short_circuit_response = (
                f"I successfully sent the email to {action_result.get('to', 'unknown recipient')}" if action_result.get('success') else
                f"I couldn't send the email due to error: {action_result.get('message', 'unknown error')}"
            )

        if action_taken and action_result is not None and action_result.get('action') in ['group_created', 'group_failed']:
            short_circuit_response = " ".join(action_result["message"].split(" ")[1:])
            short_circuit_response += " \nLet me know if you need me to send it to your teacher !"

        if action_taken and action_result is not None:

            system_content.append(
                f"\nAction taken: {action_result['action']}. "
                f"Result message: {action_result['message']}. "
                f"Success: {action_result['success']}. "
            )
            # If it's a RAG retrieval, add the results context
            if action_result.get('action') == 'information_retrieved' and action_result.get('results'):
                rag_context = "Here are the relevant documents from the knowledge base:\n\n"
                for i, result in enumerate(action_result['results'], 1):
                    rag_context += f"Document {i} (Source: {result['source']}):\n{result['text']}\n\n"
                system_content.append(rag_context)
            # For email actions, add from/to info
            if action_result.get('action') in ['email_sent', 'email_failed']:
                system_content.append(
                    f"From: {action_result.get('from', 'N/A')}. "
                    f"To: {action_result.get('to', 'N/A')}. "
                )
            
            messages = [
                {
                "role": "system",
                "content": "\n".join(system_content)
                }
            ]
        else:
            messages = [
                {
                "role": "system",
                "content": "\n".join(system_content)
                }
            ]
            messages.append({
            "role": "developer",
            "content": "\n".join(developer_content)
            })
            messages.append({
            "role": "user",
            "content": user_input
            })
        

        if short_circuit_response is None:
            logger.info(messages)
            answer = llm.create_chat_completion(messages=messages)
            response_text = answer["choices"][0]["message"]["content"]
            logger.info(f"LLM response generated for {user_name}: {response_text}")
        else:
            response_text = short_circuit_response

    # Extract drafted email for send_email intent so we can send the refined content on confirmation
    email_draft = None
    if intent_val and detected_intent == "send_email":
        # In the short-circuit flow, the draft is generated directly and stored
        if pending_candidate and pending_candidate.get("text"):
            email_draft = pending_candidate["text"]
        else:
            logger.info("No generated draft present; prompting user to retry")
            return (
                "I could not capture the email draft. "
                "Please ask again and I will provide the draft for your confirmation."
            )

    # Persist pending intent with the drafted email (when applicable) for the next confirmation step
    if intent_val and not action_taken and pending_candidate:
        pending_intent = pending_candidate
        if detected_intent == "send_email" and email_draft:
            pending_intent["text"] = email_draft
            logger.info("Stored drafted email for confirmation and sending")
        elif detected_intent == "send_email":
            # No draft parsed; block sending until a proper draft is produced
            pending_intent["text"] = None
    
    # Format response
    if intent_val:
        return response_text
    else:
        return response_text


if __name__ == '__main__':
    # Disable reloader to avoid subprocess buffering issues with logging
    # Listen on 0.0.0.0 to allow connections from other containers/services
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)