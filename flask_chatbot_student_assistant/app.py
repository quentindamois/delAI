from flask import Flask, request
from llama_cpp import Llama
import re
import threading
import logging
import sys
from logging.handlers import RotatingFileHandler
import os

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

# Lock pour empêcher les accès concurrents aux modèles
model_lock = threading.Lock()


# Action handlers
def send_email_to_teacher(user_input: str, user_name: str) -> dict:
    """Send an email to a teacher with the student's question."""
    # TODO: Implement actual email sending logic (SMTP, API, etc.)
    logger.info(f"[ACTION] Sending email to teacher from {user_name}: {user_input[:100]}")
    return {
        "success": True,
        "action": "email_sent",
        "message": f"Email sent to the teacher with your question."
    }


def create_group_formation_request(user_input: str, user_name: str) -> dict:
    """Create a request for students to form groups."""
    # TODO: Implement group formation logic (database, notifications, etc.)
    logger.info(f"[ACTION] Creating group formation request by {user_name}: {user_input[:100]}")
    return {
        "success": True,
        "action": "groups_created",
        "message": "Group formation request created and sent to students."
    }


def retrieve_information_request(user_input: str, user_name: str) -> dict:
    """Create a request to retrieve information from RAG."""
    # TODO: Implement information retrieval logic
    logger.info(f"[ACTION] Creating information retrieval request by {user_name}: {user_input[:100]}")
    return {
        "success": True,
        "action": "information_requested",
        "message": "Information retrieval request sent to students."
    }


keyword_dictionnary = {
    "send_email":{"verb":["send", "sent", "write"], "noun":["teacher", "question", "help"]},
    "make_group":{"verb":["make", "form", "assemble"], "adj":["final project", "project", "presentation"], "noun":["group", "group", "team",]},
    "get_information":{"verb":["need", "want", "look for", "get", "know", "ask about"], "noun":["what", "information", "help", "info"]},
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

@app.route("/ask", methods=['POST'])
def answer_ask():
    user_input = request.form.get('user_input')
    user_name = request.form.get('user_name', 'User')
    memory_context = request.form.get('memory_context', '')
    user_context = request.form.get('user_context', '')
    
    logger.info(f"Request from {user_name}: {user_input[:50]}...")
    
    with model_lock:
        # Detect user intent using rule-based function
        detected_intent = get_intent(user_input.lower())
        action_result = None
        action_taken = False
        
        # Execute actions based on detected intent
        # Only trigger send_email if both verb AND noun are present
        if detected_intent == "send_email":
            detected_intent_pretty = "send an email"
            action_result = send_email_to_teacher(user_input, user_name)
            action_taken = True
        elif detected_intent == "make_group":
            detected_intent_pretty = "form groups"
            action_result = create_group_formation_request(user_input, user_name)
            action_taken = True
        elif detected_intent == "get_information":
            detected_intent_pretty = "retrieve information"
            action_result = retrieve_information_request(user_input, user_name)
            action_taken = True
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Quorum, a helpful bot assistant for students and teachers. "
                    "You only speak English. "
                    "Only provide personal details if explicitly listed in the provided user information. "
                    "You must never invent, assume, or fabricate schedules, plans, events, classes, or activities. "
                    "If this information is not explicitly provided, you must say you do not have it. "
                    "You must not add examples, guesses, or placeholders. "
                    "Do not answer with partially related or generic user details. "
                    "Be friendly and professional. "
                    "Keep your answers brief."
                    
                )
            }
        ]

        developer_content = []

        developer_content.append("Trusted user information:")
        developer_content.append(user_context)

        if memory_context:
            developer_content.append("\nRelevant conversation memory:")
            developer_content.append(memory_context)

        if action_taken:
            developer_content.append(
                f"\nDetected possible intent: {detected_intent_pretty}\n"
                "If the current user request matches this intent, perform it and confirm briefly. "
                "Otherwise, ignore this instruction."
            )

        messages.append({
            "role": "developer",
            "content": "\n".join(developer_content)
        })

        messages.append({
            "role": "user",
            "content": user_input
        })

        logger.info(messages)
        answer = llm.create_chat_completion(messages=messages)
    
    response_text = answer["choices"][0]["message"]["content"]
    
    # Format response
    if action_taken:
        return f"✅ You want to {detected_intent}. {response_text}"
    else:
        return response_text


if __name__ == '__main__':
    # Disable reloader to avoid subprocess buffering issues with logging
    app.run(debug=True, use_reloader=False)