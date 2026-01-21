from flask import Flask, request
from llama_cpp import Llama
import re
import threading
import logging
import sys
from logging.handlers import RotatingFileHandler
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

# Pending intent state to enable confirmation on the next user message
pending_intent = {
    "intent": None,
    "pretty": None,
    "text": None,
    "user_name": None,
}


# Action handlers
def send_email_to_teacher(user_input: str, user_name: str) -> dict:
    """Send an email to a teacher with the student's question."""
    logger.info(f"[ACTION] Sending email to teacher from {user_name}: {user_input[:100]}")
    
    # Email configuration (use environment variables in production)
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')  # e.g., smtp.gmail.com, smtp.office365.com
    smtp_port = int(os.getenv('SMTP_PORT', '587'))  # 587 for TLS, 465 for SSL
    sender_email = os.getenv('SENDER_EMAIL', 'your-email@example.com')
    sender_password = os.getenv('SENDER_PASSWORD', 'your-password')
    teacher_email = os.getenv('TEACHER_EMAIL', 'teacher@example.com')
    
    try:
        # Create message
        message = MIMEMultipart('alternative')
        message['Subject'] = f'Question from student: {user_name}'
        message['From'] = sender_email
        message['To'] = teacher_email
        
        # Email body
        text_content = f"""
Hello,

You have received a question from student {user_name}:

{user_input}

---
This email was sent automatically by the Quorum student assistant bot.
        """
        
        html_content = f"""
<html>
  <body>
    <p>Hello,</p>
    <p>You have received a question from student <strong>{user_name}</strong>:</p>
    <blockquote style="margin: 20px; padding: 10px; background-color: #f5f5f5; border-left: 4px solid #4CAF50;">
      {user_input}
    </blockquote>
    <hr>
    <p style="color: #666; font-size: 12px;">This email was sent automatically by the Quorum student assistant bot.</p>
  </body>
</html>
        """
        
        # Attach both plain text and HTML versions
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        message.attach(part1)
        message.attach(part2)
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Upgrade connection to secure
            server.login(sender_email, sender_password)
            server.send_message(message)
        
        logger.info(f"[SUCCESS] Email sent to {teacher_email} from {user_name}")
        return {
            "success": True,
            "action": "email_sent",
            "message": f"Email sent to the teacher with the question from the user."
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to send email: {str(e)}")
        return {
            "success": False,
            "action": "email_failed",
            "message": f"Failed to send email.",
            "from": sender_email,
            "to": teacher_email,
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
    
    logger.info(f"Request from {user_name}: {user_input}")
    
    with model_lock:
        global pending_intent

        action_taken = False
        action_result = None

        confirm_yes = {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "confirm"}
        confirm_no = {"no", "n", "nope", "cancel", "stop"}
        normalized_input = user_input.strip().lower() if user_input else ""

        # If we had a pending intent from the previous message, act only after explicit confirmation
        if pending_intent["intent"]:
            if normalized_input in confirm_yes:
                logger.info(f"TAKING ACTION AFTER CONFIRMATION: {pending_intent['pretty']}")
                confirmed_intent_pretty = pending_intent["pretty"]
                if pending_intent["intent"] == "send_email":
                    action_result = send_email_to_teacher(pending_intent["text"], pending_intent["user_name"])
                    action_taken = True
                elif pending_intent["intent"] == "make_group":
                    action_result = create_group_formation_request(pending_intent["text"], pending_intent["user_name"])
                    action_taken = True
                elif pending_intent["intent"] == "get_information":
                    action_result = retrieve_information_request(pending_intent["text"], pending_intent["user_name"])
                    action_taken = True
                # Clear pending intent after handling
                pending_intent = {"intent": None, "pretty": None, "text": None, "user_name": None}
            elif normalized_input in confirm_no:
                logger.info("Confirmation denied. Clearing pending intent.")
                pending_intent = {"intent": None, "pretty": None, "text": None, "user_name": None}

        # Detect user intent in the current message to set up confirmation for the next turn
        detected_intent = get_intent(user_input.lower())
        intent_val = False
        detected_intent_pretty = None
        
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

        # Store pending intent for next user confirmation, including the original text
        if intent_val:
            pending_intent = {
                "intent": detected_intent,
                "pretty": detected_intent_pretty,
                "text": user_input,
                "user_name": user_name,
            }
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Quorum, a helpful bot assistant for students and teachers. "
                    "Your main goal is to perform tasks for students based on their requests. You must ask confirmation before performing a task when an intent is detected. "
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

        if memory_context and not action_taken:
            developer_content.append("\nRelevant conversation memory:")
            developer_content.append(memory_context)

        if intent_val:
            developer_content.append(
                f"\nDETECTED INTENT: {detected_intent_pretty}\n"
                "When an intent is detected, do not answer the content of the user's message. "
                "Ask only for a confirmation with a concise yes/no question about performing that intent. "
                "Do not provide any other reply until the user answers yes or no."
            )

        if action_taken and action_result is not None:
            developer_content.append(
                f"\nAction taken: {action_result['action']}\n"
                f"Result message: {action_result['message']}\n"
                f"Success: {action_result['success']}\n"
                f"From: {action_result.get('from', 'N/A')}\n"
                f"To: {action_result.get('to', 'N/A')}\n"
            )
            messages.append({
            "role": "developer",
            "content": "\n".join(developer_content)
            })
        else:
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
    if intent_val:
        return f"✅ You want to {detected_intent_pretty}. {response_text}"
    else:
        return response_text


if __name__ == '__main__':
    # Disable reloader to avoid subprocess buffering issues with logging
    app.run(debug=True, use_reloader=False)