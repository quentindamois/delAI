from flask import Flask, request
from llama_cpp import Llama
import re
import threading

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


intent_classifier = Llama(model_path="./models/Qwen-3B-Intent-Microplan-v2.i1-Q6_K.gguf", verbose=False)

# Lock pour empêcher les accès concurrents aux modèles
model_lock = threading.Lock()


# Action handlers
def send_email_to_teacher(user_input: str, user_name: str) -> dict:
    """Send an email to a teacher with the student's question."""
    # TODO: Implement actual email sending logic (SMTP, API, etc.)
    print(f"[ACTION] Sending email to teacher from {user_name}: {user_input}")
    return {
        "success": True,
        "action": "email_sent",
        "message": f"Email sent to the teacher with your question."
    }


def create_opinion_form(user_input: str, user_name: str) -> dict:
    """Create a form to gather student opinions about courses."""
    # TODO: Implement form creation (Google Forms API, Microsoft Forms, etc.)
    print(f"[ACTION] Creating opinion form requested by {user_name}: {user_input}")
    return {
        "success": True,
        "action": "form_created",
        "message": "Opinion form created and sent to students."
    }


def create_group_formation_request(user_input: str, user_name: str) -> dict:
    """Create a request for students to form groups."""
    # TODO: Implement group formation logic (database, notifications, etc.)
    print(f"[ACTION] Creating group formation request by {user_name}: {user_input}")
    return {
        "success": True,
        "action": "groups_requested",
        "message": "Group formation request sent to students."
    }


keyword_dictionnary = {
    "send_email":{"verb":["write", "email", "ask about"], "noun":["teacher", "email"]},
    "evaluation_form":{"verb":["evaluate", "give"], "adj":["end of semester", "end of course", "mid semester", "end of year"], "noun":["returns", "feedback", "opinions", "advice", "class"]},
    "make_group":{"verb":["make", "form", "assemble"], "adj":["final project", "project", "presentation"], "noun":["group", "group", "team",]}
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
    print(list_detected_intent)
    return "none" if len(list_detected_intent) == 0 else list_detected_intent[0]


last_message = ""

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello world"

@app.route("/back", methods=["GET"])
def get_last_message():
    return last_message

@app.route("/ask", methods=['POST'])
def answer_ask():
    global last_message  # Fix: need to declare global to update the variable
    user_input = request.form.get('user_input')
    user_name = request.form.get('user_name', 'User')
    
    with model_lock:
        # Detect user intent
        intent = intent_classifier.create_chat_completion(
            messages = [
                {"role": "system", "content": """Analyze the user's message and classify their intent into ONE of the following categories:

- 'send email': The user (student) wants to send an email to a teacher to ask a question or request information
- 'ask to form group': The user (teacher) wants students to organize themselves into groups for a project or presentation
- 'get information': The user (student or teacher) is seeking general information or help
- 'none': The message is a general question, greeting, or doesn't match any of the above categories

Respond with ONLY the intent category name."""},
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        )
        
        detected_intent = intent["choices"][0]["message"]["content"].strip().lower()
        action_result = None
        action_taken = False
        
        # Execute actions based on detected intent
        if "send email" in detected_intent or "email" in detected_intent:
            action_result = send_email_to_teacher(user_input, user_name)
            action_taken = True
        elif "ask opinion" in detected_intent or "opinion" in detected_intent:
            action_result = create_opinion_form(user_input, user_name)
            action_taken = True
        elif "form group" in detected_intent or "group" in detected_intent:
            action_result = create_group_formation_request(user_input, user_name)
            action_taken = True
        
        # Generate conversational response
        if action_taken and action_result:
            system_prompt = f"""You are a helpful assistant for students and teachers. You only speak English.

You have just performed this action: {action_result['message']}

Confirm this action to the user in a friendly and professional way. Be brief but reassuring."""
        else:
            system_prompt = f"""You are a helpful assistant for students and teachers. You only speak English. 

The user's intent is: {detected_intent}

You can help with:
- Students: Send emails to teachers with questions
- Teachers: Help students form groups for projects or presentations
- Provide general information and assistance

If the user speaks in another language, politely ask them to communicate in English. Provide helpful and friendly responses."""
        
        answer = llm.create_chat_completion(
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        )
    
    response_text = answer["choices"][0]["message"]["content"]
    
    # Format response
    if action_taken:
        last_message = f"[Intent: {detected_intent}] [Action: {action_result['action']}]\n{response_text}"
        return f"✅ {response_text}"
    else:
        last_message = f"[Intent: {detected_intent}]\n{response_text}"
        return response_text


if __name__ == '__main__':
    app.run(debug=True)