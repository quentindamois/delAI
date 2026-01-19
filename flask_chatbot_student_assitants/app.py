from flask import Flask, request
from llama_cpp import Llama
import re

"""
llm = Llama.from_pretrained(
	repo_id="hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF",
	filename="llama-3.2-1b-instruct-q8_0.gguf",
)
"""

llm = Llama(model_path="./models/llama-3.2-1b-instruct-q8_0.gguf")





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
    user_input = request.form.get('user_input')
    user_name = request.form.get('user_name')
    intent = get_intent(text=user_input)
    answer = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": f"You are an assistant who help student and teacher interacting togethers. You only speak english. The intent of the user is {intent["choices"][0]["message"]["content"]}. The student can ask you to send email to the teacher to ask question, you can send form to students to ask their opinions on one or more course, the teachers can ask you to send a form to students to form group for project or presentation."},
            {
                "role": "user",
                "content": user_input
            }
        ]
    )
    last_message = f"The intent is : {intent["choices"][0]["message"]["content"]}\nMy answer is : {answer["choices"][0]["message"]["content"]}"
    return f"The intent is : {intent["choices"][0]["message"]["content"]}\nMy answer is : {answer["choices"][0]["message"]["content"]}"


if __name__ == '__main__':
    app.run(debug=True)