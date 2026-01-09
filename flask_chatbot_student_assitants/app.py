from flask import Flask, request
from llama_cpp import Llama



llm = Llama.from_pretrained(
	repo_id="hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF",
	filename="llama-3.2-1b-instruct-q8_0.gguf",
)


intent_classifier = Llama.from_pretrained(
	repo_id="mradermacher/Qwen-3B-Intent-Microplan-v2-i1-GGUF",
	filename="Qwen-3B-Intent-Microplan-v2.i1-Q6_K.gguf",
)


import sqlite3
con = sqlite3.connect("tutorial.db")

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello world"

@app.route("/ask", methods=['POST'])
def answer_ask():
    user_input = request.form.get('user_input')
    user_name = request.form.get('user_name')
    intent = intent_classifier.create_chat_completion(
        messages = [
            {"role": "system", "content": "Give the intent of this message between the following : 'send email' for sending an email to a teacher in order to ask a question, 'ask opinion' send a form to students to evaluate and ask their opinions on one ore more courses, 'ask to form group' ask the student to form groups for a project or a presentation, 'none' if it doesn't correspond to the other."},
            {
                "role": "user",
                "content": user_input
            }
        ]
    )
    answer = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": f"You are an assistant who help student and teacher interacting togethers. You only speak english. The intent of the user is {intent["choices"][0]["message"]["content"]}. The student can ask you to send email to the teacher to ask question, you can send form to students to ask their opinions on one or more course, the teachers can ask you to send a form to students to form group for project or presentation."},
            {
                "role": "user",
                "content": user_input
            }
        ]
    )
    return f"The intent is : {intent["choices"][0]["message"]["content"]}\nMy answer is : {answer["choices"][0]["message"]["content"]}"


if __name__ == '__main__':
    app.run(debug=True)