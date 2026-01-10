import asyncio
import re
#from llama_cpp import Llama
import requests
from microsoft_teams.api import MessageActivity, TypingActivityInput
from microsoft_teams.apps import ActivityContext, App
from microsoft_teams.devtools import DevToolsPlugin



"""
llm = Llama.from_pretrained(
	repo_id="hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF",
	filename="llama-3.2-1b-instruct-q8_0.gguf",
)


intent_classifier = Llama.from_pretrained(
	repo_id="mradermacher/Qwen-3B-Intent-Microplan-v2-i1-GGUF",
	filename="Qwen-3B-Intent-Microplan-v2.i1-Q6_K.gguf",
)
"""


app = App(plugins=[DevToolsPlugin()])


"""Handle greeting messages."""
"""
@app.on_message_pattern(re.compile(r"hello|hi|greetings"))
async def handle_greeting(ctx: ActivityContext[MessageActivity]) -> None:
    
    await ctx.send("Hello! How can I assist you today?")
"""
"""Handle message activities using the new generated handler system."""

@app.on_message
async def handle_message(ctx: ActivityContext[MessageActivity]):
   
    await ctx.reply(TypingActivityInput())
    #await ctx.send(f"I cannot connect to the flask app.")
    responce = requests.post("http://flask_app_llm:5000/ask", data=dict(user_input=ctx.activity.text, user_name=ctx.activity.from_.name), timeout=120)
    if responce.status_code == 200:
        await ctx.send(responce.text)
    else:
        await ctx.send(f"I cannot connect to the flask app.\n{responce.status_code}")
"""
async def handle_message(ctx: ActivityContext[MessageActivity]):
   
    await ctx.reply(TypingActivityInput())
    intent = intent_classifier.create_chat_completion(
        messages = [
            {"role": "system", "content": "Give the intent of this message between the following : 'send email' for sending an email to a teacher in order to ask a question, 'ask opinion' send a form to students to evaluate and ask their opinions on one ore more courses, 'ask to form group' ask the student to form groups for a project or a presentation, 'none' if it doesn't correspond to the other."},
            {
                "role": "user",
                "content": ctx.activity.text
            }
        ]
    )
    answer = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": f"You are an assistant who help student and teacher interacting togethers. You only speak english. The intent of the user is {intent["choices"][0]["message"]["content"]}. The student can ask you to send email to the teacher to ask question, you can send form to students to ask their opinions on one or more course, the teachers can ask you to send a form to students to form group for project or presentation."},
            {
                "role": "user",
                "content": ctx.activity.text
            }
        ]
    )
    await ctx.send(f"The intent is : {intent["choices"][0]["message"]["content"]}\nMy answer is : {answer["choices"][0]["message"]["content"]}")
"""

def main():
    asyncio.run(app.start())


if __name__ == "__main__":
    main()
