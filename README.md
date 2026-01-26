Install the library from requirements.txt in a virtual environment called venv
to do this run:
 - python -m venv venv
 - pip install -r requirements.txt

To not have any issue when installing everything, you must have python 3.12.

Add a discord.env in *students-assistant* with the following field:
```
ENABLE_DISCORD_BOT=1
ENABLE_TEAMS_BOT=0

DISCORD_BOT_TOKEN=<your discord bot token>
LLM_ENDPOINT=http://127.0.0.1:5000/ask
SUMMARIZATION_ENDPOINT=http://127.0.0.1:5000/summary
```
Inside the folder *flask_chatbot_student_assistant* create a dot .env and put the following content inside
```
PATH_CREDENTIAL= <path to the credential for the google service account>
SMTP_SERVER=<addresse of the smtp server>
SMTP_PORT=<port of the smtp server>


MAIL_FROM=<email address that will be used by the chat bot>
MAIL_PASSWORD=<password of the account of the email address> 
MAIL_TO=<email address that will be used by the chat bot>

IMAP_USER=<email address that will be used by the chat bot>
IMAP_PASSWORD=<password of the account of the email address> 
DISCORD_WEBHOOK_URL=<discord webhook url>
```
Add the models to the models/ folder in flask_chatbot_student_assistant/
They must be:
 - llama-3.2-1b-instruct-q8_0.gguf
 - nomic-embed-text-v1.5.Q4_K_M.gguf

You can download them from hugging face at following address:

https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF

https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF


Add a credential to the google service account with the Google Sheets api activated in the folder *flask_chatbot_student_assistant*

In two terminals run :
- start-discord-bot.ps1 
- start-flask.ps1

In order to run them you must have a vitual environment called venv in python 3.12 and run the script at the root of the project.

When lauching the script for the flask app, you may have an os error.
If this is the case re run the lauching script with foollwing command.
 - cd ../
 - start-flask.ps1
