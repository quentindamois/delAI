Install the library from requirements.txt in a venv if possible
to do this run:
 - python -m venv venv
 - pip install -r requirements.txt

Add a discord.env in *students-assistant* with the following field:
```
ENABLE_DISCORD_BOT=1
ENABLE_TEAMS_BOT=0

DISCORD_BOT_TOKEN=<your discord bot token>
LLM_ENDPOINT=http://127.0.0.1:5000/ask
LLM_ENDPOINT=http://127.0.0.1:5000/summary
```
Add the models to the models/ folder in flask_chatbot_student_assistant/

Add a credential to the google service account with the google sheet api activated in the folder *flask_chatbot_student_assistant*

In two terminals run :
- start-discord-bot.ps1 
- start-flask.ps1


