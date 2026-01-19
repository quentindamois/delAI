

# Python Teams Bot

This chat bot based on the minimal Microsoft Teams echo bot template using [microsoft-teams](https://github.com/microsoft/teams.py).

## Structure

- `src/main.py`: Main application code for the Teams bot.
- `pyproject.toml`: Project dependencies and metadata (use [uv](https://github.com/astral-sh/uv) for dependency management).
- `apppackage/`: Teams app manifest and related files.

## Getting Started

1. Install [uv](https://github.com/astral-sh/uv).
2. Run `uv run start`

## Discord Bot (optional)

1. Create a Discord application and bot, then copy the bot token.
2. Set environment variables:
	- `DISCORD_BOT_TOKEN`: the token from the Discord bot portal.
	- `ENABLE_DISCORD_BOT=1` to start the Discord bot alongside the Teams bot.
	- Optionally set `ENABLE_TEAMS_BOT=0` if you want to run only Discord.
	- `LLM_ENDPOINT` overrides the default `http://flask_app_llm:5000/ask` endpoint if needed.
3. Start the service: `ENABLE_DISCORD_BOT=1 DISCORD_BOT_TOKEN=your_token uv run start`.
4. Invite the bot to your server and send a message; it will relay your text to the LLM backend just like the Teams bot.
