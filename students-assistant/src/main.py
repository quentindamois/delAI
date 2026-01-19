import asyncio
import os

import discord
import httpx
from microsoft_teams.api import MessageActivity, TypingActivityInput
from microsoft_teams.apps import ActivityContext, App
from microsoft_teams.devtools import DevToolsPlugin


LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://flask_app_llm:5000/ask")

app = App(plugins=[DevToolsPlugin()])

discord_intents = discord.Intents.default()
discord_intents.message_content = True
discord_client = discord.Client(intents=discord_intents)


async def fetch_llm_response(message_text: str, user_name: str) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            LLM_ENDPOINT,
            data={"user_input": message_text, "user_name": user_name},
        )
        response.raise_for_status()
        return response.text


@app.on_message
async def handle_message(ctx: ActivityContext[MessageActivity]):
    await ctx.reply(TypingActivityInput())

    try:
        response_text = await fetch_llm_response(
            ctx.activity.text, ctx.activity.from_.name
        )
    except httpx.HTTPError as exc:
        await ctx.send(
            f"I cannot connect to the flask app. ({exc.__class__.__name__}: {exc})"
        )
        return

    await ctx.send(response_text)


@discord_client.event
async def on_ready():
    print(f"Discord bot connected as {discord_client.user}")


@discord_client.event
async def on_message(message: discord.Message):
    if message.author == discord_client.user or message.author.bot:
        return

    async with message.channel.typing():
        try:
            response_text = await fetch_llm_response(
                message.content, message.author.display_name
            )
        except httpx.HTTPError as exc:
            await message.channel.send(
                f"I cannot connect to the flask app. ({exc.__class__.__name__}: {exc})"
            )
            return

    await message.channel.send(response_text)


async def start_teams_bot() -> None:
    await app.start()


async def start_discord_bot(token: str) -> None:
    await discord_client.start(token)


async def run_bots(run_teams: bool, run_discord: bool, discord_token: str | None):
    tasks = []

    if run_teams:
        tasks.append(asyncio.create_task(start_teams_bot()))

    if run_discord:
        if not discord_token:
            print(
                "ENABLE_DISCORD_BOT is set but DISCORD_BOT_TOKEN is missing; skipping Discord bot."
            )
        else:
            tasks.append(asyncio.create_task(start_discord_bot(discord_token)))

    if not tasks:
        raise RuntimeError("Nothing to run. Enable at least one bot.")

    await asyncio.gather(*tasks)


def main():
    run_teams = os.getenv("ENABLE_TEAMS_BOT", "1").lower() not in {"0", "false", "no"}
    run_discord = os.getenv("ENABLE_DISCORD_BOT", "0").lower() in {"1", "true", "yes"}
    discord_token = os.getenv("DISCORD_BOT_TOKEN")

    asyncio.run(run_bots(run_teams, run_discord, discord_token))


if __name__ == "__main__":
    main()
