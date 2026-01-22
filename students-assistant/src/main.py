import asyncio
import os
import sys
import logging

import discord
import httpx
from microsoft_teams.api import MessageActivity, TypingActivityInput
from microsoft_teams.apps import ActivityContext, App
from microsoft_teams.devtools import DevToolsPlugin

from conversation_logger import (
    init_csv,
    log_user_message,
    log_bot_message,
    get_last_interactions
)
from user_context import (
    init_user_file,
    get_user_info,
    format_user_context,
    update_user_info_from_message
)
from rag_utils import search as search_rag, format_search_results, is_rag_available

# Configure logging to output immediately
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

from long_term_memory import retrieve_relevant_memories, format_long_term_memory


# main.py - modifier ces lignes
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:5000/ask")
logger.info(f"Using LLM ENDPOINT: {LLM_ENDPOINT}")
SUMMARIZATION_ENDPOINT = os.getenv("SUMMARIZATION_ENDPOINT", "http://127.0.0.1:5000/summarize")

MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "400"))
ENABLE_RAG = os.getenv("ENABLE_RAG", "1").lower() in {"1", "true", "yes"}

app = App(plugins=[DevToolsPlugin()])

discord_intents = discord.Intents.default()
discord_intents.message_content = True
discord_client = discord.Client(intents=discord_intents)


async def summarize_memory_context(context: str, user_name: str) -> str:
    """
    Use the generative model to summarize memory context if it exceeds MAX_CONTEXT_LENGTH.
    Returns the original context if it's short enough, or a summary otherwise.
    """
    if len(context) <= MAX_CONTEXT_LENGTH:
        return context
    
    logger.info(
        "Context too long (%d chars), generating summary for user %s",
        len(context),
        user_name
    )
    
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                SUMMARIZATION_ENDPOINT,
                data={
                    "context": context,
                    "user_name": user_name,
                },
            )
            response.raise_for_status()
            summary = response.text
            logger.info("Context summarized: %d chars -> %d chars", len(context), len(summary))
            return summary
    except httpx.HTTPError as exc:
        logger.warning(
            "Failed to summarize context (%s), using original",
            exc.__class__.__name__
        )
        return context


async def fetch_llm_response(message_text: str, user_name: str) -> str:
    short_term_interactions = get_last_interactions(user_name)
    short_term_context = format_short_term_memory(short_term_interactions)

    long_term_memories = retrieve_relevant_memories(user_name, message_text)
    long_term_context = format_long_term_memory(long_term_memories)

    memory_context = f"{short_term_context}{long_term_context}"

    user_context = format_user_context(get_user_info(user_name))

    # Add RAG results if available
    rag_context = ""
    if ENABLE_RAG and is_rag_available():
        try:
            rag_results = search_rag(message_text, top_k=1)
            if rag_results:
                rag_context = format_search_results(rag_results)
                logger.info(f"RAG found {len(rag_results)} relevant documents for user {user_name}")
        except Exception as e:
            logger.warning(f"Error retrieving RAG results: {e}")

    # Combine all contexts
    combined_context = f"{memory_context}{rag_context}"

    # Summarize memory context if it's too long
    combined_context = await summarize_memory_context(combined_context, user_name)

    print("Memory context sent to LLM:", combined_context)
    
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            LLM_ENDPOINT,
            data={
                "user_input": message_text,
                "user_name": user_name,
                "memory_context": combined_context,
                "user_context": user_context,
            },
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
    logger.info(f"Discord bot connected as {discord_client.user}")


@discord_client.event
async def on_message(message: discord.Message):
    if message.author == discord_client.user or message.author.bot:
        return

    # Use Discord ID as unique identifier, display_name for readability
    user_id = str(message.author.id)
    display_name = message.author.display_name

    # Try to extract and store user info from the message
    update_user_info_from_message(
        user_id,
        display_name,
        message.content
    )

    logger.info(f"Message from {display_name}: {message.content}")

    # Log message utilisateur
    log_user_message(
        user_id,
        message.content
    )

    async with message.channel.typing():
        try:
            response_text = await fetch_llm_response(
                message.content, user_id
            )
        except httpx.HTTPError as exc:
            await message.channel.send(
                f"I cannot connect to the flask app. ({exc.__class__.__name__}: {exc})"
            )
            return

    # Log réponse du bot
    log_bot_message(
        user_id,
        response_text
    )

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

def format_interaction_blocks(interactions: list[tuple[str, str]]) -> str:
    blocks = []
    for user_msg, bot_msg in interactions:
        blocks.append(
            f"User message: {user_msg}\n"
            f"Agent response: {bot_msg}"
        )
    return "\n\n".join(blocks)

def format_short_term_memory(interactions: list[tuple[str, str]]) -> str:
    if not interactions:
        return ""

    content = format_interaction_blocks(interactions)

    return (
        "Here are the most recent interactions you had with the user.\n"
        f"{content}\n\n"
    )

def format_long_term_memory(memories: list[dict]) -> str:
    if not memories:
        return ""

    interactions = [
        (mem["user_message"], mem["agent_response"])
        for mem in memories
    ]

    content = format_interaction_blocks(interactions)

    return (
        "Here are past interactions with the user that may be relevant.\n"
        f"{content}\n\n"
    )

def main():
    init_csv()
    init_user_file()

    run_teams = os.getenv("ENABLE_TEAMS_BOT", "1").lower() in {"1", "true", "yes"}
    run_discord = os.getenv("ENABLE_DISCORD_BOT", "1").lower() in {"1", "true", "yes"}
    discord_token = os.getenv("DISCORD_BOT_TOKEN")

    asyncio.run(run_bots(run_teams, run_discord, discord_token))


if __name__ == "__main__":
    main()