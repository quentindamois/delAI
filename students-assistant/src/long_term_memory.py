import csv
import os
from collections import deque
from typing import List, Dict
import numpy as np

from pathlib import Path
from llama_cpp import Llama

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "nomic-embed-text-v1.5.Q4_K_M.gguf"
CSV_FILE = "conversation_log.csv"

MAX_HISTORY = int(os.getenv("LONG_TERM_HISTORY", 50))
TOP_K = int(os.getenv("LONG_TERM_TOP_K", 2))


# Load embedding model once
embedding_model = Llama(
    model_path=str(MODEL_PATH),
    embedding=True,
    verbose=False,
)

def get_last_interaction_pairs(username: str, limit: int = MAX_HISTORY) -> List[Dict]:
    """
    Returns the last `limit` interaction pairs for a user:
    [{ "user_message": ..., "agent_response": ... }]
    """
    interactions = deque(maxlen=limit)
    current_user_message = None

    if not os.path.exists(CSV_FILE):
        return []

    with open(CSV_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"] != username:
                continue

            if row["role"] == "user":
                current_user_message = row["message"]

            elif row["role"] == "bot" and current_user_message:
                interactions.append({
                    "user_message": current_user_message,
                    "agent_response": row["message"],
                })
                current_user_message = None

    return list(interactions)

def embed_text(text: str) -> np.ndarray:
    embedding = embedding_model.embed(text)
    return np.array(embedding)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_relevant_memories(
    username: str,
    current_user_message: str,
) -> List[Dict]:
    interactions = get_last_interaction_pairs(username)

    if not interactions:
        return []

    query_embedding = embed_text(current_user_message)

    scored_memories = []

    for interaction in interactions:
        past_embedding = embed_text(interaction["user_message"])
        score = cosine_similarity(query_embedding, past_embedding)

        scored_memories.append((score, interaction))

    scored_memories.sort(key=lambda x: x[0], reverse=True)

    return [interaction for _, interaction in scored_memories[:TOP_K]]

def format_long_term_memory(memories: List[Dict]) -> str:
    if not memories:
        return ""

    blocks = []
    for mem in memories:
        blocks.append(
            f"User message: {mem['user_message']}\n"
            f"Agent response: {mem['agent_response']}"
        )

    joined = "\n\n".join(blocks)

    return (
        "Here are past interactions with the user that may be relevant.\n"
        "Use them only if they help you answer the current message.\n\n"
        f"{joined}\n\n"
    )
