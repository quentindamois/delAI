import csv
import os
from datetime import datetime
from threading import Lock
from collections import deque

CSV_FILE = "conversation_log.csv"
_lock = Lock()

# Hyperparameter: number of past interactions
MEMORY_SIZE = int(os.getenv("MEMORY_SIZE", 3))


def _get_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["role", "username", "message", "timestamp"]
            )


def log_user_message(username: str, message: str):
    with _lock:
        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["user", username, message, _get_timestamp()]
            )


def log_bot_message(username: str, message: str):
    with _lock:
        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["bot", username, message, _get_timestamp()]
            )


def get_last_interactions(username: str, n: int | None = None) -> list[tuple[str, str]]:
    """
    Returns the last n (user_message, bot_message) pairs for a given user.
    """
    if n is None:
        n = MEMORY_SIZE

    interactions = deque(maxlen=n)
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

            elif row["role"] == "bot" and current_user_message is not None:
                interactions.append(
                    (current_user_message, row["message"])
                )
                current_user_message = None

    return list(interactions)
