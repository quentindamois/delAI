import csv
import os
from datetime import datetime
from threading import Lock

CSV_FILE = "conversation_log.csv"
_lock = Lock()


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
