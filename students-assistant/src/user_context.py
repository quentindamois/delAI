import csv
import os
import re
from threading import Lock

USER_FILE = "user_profiles.csv"
_lock = Lock()

# Regex patterns to extract user information from messages
EXTRACTION_PATTERNS = {
    "class": [
        # Match patterns where user is STATING their class (not asking)
        r"(?:i'?m\s+in|i\s+am\s+in|enrolled\s+in)\s+(?:the\s+)?([A-Z]{2,}(?:\s+[A-Z]{2,})?)(?:\s+class)?",
        # Match "my class is DAI", "class: DAI"
        r"(?:my\s+)?class\s*(?:is|:)\s+([A-Z]{2,}(?:\s+[A-Z]{2,})?)\b",
    ],
    "email": [
        # Match email patterns
        r"(?:my\s+)?(?:email|e-mail)\s*(?:is|:)?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b",
    ],
    "student_id": [
        # Match student ID patterns
        r"(?:my\s+)?(?:student\s+)?(?:id|ID|number)\s*(?:is|:)?\s*([0-9]{4,})",
        r"(?:matricule|matricula)\s*(?:is|:)?\s*([0-9]{4,})",
    ],
}

# Common filler words that are NOT class names
IGNORE_WORDS = {"again", "please", "thanks", "thank", "you", "hello", "hi", "hey", "yes", "no", "ok", "okay"}


def init_user_file():
    """Initialize user profiles CSV file if it doesn't exist."""
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["username", "display_name", "class", "student_id", "email"]
            )


def get_user_info(username: str) -> dict:
    """
    Retrieve user information by username.
    Returns a dict with user info or empty dict if not found.
    """
    if not os.path.exists(USER_FILE):
        return {}

    with open(USER_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"].lower() == username.lower():
                return {
                    "username": row["username"],
                    "display_name": row["display_name"],
                    "class": row["class"],
                    "student_id": row["student_id"],
                    "email": row["email"],
                }
    return {}


def set_user_info(username: str, display_name: str, class_name: str = "", student_id: str = "", email: str = ""):
    """
    Add or update user information.
    """
    with _lock:
        # Read existing users
        existing_users = {}
        if os.path.exists(USER_FILE):
            with open(USER_FILE, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_users[row["username"].lower()] = row

        # Update or add user
        existing_users[username.lower()] = {
            "username": username,
            "display_name": display_name,
            "class": class_name,
            "student_id": student_id,
            "email": email,
        }

        # Write back to file
        with open(USER_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["username", "display_name", "class", "student_id", "email"])
            writer.writeheader()
            for user in existing_users.values():
                writer.writerow(user)


def format_user_context(user_info: dict) -> str:
    """Format user information into a context string for the LLM."""
    if not user_info:
        return ""

    lines = ["User Information:"]
    if user_info.get("display_name"):
        lines.append(f"- Name: {user_info['display_name']}")
    if user_info.get("class"):
        lines.append(f"- Class: {user_info['class']}")
    if user_info.get("student_id"):
        lines.append(f"- Student ID: {user_info['student_id']}")
    if user_info.get("email"):
        lines.append(f"- Email: {user_info['email']}")

    return "\n".join(lines) + "\n"


def extract_user_info_from_message(message: str) -> dict:
    """
    Try to extract user information from a message using regex patterns.
    Returns a dict with extracted info (some fields may be empty).
    """
    extracted = {
        "class": "",
        "email": "",
        "student_id": "",
    }

    # Don't extract from questions
    if any(q in message.lower() for q in ["?", "what is", "what's", "which", "tell me"]):
        return extracted

    # Try to extract class (search on original message to preserve case)
    for pattern in EXTRACTION_PATTERNS["class"]:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            class_name = match.group(1).strip().upper()
            # Ignore common filler words
            if class_name.lower() not in IGNORE_WORDS:
                extracted["class"] = class_name
                break

    # Try to extract email (case-sensitive for email part)
    for pattern in EXTRACTION_PATTERNS["email"]:
        match = re.search(pattern, message)
        if match:
            extracted["email"] = match.group(1).strip()
            break

    # Try to extract student ID
    for pattern in EXTRACTION_PATTERNS["student_id"]:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            extracted["student_id"] = match.group(1).strip()
            break

    return extracted


def update_user_info_from_message(username: str, display_name: str, message: str):
    """
    Extract user info from message and update CSV.
    Always creates a user entry if they don't exist (even with blank fields).
    """
    current_info = get_user_info(username)
    
    # If user doesn't exist yet, create them with blank fields
    if not current_info:
        set_user_info(username, display_name, "", "", "")
        print(f"[USER INFO] Created new user profile for {display_name} (ID: {username})")
        current_info = get_user_info(username)
    
    # Try to extract additional info from the message
    extracted = extract_user_info_from_message(message)

    # Update if we found new information
    if extracted["class"] or extracted["email"] or extracted["student_id"]:
        # Merge: keep existing info, add new extracted info
        updated_info = {
            "username": username,
            "display_name": display_name,
            "class": extracted["class"] or current_info.get("class", ""),
            "student_id": extracted["student_id"] or current_info.get("student_id", ""),
            "email": extracted["email"] or current_info.get("email", ""),
        }

        set_user_info(
            username,
            display_name,
            updated_info["class"],
            updated_info["student_id"],
            updated_info["email"],
        )
        print(f"[USER INFO] Updated {display_name}: {extracted}")
