import imaplib
import os
import time
import logging
import email
from email.parser import BytesParser
from email.policy import default as default_policy
from email.utils import parsedate_to_datetime, parseaddr
import requests
from dotenv import load_dotenv
from email_sender import load_teachers

# Load environment variables from .env
load_dotenv()

# Basic logger setup
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Config
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_USER = os.getenv("IMAP_USER", "")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD", "")
POLL_INTERVAL = int(os.getenv("IMAP_POLL_INTERVAL", "30"))
TEACHERS_PATH = os.getenv("TEACHERS_PATH", "./data/teachers.json")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Known senders
teachers, teacher_lookup = load_teachers(TEACHERS_PATH, logger)
KNOWN_SENDER_EMAILS = {t.get("email", "").lower() for t in teachers if t.get("email")}


def connect_imap():
    if not IMAP_HOST or not IMAP_USER or not IMAP_PASSWORD:
        raise RuntimeError("IMAP credentials not configured")
    logger.info("Connecting to IMAP host %s as %s", IMAP_HOST, IMAP_USER)
    mail = imaplib.IMAP4_SSL(IMAP_HOST)
    mail.login(IMAP_USER, IMAP_PASSWORD)
    mail.select("INBOX")
    return mail


def fetch_unseen_uids(mail):
    status, data = mail.search(None, "UNSEEN")
    if status != "OK":
        logger.warning("IMAP search failed: %s", status)
        return []
    uids = data[0].split()
    return uids


def parse_email(msg_bytes):
    parser = BytesParser(policy=default_policy)
    msg = parser.parsebytes(msg_bytes)
    from_name, from_addr = parseaddr(msg.get("From", ""))
    subject = msg.get("Subject", "(no subject)")
    try:
        received_dt = parsedate_to_datetime(msg.get("Date")) if msg.get("Date") else None
    except Exception:
        received_dt = None
    text = None
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    text = part.get_content()
                    break
                except Exception:
                    continue
    else:
        try:
            text = msg.get_content()
        except Exception:
            text = None
    if text:
        text = text.strip()
    return {
        "from_name": from_name or from_addr,
        "from_addr": (from_addr or "").lower(),
        "subject": subject,
        "body": text or "(no body)",
        "received": received_dt,
    }


def send_to_discord(webhook_url: str, content: str):
    if not webhook_url:
        logger.warning("No Discord webhook configured; skipping broadcast")
        return
    logger.info("Sending to Discord webhook: %s...", webhook_url[:50])
    try:
        resp = requests.post(webhook_url, json={"content": content}, timeout=10)
        if resp.status_code >= 300:
            logger.error("Discord webhook failed: %s %s", resp.status_code, resp.text)
        else:
            logger.info("Discord message sent successfully (status %s)", resp.status_code)
    except Exception as e:
        logger.error("Discord webhook error: %s", e)


def format_message(parsed):
    ts = parsed.get("received")
    ts_str = ts.isoformat() if ts else "(unknown time)"
    body = parsed.get("body", "")
    snippet = body[:1000]
    return (
        f"New email from professor {parsed.get('from_name')} ({parsed.get('from_addr')}) at {ts_str}\n"
        f"Subject: {parsed.get('subject')}\n\n"
        f"{snippet}"
        "@everyone"  # Notify everyone in the Discord channel
    )


def process_unseen(mail, seen_uids: set):
    new_uids = fetch_unseen_uids(mail)
    logger.info("Found %d unseen emails", len(new_uids))
    for uid in new_uids:
        if uid in seen_uids:
            logger.debug("UID %s already seen, skipping", uid)
            continue
        status, data = mail.fetch(uid, "(RFC822)")
        if status != "OK" or not data or not data[0]:
            logger.warning("Failed to fetch UID %s", uid)
            continue
        msg_bytes = data[0][1]
        parsed = parse_email(msg_bytes)
        sender = parsed.get("from_addr")
        logger.info("Processing email from %s (known senders: %s)", sender, KNOWN_SENDER_EMAILS)
        if sender not in KNOWN_SENDER_EMAILS:
            logger.info("Skipping email from unknown sender %s", sender)
            seen_uids.add(uid)
            continue
        logger.info("Recognized email from teacher %s, sending to Discord", sender)
        content = format_message(parsed)
        send_to_discord(DISCORD_WEBHOOK_URL, content)
        seen_uids.add(uid)


def run_polling():
    seen_uids = set()
    logger.info("Known teacher emails: %s", KNOWN_SENDER_EMAILS)
    logger.info("Discord webhook URL configured: %s", "Yes" if DISCORD_WEBHOOK_URL else "No")
    while True:
        try:
            logger.debug("Polling IMAP...")
            mail = connect_imap()
            process_unseen(mail, seen_uids)
            mail.logout()
        except Exception as e:
            logger.error("Polling error: %s", e)
        logger.debug("Waiting %d seconds before next poll", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_polling()
