import json
import logging
import os
import re
import smtplib
import time
from difflib import SequenceMatcher
from email import policy
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def normalize_teacher_key(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_teachers(path: str, logger: logging.Logger | None = None):
    logger = logger or logging.getLogger(__name__)
    if not os.path.exists(path):
        logger.warning(f"Teacher file not found at {path}; email send will require manual address")
        return [], {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error("Teacher file must be a list of objects")
            return [], {}
        teacher_lookup: dict[str, dict] = {}
        for entry in data:
            names = [entry.get("name", "")]
            aliases = entry.get("aliases", []) or []
            for key in names + aliases:
                norm = normalize_teacher_key(key)
                if norm:
                    teacher_lookup[norm] = entry
        logger.info(f"Loaded {len(data)} teachers from {path}")
        return data, teacher_lookup
    except Exception as e:
        logger.error(f"Failed to load teachers: {e}")
        return [], {}


def _fuzzy_match_in_text(norm_text: str, norm_name: str, threshold: float = 0.8) -> bool:
    if not norm_text or not norm_name:
        return False
    text_tokens = norm_text.split()
    name_tokens = norm_name.split()
    if not text_tokens or not name_tokens:
        return False
    win = len(name_tokens)
    for i in range(len(text_tokens) - win + 1):
        candidate = " ".join(text_tokens[i : i + win])
        if norm_name in candidate:
            return True
        if SequenceMatcher(None, norm_name, candidate).ratio() >= threshold:
            return True
    return False


def resolve_teacher_from_text(text: str, teacher_lookup: dict, allow_fuzzy: bool = True, threshold: float = 0.8):
    """Return (match, matches_list). match is None if none or ambiguous."""
    if not teacher_lookup:
        return None, []
    norm_text = normalize_teacher_key(text)
    found = {}
    # exact/substring match first
    for key, entry in teacher_lookup.items():
        if key and key in norm_text:
            found[id(entry)] = entry
    # fuzzy match if nothing found
    if not found and allow_fuzzy:
        for key, entry in teacher_lookup.items():
            if _fuzzy_match_in_text(norm_text, key, threshold=threshold):
                found[id(entry)] = entry
    matches = list(found.values())
    if len(matches) == 1:
        return matches[0], matches
    return None, matches


def generate_email_draft(llm, model_lock, user_input: str, user_name: str, teacher: dict | None = None, logger: logging.Logger | None = None, draft_lock_timeout: int = 20) -> str:
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Generating email draft for {user_name}: {user_input[:100]}")
    teacher_name = (teacher or {}).get("name", "your teacher")
    teacher_course = (teacher or {}).get("course")
    prompt = (
        "Write a short, respectful email from the student to their teacher.\n"
        "- The teacher is: " + teacher_name + ".\n"
        + (f"- The course is: {teacher_course}.\n" if teacher_course else "")
        + "- Use the student's display name: " + user_name + " in the sign-off.\n"
        "- Keep it clear and factual.\n"
        "- Base it strictly on this request: " + user_input + "\n"
        "- Include a greeting and a courteous closing.\n"
        "- Do NOT add extra details or any preamble like 'Here is the email'.\n"
        "Return only the email body."
    )
    if not model_lock.acquire(timeout=draft_lock_timeout):
        logger.error("Model lock busy for more than %ss while drafting email", draft_lock_timeout)
        raise TimeoutError("Model busy while drafting email")

    try:
        start_time = time.time()
        messages = [
            {"role": "system", "content": "You write short, polite emails for students. Keep it factual and concise."},
            {"role": "user", "content": prompt},
        ]
        answer = llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.3,
        )
        draft = answer["choices"][0]["message"]["content"].strip()
        elapsed = time.time() - start_time
        logger.info(f"Email draft generated in {elapsed:.2f}s: {draft[:120]}")
        return draft
    finally:
        model_lock.release()


def send_email_to_teacher(user_input: str, user_name: str, teacher: dict, logger: logging.Logger | None = None) -> dict:
    """Send an email to a teacher with the student's question."""
    logger = logger or logging.getLogger(__name__)
    teacher_name = (teacher or {}).get("name", "the teacher")
    teacher_course = (teacher or {}).get("course")
    teacher_email = (teacher or {}).get("email")
    logger.info(f"[ACTION] Sending email to teacher {teacher_name} from {user_name}: {user_input[:100]}")
    
    if not teacher_email:
        logger.error("No teacher email provided; aborting send")
        return {
            "success": False,
            "action": "email_failed",
            "message": "Teacher email not available; update teachers.json",
            "from": None,
            "to": None,
        }

    # Email configuration (use environment variables in production)
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')  # e.g., smtp.gmail.com, smtp.office365.com
    smtp_port = int(os.getenv('SMTP_PORT', '587'))  # 587 for TLS, 465 for SSL
    sender_email = os.getenv('MAIL_FROM', 'your-email@example.com')
    sender_password = os.getenv('MAIL_PASSWORD', 'your-password')

    # Strip potential BOM or hidden characters from env inputs
    sender_email = sender_email.encode('utf-8').decode('utf-8-sig')
    sender_password = sender_password.encode('utf-8').decode('utf-8-sig')
    teacher_email = teacher_email.encode('utf-8').decode('utf-8-sig')
    
    try:
        # Create message with SMTP policy; subject encoded as RFC2047 for safety
        message = MIMEMultipart('alternative', policy=policy.SMTP)
        subject = f"Message from {user_name}"
        if teacher_course:
            subject = f"{subject} about {teacher_course}"
        message['Subject'] = str(Header(subject, 'utf-8'))
        message['From'] = sender_email
        message['To'] = teacher_email
        
        # Email body
        text_content = f"""
Hello {teacher_name},

You have received a message from student {user_name}.
{f'Course: {teacher_course}\n' if teacher_course else ''}

{user_input}

---
This email was sent automatically by the Quorum student assistant bot.
        """
        
        html_content = f"""
<html>
  <body>
    <p>Hello {teacher_name},</p>
    <p>You have received a message from student <strong>{user_name}</strong>.</p>
    {f'<p><em>Course: {teacher_course}</em></p>' if teacher_course else ''}
    <blockquote style="margin: 20px; padding: 10px; background-color: #f5f5f5; border-left: 4px solid #4CAF50;">
      {user_input}
    </blockquote>
    <hr>
    <p style="color: #666; font-size: 12px;">This email was sent automatically by the Quorum student assistant bot.</p>
  </body>
</html>
        """
        
        # Attach both plain text and HTML versions with UTF-8 encoding
        part1 = MIMEText(text_content, 'plain', _charset='utf-8')
        part2 = MIMEText(html_content, 'html', _charset='utf-8')
        message.attach(part1)
        message.attach(part2)        
        # Send email
        # Force ASCII-safe local hostname to avoid non-ASCII encoding in EHLO
        with smtplib.SMTP(smtp_server, smtp_port, local_hostname="localhost") as server:
            logger.info("SMTP connecting: starttls -> login -> sendmail")
            server.starttls()  # Upgrade connection to secure

            # Additional diagnostics for auth encoding
            try:
                sender_email.encode('ascii')
                sender_password.encode('ascii')
            except Exception as enc_err:
                logger.error("Auth contains non-ascii chars: %s", enc_err)

            logger.info("SMTP login...")
            server.login(sender_email, sender_password)
            logger.info("SMTP login OK, sending mail...")
            # send as bytes; headers are encoded-word, body utf-8 encoded
            server.sendmail(sender_email, [teacher_email], message.as_bytes())
            logger.info("SMTP sendmail done")
        
        logger.info(f"[SUCCESS] Email sent to {teacher_email} from {user_name}")
        return {
            "success": True,
            "action": "email_sent",
            "message": f"[SUCCESS] Email sent to {teacher_email} from {user_name}",
            "from": sender_email,
            "to": teacher_email,
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to send email: {str(e)}")
        return {
            "success": False,
            "action": "email_failed",
            "message": f"[ERROR] Failed to send email: {str(e)}",
            "from": sender_email,
            "to": teacher_email,
        }
