import re
import json
import os
import logging
from datetime import datetime
from typing import Tuple

logger = logging.getLogger(__name__)

# Regex patterns for harmful content detection (school-appropriate, English)
HARMFUL_PATTERNS = {
    "bullying": r"\b(stupid|idiot|dumb|loser|sucks|worthless|ugly|fat|lazy)\b",
    "disrespect": r"\b(hate you|shut up|go away|nobody likes|screw you)\b",
    "minor_threats": r"\b(gonna punch|will hit|beat you up|fight you)\b",
    "bathroom_humor": r"\b(poop|fart|butt|pee|gross|ew)\b",
}

def detect_harmful_content(text: str) -> Tuple[bool, str | None]:
    """
    Detect harmful content in text using regex patterns.
    Returns (is_harmful, category_name)
    """
    if not text:
        return False, None
    
    text_lower = text.lower()
    
    for category, pattern in HARMFUL_PATTERNS.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True, category
    
    return False, None


def log_warning(user_id: str, user_name: str, message: str, category: str, logger: logging.Logger | None = None):
    """
    Log a harmful message to console and save to data/warnings/warnings.json
    """
    _logger = logger or logging.getLogger(__name__)
    
    timestamp = datetime.now().isoformat()
    
    # Create huge warning message for console
    warning_message = (
        f"\n{'='*80}\n"
        f"!!! ALERT: HARMFUL/DANGEROUS CONTENT DETECTED !!!\n"
        f"{'='*80}\n"
        f"Timestamp: {timestamp}\n"
        f"User ID: {user_id}\n"
        f"User Name: {user_name}\n"
        f"Category: {category}\n"
        f"Message: {message}\n"
        f"{'='*80}\n"
    )
    
    _logger.warning(warning_message)
    
    # Save to JSON file
    try:
        warnings_dir = "./data/warnings"
        os.makedirs(warnings_dir, exist_ok=True)
        
        filename = os.path.join(warnings_dir, "warnings.json")
        
        # Load existing warnings or create new list
        warnings_list = []
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    warnings_list = json.load(f)
            except json.JSONDecodeError:
                warnings_list = []
        
        # Add new warning entry
        warning_entry = {
            "timestamp": timestamp,
            "user_id": user_id,
            "user_name": user_name,
            "category": category,
            "message": message
        }
        warnings_list.append(warning_entry)
        
        # Save back to file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(warnings_list, f, indent=2, ensure_ascii=False)
        
        _logger.info(f"Warning saved to {filename}")
    except Exception as e:
        _logger.error(f"Failed to save warning to file: {e}")
