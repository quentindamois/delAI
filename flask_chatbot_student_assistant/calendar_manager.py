import re
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import List, Dict, Optional
from icalendar import Calendar, Event
# Setup logger
logger = logging.getLogger(__name__)
# Add console handler to ensure output is visible
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)



def parse_date_from_query(query: str) -> Optional[datetime]:
    """
    Extract date from user query.
    Supports: today, tomorrow, specific dates
    Returns datetime object or None
    """
    query_lower = query.lower()
    query_lower = re.sub(r'[?.!,]', '', query_lower)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Check for "today" or "aujourd'hui"
    if re.search(r'\b(today|aujourd\'?hui)\b', query_lower):
        return today
    
    # Check for "tomorrow" or "demain"
    if re.search(r'\b(tomorrow|demain)\b', query_lower):
        return today + timedelta(days=1)
    
    # Check for "J+n" pattern
    j_plus_match = re.search(r'j\+(\d+)', query_lower)
    if j_plus_match:
        days = int(j_plus_match.group(1))
        return today + timedelta(days=days)
    
    # Check for "in n days" pattern
    dans_match = re.search(r'in\s+(\d+)\s*(days?|j)\b', query_lower)
    if dans_match:
        days = int(dans_match.group(1))
        return today + timedelta(days=days)
    
    # "day after tomorrow" 
    if re.search(r"\bday after[-\s]?tomorrow\b", query_lower):
        return today + timedelta(days=2)
    
    # Check for date patterns: DD/MM, DD/MM/YYYY, YYYY-MM-DD
    date_patterns = [
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', '%d/%m/%Y'),  # DD/MM/YYYY
        (r'(\d{1,2})/(\d{1,2})', '%d/%m'),  # DD/MM (current year)
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d'),  # YYYY-MM-DD
    ]
    
    for pattern, date_format in date_patterns:
        match = re.search(pattern, query)
        if match:
            try:
                if date_format == '%d/%m':
                    # Add current year
                    date_str = f"{match.group(1)}/{match.group(2)}/{today.year}"
                    return datetime.strptime(date_str, '%d/%m/%Y')
                else:
                    return datetime.strptime(match.group(0), date_format)
            except ValueError:
                continue
    
    # Default to today if no date found
    logger.info("No date found in query, defaulting to today")
    return today


def read_ics_file(file_path: str = "./data/planning.ics") -> Optional[Calendar]:
    """Read and parse the ICS calendar file."""
    try:
        if not Path(file_path).exists():
            logger.warning(f"Calendar file not found: {file_path}")
            return None
        
        with open(file_path, 'rb') as f:
            cal = Calendar.from_ical(f.read())
        
        logger.info(f"Successfully loaded calendar from {file_path}")
        return cal
    except Exception as e:
        logger.error(f"Error reading calendar file: {e}")
        return None


def get_events_for_date(cal: Calendar, target_date: datetime) -> List[Dict]:
    """
    Get all events for a specific date from the calendar.
    Returns list of event dictionaries with summary, start, end, location, description.
    """
    events = []
    target_date_only = target_date.date()
    
    for component in cal.walk():
        if component.name == "VEVENT":
            event_start = component.get('dtstart').dt
            
            # Handle both datetime and date objects
            if hasattr(event_start, 'date'):
                event_date = event_start.date()
            else:
                event_date = event_start
            
            # Check if event is on target date
            if event_date == target_date_only:
                event_info = {
                    'summary': str(component.get('summary', 'No title')),
                    'start': event_start,
                    'end': component.get('dtend').dt if component.get('dtend') else None,
                    'location': str(component.get('location', '')) if component.get('location') else None,
                    'description': str(component.get('description', '')) if component.get('description') else None,
                }
                events.append(event_info)
    
    # Sort events by start time
    events.sort(key=lambda x: x['start'])
    return events


def format_events(events: List[Dict], date: datetime) -> str:
    """Format events list into a nice readable string."""
    if not events:
        return f"No events scheduled for {date.strftime('%A, %B %d, %Y')}."
    
    date_str = date.strftime('%A, %B %d, %Y')
    result = [f"Schedule for {date_str}:\n"]
    
    for i, event in enumerate(events, 1):
        # Format time
        if hasattr(event['start'], 'strftime'):
            time_str = event['start'].strftime('%H:%M')
            if event['end'] and hasattr(event['end'], 'strftime'):
                time_str += f" - {event['end'].strftime('%H:%M')}"
        else:
            time_str = "All day"
        
        # Build event line
        event_line = f"{i}. {time_str} - {event['summary']}"
        
        if event['location']:
            event_line += f" ({event['location']})"
        
        result.append(event_line)
        
        if event['description']:
            result.append(f"   {event['description']}")
    
    return "\n".join(result)


def get_planning(user_query: str) -> dict:
    """
    Main function to get planning information based on user query.
    Returns a dict with success, action, message.
    """
    logger.info(f"[ACTION] Getting planning information for query: {user_query[:100]}")
    
    # Parse date from query
    target_date = parse_date_from_query(user_query)
    logger.info(f"Parsed target date: {target_date}")
    if not target_date:
        return {
            "success": False,
            "action": "planning_failed",
            "message": "Could not understand the date in your request. Please specify 'today', 'tomorrow', or a specific date."
        }
    
    # Read calendar file
    cal = read_ics_file()
    if not cal:
        return {
            "success": False,
            "action": "planning_failed",
            "message": "Could not access the calendar file. Please contact support."
        }
    
    # Get events for the date
    events = get_events_for_date(cal, target_date)
    
    # Format the result
    formatted_schedule = format_events(events, target_date)

    if events == []:
        logger.info(f"No events found for {target_date.date()}")
        return {
            "success": True,
            "action": "planning_empty",
            "message": f"No events scheduled for {target_date.strftime('%A, %B %d, %Y')}.",
            "results": None,
            "events_count": 0,
            "date": target_date.isoformat()
        }
    
    return {
        "success": True,
        "action": "planning_retrieved",
        "message": "Schedule information found in the calendar.",
        "results": formatted_schedule,
        "events_count": len(events),
        "date": target_date.isoformat()
    }
