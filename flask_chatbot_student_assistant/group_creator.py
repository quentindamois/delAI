import pandas as pd
import re
#from dotenv import load_dotenv
import gspread
import os
import requests
import re
import time
import random 
import json
import logging
import sys
import traceback
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)
# Add console handler to ensure output is visible
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

path_to_credential = "./delai-484723-9b0f884a2fcb.json"

print(f"[INIT] Attempting to load Google Sheets credentials from: {path_to_credential}")
try:
    gc = gspread.service_account(filename=path_to_credential)
    print(f"[INIT] Google Sheets credentials loaded successfully")
except Exception as e:
    print(f"[INIT] ERROR - Failed to load credentials: {str(e)}")
    print(f"[INIT] Traceback: {traceback.format_exc()}")
    logger.error(f"Failed to load Google Sheets credentials: {str(e)}", exc_info=True)


def save_group_link(group_link: str) -> bool:
    """Save the group spreadsheet link with a timestamp to data/group_links/."""
    try:
        group_links_dir = "./data/group_links"
        os.makedirs(group_links_dir, exist_ok=True)
        
        # Create filename based on timestamp
        timestamp = datetime.now().isoformat()
        filename = os.path.join(group_links_dir, "group_links.json")
        
        # Load existing links or create new list
        links_data = []
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                links_data = json.load(f)
        
        # Add new link entry
        links_data.append({
            "link": group_link,
            "timestamp": timestamp
        })
        
        # Save back to file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(links_data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save group link: {e}")
        return False

def create_group_spreadsheet(url_name, nb_team, group_size, name_project="group projet"):
    """We put the a row for each group and a column for each group member"""
    ### We first need to extract the link from the message ###
    try:
        ### We get the number of team to be filled ###
        #nb_team = (size_class //  group_size) + (1 if size_class %  group_size != 0 else 0)
        ### we create a new google sheet ###
        logger.info(f"[INFO] Opening Google Sheet at: {url_name}...")

        sh = gc.open_by_url(url_name)
        logger.info(f"[INFO] Google Sheet opened successfully: {url_name}")

        time.sleep(random.randint(1,3))
        ### We create a new Worksheet ###
        worksheet = sh.get_worksheet(0)
        worksheet.update_title(name_project)

        logger.info(f"[INFO] Worksheet titled '{name_project}' created successfully.")

        time.sleep(random.randint(1,3))
        ### We put the name of the project at the top right corner of the spread sheet ###
        worksheet.update_cell(1, 1, name_project)

        ### We add the name of theam on the left of the worksheet ###
        for i in range(2, nb_team + 2):
            time.sleep(random.randint(1,3))
            worksheet.update_cell(i, 1, f"team {i - 1}")

        ### We add the name for the column of each members ###
        for i in range(2, group_size + 2):
            time.sleep(random.randint(1,3))
            worksheet.update_cell(1, i, f"student {i - 1}")
        
        logger.info(f"[INFO] Saving group link...")
        # Save the group link with timestamp
        if save_group_link(url_name):
            res = f"[SUCCESS] The google sheet has been edited."
            logger.info(f"[INFO] {res}")
        else:
            res = f"[SUCCESS] The google sheet has been edited (but link could not be saved)."
            logger.info(f"[INFO] {res}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to edit google sheet: {str(e)}")
        logger.error(f"[ERROR] Exception type: {type(e).__name__}")
        logger.error(f"[ERROR] Traceback: {traceback.format_exc()}")
        logger.error(f"[ERROR] Failed to edit google sheet: {str(e)}", exc_info=True)
        res = f"[ERROR] The google sheet could not be edited due to an unexpected error: {str(e)}"
    return res


def vary_instruction():
    """select a random version of the instruction to have more variety"""
    list_instruction = ["I am sorry, i did not understand, could explain it in the format : 'number_of_group' groups of 'number_of_person' in each group.", "I only understand the format X group of Y with X being the number of group and Y the number of person"]
    return random.choice(list_instruction)


def create_group(text):
    """Parse user input to extract group info and create spreadsheet."""
    ### This regex is used to get the number of team and their size ###
    regex_nb = r"(?P<number_of_team>\d+)?\s*([Gg]+[rR]+[Oo]+[uU]+[pP]+[eE]?|[Tt]+[eE]+[aA]+[mM]+)[sS]?\s*[Oo]?[fF]?\s*(?P<size_of_team>\d+)?"
    list_tag = ["number_of_team", "size_of_team"]
    ### This regex is used to extract the link to the google sheet ###
    regex_link = r".*(?P<ggl_sheet_link>https:\/\/docs\.google\.com\/spreadsheets\/d\/[\w-]+\/?).*"
    reg_out = re.search(regex_nb, text)
    reg_out_link = re.search(regex_link, text)
    if reg_out and not(reg_out.group("number_of_team") is None) and not(reg_out.group("size_of_team") is None) and reg_out_link:
        ### If we have the number of teams, the number of person per team and the link we will try to see if the project or the presentation has a name
        name_project_reg = r"(?:[tT]+[hH]+[eE]+\s+(?:(?:[Pp]+[Rr]+[oO]+[jJ]+[Ee]+[cC]+[Tt]+|[Pp]+[Rr]+[eE]+[sS]+[Ee]+[Nn]+[Tt]+[Aa]+[Tt]+[Ii]+[Oo]+[Nn]+)[sS]?)\s*(?:[iI]+[Nn]+|[oO]+[fF]+)\s*(?P<type_project>\b\S+\b)?|(?P=type_project)?\s+(?:(?:[Pp]+[Rr]+[oO]+[jJ]+[Ee]+[cC]+[Tt]+|[Pp]+[Rr]+[eE]+[sS]+[Ee]+[Nn]+[Tt]+[Aa]+[Tt]+[Ii]+[Oo]+[Nn]+)[sS]?))"
        identified_project = re.search(name_project_reg, text)
        # Ensure name_project is never None or empty
        name_project = "group project"
        if identified_project and identified_project.group("type_project"):
            name_project = identified_project.group("type_project")
        tem = create_group_spreadsheet(reg_out_link.group("ggl_sheet_link"), int(reg_out.group("number_of_team")), int(reg_out.group("size_of_team")), name_project)
        res = {"success":"[SUCCESS]" in tem, "action":"group_created" if "[SUCCESS]" in tem else "group_failed", "message":tem}
    elif not(reg_out):
        res = {"success":False, "action":"group_failed", "message":f"[Error] {vary_instruction()} The number of group and the size of group have not been given." + ("" if reg_out_link else " No link to a google sheet was provided.")}
        return res
    elif reg_out and (reg_out.group("number_of_team") is None or reg_out.group("size_of_team") is None):
        list_tag_no_int = list(map(lambda b: re.sub("_", " ", b), filter(lambda a: reg_out.group(a) is None, list_tag)))
        res = {"success":False, "action":"group_failed", "message":f"[Error] {vary_instruction()} The following information{ 's are' if len(list_tag_no_int) > 1 else ' is'} missing : {','.join(list_tag_no_int)}."+ ("" if reg_out_link else " No link to a google sheet was provided.")}
    elif not(reg_out_link):
        res = {"success":False, "action":"group_failed", "message":f"[Error] No link to a google sheet was provided."}
    else:
        res = {"success":False, "action":"group_failed", "message":f"[Error] {vary_instruction()} I could not fill the google sheet."}
    return res
    