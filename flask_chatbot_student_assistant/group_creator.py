import pandas as pd
import re
#from dotenv import load_dotenv
import gspread
import os
import requests
import re
import time
import random 


#load_dotenv()
# used to get the id to get link to spreadsheet : https://stackoverflow.com/questions/53021610/python-gspread-how-to-get-a-spreadsheet-url-path-in-after-i-create-it
#path_to_credential = os.environ["PATH_CREDENTIAL"]

path_to_credential = "./delai-484723-9b0f884a2fcb.json"

gc = gspread.service_account(filename=path_to_credential)


def create_group_spreadsheet(url_name, nb_team, group_size, name_project="group projet"):
    ### We first need to extract the link from the message ###
    try:
        ### We get the number of team to be filled ###
        #nb_team = (size_class //  group_size) + (1 if size_class %  group_size != 0 else 0)
        ### we create a new google sheet ###
        sh = gc.open_by_url(url_name)

        time.sleep(random.randint(1,3))
        ### We create a new Worksheet ###
        worksheet = sh.get_worksheet(0)
        worksheet.update_title(name_project)

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
        res = f"[SUCCESS] The google sheet has been edited"
    except Exception as e:
        res = f"[ERROR] The google sheet could not be edited due to an unexpected error"
    return res


def generate_csv(nb_team, size_team):
    tem_dict = dict(name_team=list(map(lambda a: f"team {a}", range(1, nb_team + 1))))
    for i in range(1, size_team + 1):
        tem_dict[f"group member {i}"] = [""] * nb_team
    res_df = pd.DataFrame(tem_dict)
    return res_df






def vary_instruction():
    list_instruction = ["I am sorry, i did not understand, could explain it in the format : 'number_of_group' groups of 'number_of_person' in each group.", "I only understand the format X group of Y with X being the number of group and Y the number of person"]
    return random.choice(list_instruction)



def create_group(text):
    regex_nb = r"(?P<number_of_team>\d+)?\s*([Gg]+[rR]+[Oo]+[uU]+[pP]+[eE]?|[Tt]+[eE]+[aA]+[mM]+)[sS]?\s*[Oo]?[fF]?\s*(?P<size_of_team>\d+)?"
    list_tag = ["number_of_team", "size_of_team"]
    regex_link = r".*(?P<ggl_sheet_link>https:\/\/docs\.google\.com\/spreadsheets\/d\/[\w-]+\/?).*"
    reg_out = re.search(regex_nb, text)
    reg_out_link = re.search(regex_link, text)
    if reg_out and not(reg_out.group("number_of_team") is None) and not(reg_out.group("size_of_team") is None) and reg_out_link:
        name_project_reg = r"(?:[tT]+[hH]+[eE]+\s+(?:(?:[Pp]+[Rr]+[oO]+[jJ]+[Ee]+[cC]+[Tt]+|[Pp]+[Rr]+[eE]+[sS]+[Ee]+[Nn]+[Tt]+[Aa]+[Tt]+[Ii]+[Oo]+[Nn]+)[sS]?)\s*(?:[iI]+[Nn]+|[oO]+[fF]+)\s*(?P<type_project>\b\S+\b)?|(?P=type_project)?\s+(?:(?:[Pp]+[Rr]+[oO]+[jJ]+[Ee]+[cC]+[Tt]+|[Pp]+[Rr]+[eE]+[sS]+[Ee]+[Nn]+[Tt]+[Aa]+[Tt]+[Ii]+[Oo]+[Nn]+)[sS]?))"
        identified_project = re.search(name_project_reg, text)
        name_project = "group project" if not(identified_project) else identified_project.group("type_project")
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
    