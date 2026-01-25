from fastapi import logger
import logging
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import sys
CSV_ACTIVITY = "./activity_user.csv"

logger = logging.getLogger(__name__)
# Add console handler to ensure output is visible
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

#############################################
##########                         ##########
########## The generation of regex ##########
##########                         ##########
#############################################


### The dictionnary containing the different keyword for each column than can be 1 or 0. ###

dict_extract_binary_value = {
    "Tutoring":{
        "verb":["received", "have", "'m", "am", "helped"],
        "inner":["been", "being", "from", "by"],
        "negation":["never", "not", "n't", "hate"],
        "target_term":["tutor"]
    },
    "ParentalSupport":{
        "verb":["received", "have", "'m", "am", "helped"],
        "inner":["been", "being", "from", "by"],
        "negation":["never", "not", "n't", "hate"],
        "target_term":["parental sup", "parental help", "help from my parent", "parent's help"]
    },
    "Extracurricular":{
        "verb":["have", "do", "doing", "enrolled", "registered", "and"],
        "inner":["at", "in"],
        "negation":["never", "not", "n't", "hate"],
        "target_term":["extracurricular", "activities", "activities out of school"]
    },
    "Sports":{
        "verb":["have", "do", "doing", "enrolled", "registered", "and"],
        "inner":["at", "in"],
        "negation":["never", "not", "n't", "hate"],
        "target_term":["sport", "physical activities", "physic"]
    },
    "Music":{
        "verb":["have", "do", "doing", "enrolled", "registered", "and"],
        "inner":["at", "in"],
        "negation":["never", "not", "n't", "hate"],
        "target_term":["music", "music instrument", "musical instrument", "instrument"]
    },
    "Volunteering":{
        "verb":["have", "do", "doing", "enrolled", "registered", "and"],
        "inner":["at", "in"],
        "negation":["never", "not", "n't", "hate"],
        "target_term":["volunteering", "volunteer", "charity", "charity work"]
    }
}



def flatten_unique(list_list:list)-> list:
    """This function is used to turn a list of list into a list"""
    res = list()
    for l in list_list:
        res = res + l
    res = list(set(res))
    return res


### The dictionnary containing the different keyword to identify if the user has started or stopped a new activity ###

dict_new_activity = {
    "verb":["have", "do", "doing", "enrolled", "registered"],
    "negative_verb":["quit", "quitting", "stopped", "stop"],
    "inner":["at", "in", "at a", "in a", "doing"],
    "negation":["never", "not", "n't"],
    "target_term":list(filter(lambda a: not(a in ["_", ""]) and len(a) > 0, flatten_unique(pd.read_csv(CSV_ACTIVITY)["Activity"].apply(lambda a: a.split("|")).tolist())))
}

### This function is used to generate a regex for a single character ###
convert_morph_letter = lambda a: r"\s+" if a == " "  else rf"[{a.lower()}{a.upper()}]+"



def convert_word_regex(sentence):
    """This fonction generate a regex for a word"""
    return rf"{r''.join(list(map(convert_morph_letter, map(lambda b: b[0], filter(lambda a:  sentence[a[1]] == a[0], zip(sentence, range(len(sentence))))))))}"

def regex_list_word(list_word):
    """This function create a regex for detection a list of word"""
    res = r"|".join(list(map(lambda a: convert_word_regex(a), list_word)))
    return res


def gen_dict_regex_activity(dict_list):
    """This function is used to gereate the regex to detect when the user start or stop an activity"""
    return ([rf"(?:{regex_list_word(dict_list["negative_verb"])})\s+(?:{regex_list_word(dict_list["inner"])})?\s*(?P<Activies>{regex_list_word(dict_list["target_term"])})",
             rf"(?:{regex_list_word(dict_list["verb"])})\s*(?:{regex_list_word(dict_list["negation"])})\s+(?:{regex_list_word(dict_list["inner"])})?\s*(?P<Activies>{regex_list_word(dict_list["target_term"])})"], 
            [rf"(?:{regex_list_word(dict_list["verb"])})\s+(?:{regex_list_word(dict_list["inner"])})?\s*(?P<Activies>{regex_list_word(dict_list["target_term"])})"])

### create the regex for the activities ###
regex_activites = gen_dict_regex_activity(dict_new_activity)



def gen_list_regex_binary_extract(dict_extract):
    """This function generate the regex for the columns that contain either 1 or 0"""
    res_dict = dict()
    for key_bin_col in dict_extract.keys():
        regex_pos = rf"(?:{regex_list_word(dict_extract[key_bin_col]["verb"])})\s*(?:{regex_list_word(dict_extract[key_bin_col]["inner"])})?\s*(?:[aA]+\s+)?(?:{regex_list_word(dict_extract[key_bin_col]["target_term"])})"  
        regex_negative = rf"(?:{regex_list_word(dict_extract[key_bin_col]["verb"])})\s*(?:{regex_list_word(dict_extract[key_bin_col]["negation"])})\s+(?:{regex_list_word(dict_extract[key_bin_col]["inner"])})?\s*(?:[aA]+\s+)?(?:{regex_list_word(dict_extract[key_bin_col]["target_term"])})"
        res_dict[key_bin_col] = (regex_negative, regex_pos)
    return res_dict

### create the regex for the columns that contain either 1 or 0 ###
dict_regex_extract_binary_value = gen_list_regex_binary_extract(dict_extract_binary_value)



### Create the regex for the columns that contain a integer or a float ###
dict_regex_extract_value = {
    "Absences": ([r"[Aa]+[Bb]+[Ss]+[Ee]+[nN]+[Tt]+\s+(?:[fF]+[oO]+[rR]+\s+)?(?P<Absences>\d+)", r"(?P<Absences>\d+)\s*[Aa]+[Bb]+[Ss]+[Ee]+[nN]+[cC]+[eE]+[Ss]*"], lambda a: int(a)),
    "Age": ([r"(?P<Age>\d+)\s+[Yy]+[Ee]+[Aa]+[Rr]+[Ss]*\s+[Oo]+[Ll]+[dD]+", r"[Aa]*[mM]+\s+(?P<Age>\d+)"], lambda a: int(a)),
    "StudyTimeWeekly": ([r"(?:[sS][Tt][Uu][Dd])([Yy]+|[Ii][Ee][Dd])\s+(?:[fF]+[oO]+[rR]+|[Dd]+[Uu]+[Rr]+[Ii]+[nN]+[gG]+)?\s+(?:[eE]+[vV]+[eE]+[rR]+\s+[wW]+[eE]+[kK]+\s+)?(?P<StudyTimeWeekly>\d+\.?\d*)"], lambda a: float(a)),
    "GPA":([r"[Gg]+[Pp]+[Aa]+\s+(?:[Ii]+[Ss]+|[Ee]+[Qq]+[Uu]+[Aa]+[Ll]+\s+[Tt]+[Oo]+|[Oo]+[Ff]+)\s+(?P<GPA>\d+\.?\d*)"], lambda a: float(a)),
}

##############################################################
##########                                          ##########
########## Extracting the information from the text ##########
##########                                          ##########
##############################################################

def get_bin_column(text):
    """Extract the value for the column that contain either a 1 or a 0"""
    res_list = list()
    for key_regex in dict_regex_extract_binary_value.keys():
        tem_regex_negative = re.search(dict_regex_extract_binary_value[key_regex][0], text)
        tem_regex_positive = re.search(dict_regex_extract_binary_value[key_regex][1], text)
        if not(tem_regex_negative is None):
            ### We change the value to 0 if the information about that column is negative ###
            res_list.append((key_regex, 0))
        elif not(tem_regex_positive is None):
            ### We change the value to 1 if the information about that column is positive ###
            res_list.append((key_regex, 1))
    return res_list

def get_value_column(text):
    """Extract the value for the column that contain a float or an integer"""
    res_list = list()
    ### for each column containing an integer or a float we use a regex to extract a new value ###
    for key_regex in dict_regex_extract_value.keys():
        ### we iterate over each regex for one column ###
        for extract_regex in dict_regex_extract_value[key_regex][0]:
            tem_search = re.search(extract_regex, text)
            if not(tem_search is None):
                res_list.append((key_regex, dict_regex_extract_value[key_regex][1](tem_search.group(key_regex))))
    logger.info(f"get_value_column: {res_list}")
    return res_list

def get_values_activies(text):
    list_new_activity = list()
    list_activity_to_be_removed = list()
    for sub_regex in regex_activites[1]:
        list_new_activity = list_new_activity + re.findall(sub_regex, text)
    for sub_regex in regex_activites[0]:
        logger.info(f"list_activity_to_be_removed: {list_activity_to_be_removed}")
        list_activity_to_be_removed =  list_activity_to_be_removed + re.findall(sub_regex, text)
    list_new_activity = list(filter(lambda a: not(a in list_activity_to_be_removed), list_new_activity))
    logger.info([("Activity", (list_activity_to_be_removed, list_new_activity))])
    return [("Activity", (list_activity_to_be_removed, list_new_activity))]


######################################################################################
##########                                                                  ##########
########## updating the user information based on the information extracted ##########
##########                                                                  ##########
######################################################################################


def get_list_update(text):
    """Create the list of column that have to be updated"""
    return get_bin_column(text) + get_value_column(text) + get_values_activies(text)

# dataframe value update inpired from https://sqlpey.com/python/top-4-ways-to-update-row-values-in-pandas/#solution-3-the-power-of-update

def update_user_once(user_id, df, name_col, new_value):
    """update one column for one user"""
    ### We choose the function to update the column ###
    if name_col == "Activity":
        up_date_col = lambda a: "|".join(list(set(list(filter(lambda a: not(a in new_value[0]), a.split("|"))) + new_value[1])))
    else:
        up_date_col = lambda a: new_value
    logger.info("old value")
    logger.info(df.loc[df["StudentID"].apply(str) == user_id, name_col])
    df.loc[df["StudentID"].apply(str) == user_id, name_col] = df.loc[df["StudentID"].apply(str) == user_id, name_col].apply(up_date_col)
    logger.info("new value")
    logger.info(df.loc[df["StudentID"].apply(str) == user_id, name_col])
    logger.info("expcted result")
    logger.info(df.loc[df["StudentID"].apply(str) == user_id, name_col].apply(up_date_col))
    df = df
    return df

def update_user(user_id, text, df):
    """Update the information of a user based on the information inside of the user message"""
    ### We get the list of column to update and the new value ###
    list_update = get_list_update(text)
    ### We perform the update the column ###
    for tuple_name_value in list_update:
        logger.info(f"tuple_name_value: {tuple_name_value}")
        df = update_user_once(user_id, df, tuple_name_value[0], tuple_name_value[1])
    return df



def update_csv_activity(text:str, user_id:str):
    """We update the csv containing the activity of each user"""
    df = pd.read_csv(CSV_ACTIVITY)
    user_id = str(user_id)
    ### If the id is not in the csv we create a new row ###
    if not(user_id in list(map(lambda a: str(a), df["StudentID"].to_list()))):
        df = add_user(user_id, df)
    ### We update the column of the user
    df = update_user(user_id, text, df)
    ### we save csv ###
    df.to_csv(CSV_ACTIVITY, index=False)


# dataframe concatenation based on the example from https://www.geeksforgeeks.org/pandas/how-to-add-one-row-in-an-existing-pandas-dataframe/

def add_user(user_id, df):
    ### verify to see if the id is inside of already added student id ###
    if not(user_id in df["StudentID"].to_list()):
        ### Create a new row ###
        new_row = pd.DataFrame({"StudentID": [f"{user_id}"], "Age": [16], "StudyTimeWeekly":[0.0], "Absences":[0], "Tutoring":[0], "ParentalSupport":[0], "Extracurricular":[0], "Sports":[0], "Music":[0], "Volunteering":[0], "GPA":[0.0], "Activity":["_"]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df


#############################################
##########                         ##########
########## The prediction function ##########
##########                         ##########
#############################################

def recommand_activity(studentID, top=3):
    """Get the list of activity of individual the most similar profil"""
    studentID = str(studentID)
    try:
        ### Load the csv ###
        df = pd.read_csv(CSV_ACTIVITY)
        ### seperate the user from the rest of user ###
        df_input = df.loc[df["StudentID"].apply(str) != studentID, df.columns != "StudentID"]
        array_input = df_input.to_numpy()
        row_student = df.loc[df["StudentID"].apply(str) == studentID, df.columns  != "StudentID"].to_numpy()
        user_already_hobby = row_student[0,-1]
        ### compute the sumilarity between the user and the rest with the cosine similarity ###
        list_similary_activity = list(zip(cosine_similarity(array_input[:,:-1], row_student[:,:-1]), array_input[:,-1]))
        list_similary_activity.sort(key=lambda a: a[0], reverse=True)
        tem_list_activity_recommanded = list()
        #### We collect the list of activity ###
        for i in range(top):
            list_from_ind = list_similary_activity[i][1].split("|")
            tem_list_activity_recommanded = tem_list_activity_recommanded + list_from_ind
        ### We remove duplicate ###
        tem_list_activity_recommanded = list(set(tem_list_activity_recommanded))
        ### We get the list of recommmended activity ###
        recommanded_new_activity = list(filter(lambda a: a != "_" and re.search(convert_word_regex(a), user_already_hobby) is None, tem_list_activity_recommanded))
        if len(recommanded_new_activity) > 0:
            res = {
                "success":True,
                "action":"recommendation_success",
                "message": f"[SUCESS] Here {'are' if len(recommanded_new_activity) > 1 else 'is'} recommended {'' if len(recommanded_new_activity) > 1 else 'a '} activit{'ies' if len(recommanded_new_activity) > 1 else 'y'} : {', '.join(recommanded_new_activity)}."
            }
        else:
            res = {
                "success":False,
                "action":"recommendation_failed",
                "message": f"[ERROR] I did not find any recommendation for you."
            }
    except Exception as e:
        res = {
            "success":False,
            "action":"recommendation_failed",
            "message": f"[ERROR] Encoutered the following error when doing the recommendation : {e}."
        }
    return res