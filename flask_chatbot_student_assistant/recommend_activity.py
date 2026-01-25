import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
CSV_ACTIVITY = "./activity_user.csv"


def update_csv_activity(text:str, user_id:str):
    df = pd.read_csv(CSV_ACTIVITY)
    user_id = str(user_id)
    if not(user_id in list(map(lambda a: str(a), df["StudentID"].to_list()))):
        df = add_user(user_id, df)
    df = update_user(user_id, text, df)
    df.to_csv(CSV_ACTIVITY, index=False)


# dataframe concatenation based on the example from https://www.geeksforgeeks.org/pandas/how-to-add-one-row-in-an-existing-pandas-dataframe/


def xor(a, b):
    return (not(a) and b) and (a and not(b)) 

def add_user(user_id, df):
    if not(user_id in df["StudentID"].to_list()):
        new_row = pd.DataFrame({"StudentID": [f"{user_id}"], "Age": [16], "StudyTimeWeekly":[0.0], "Absences":[0], "Tutoring":[0], "ParentalSupport":[0], "Extracurricular":[0], "Sports":[0], "Music":[0], "Volunteering":[0], "GPA":[0.0], "Activity":["_"]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df



dict_extract_binary_value = {
    "Tutoring":{
        "verb":["recieved", "have", "'m", "am", "helped"],
        "inner":["been", "being", "from", "by"],
        "negation":["never", "not", "n't"],
        "target_term":["tutor"]
    },
    "ParentalSupport":{
        "verb":["recieved", "have", "'m", "am", "helped"],
        "inner":["been", "being", "from", "by"],
        "negation":["never", "not", "n't"],
        "target_term":["panrental sup", "parental help", "help from my parent", "parent's help"]
    },
    "Extracurricular":{
        "verb":["have", "do", "doing", "enrolled", "registerd", "and"],
        "inner":["at", "in"],
        "negation":["never", "not", "n't"],
        "target_term":["extracurricular", "activities", "activities out of school"]
    },
    "Sports":{
        "verb":["have", "do", "doing", "enrolled", "registerd", "and"],
        "inner":["at", "in"],
        "negation":["never", "not", "n't"],
        "target_term":["sport", "physical activities", "physic"]
    },
    "Music":{
        "verb":["have", "do", "doing", "enrolled", "registerd", "and"],
        "inner":["at", "in"],
        "negation":["never", "not", "n't"],
        "target_term":["music", "music instrument", "musical instrument", "instrument"]
    },
    "Volunteering":{
        "verb":["have", "do", "doing", "enrolled", "registerd", "and"],
        "inner":["at", "in"],
        "negation":["never", "not", "n't"],
        "target_term":["voluteering", "voluteer", "charity", "charity work"]
    }
}

def flatten_unique(list_list):
    res = list()
    for l in list_list:
        res = res + l
    res = list(set(res))
    return res

dict_new_activity = {
    "verb":["have", "do", "doing", "enrolled", "registerd"],
    "negative_verb":["quit", "quitting", "stopped", "stop"],
    "inner":["at", "in", "at a", "in a", "doing"],
    "negation":["never", "not", "n't"],
    "target_term":list(filter(lambda a: not(a in ["_", ""]) and len(a) > 0, flatten_unique(pd.read_csv(CSV_ACTIVITY)["Activity"].apply(lambda a: a.split("|")).tolist())))
}


convert_morph_letter = lambda a: r"\s+" if a == " "  else rf"[{a.lower()}{a.upper()}]+"

def convert_word_regex(sentence):
    return rf"{r''.join(list(map(convert_morph_letter, map(lambda b: b[0], filter(lambda a:  sentence[a[1]] == a[0], zip(sentence, range(len(sentence))))))))}"

def regex_list_word(list_word):
    res = r"|".join(list(map(lambda a: convert_word_regex(a), list_word)))
    return res


def gen_dict_regex_activity(dict_list):
    return ([rf"(?:{regex_list_word(dict_list["negative_verb"])})\s+(?:{regex_list_word(dict_list["inner"])})?\s*(?P<Activies>{regex_list_word(dict_list["target_term"])})",
             rf"(?:{regex_list_word(dict_list["verb"])})\s*(?:{regex_list_word(dict_list["negation"])})\s+(?:{regex_list_word(dict_list["inner"])})?\s*(?P<Activies>{regex_list_word(dict_list["target_term"])})"], 
            [rf"(?:{regex_list_word(dict_list["verb"])})\s+(?:{regex_list_word(dict_list["inner"])})?\s*(?P<Activies>{regex_list_word(dict_list["target_term"])})"])


regex_activites = gen_dict_regex_activity(dict_new_activity)

def gen_list_regex_binary_extract(dict_extract):
    res_dict = dict()
    for key_bin_col in dict_extract.keys():
        regex_pos = rf"(?:{regex_list_word(dict_extract[key_bin_col]["verb"])})\s*(?:{regex_list_word(dict_extract[key_bin_col]["inner"])})?\s*(?:[aA]+\s+)?(?:{regex_list_word(dict_extract[key_bin_col]["target_term"])})"  
        regex_negative = rf"(?:{regex_list_word(dict_extract[key_bin_col]["verb"])})\s*(?:{regex_list_word(dict_extract[key_bin_col]["negation"])})\s+(?:{regex_list_word(dict_extract[key_bin_col]["inner"])})?\s*(?:[aA]+\s+)?(?:{regex_list_word(dict_extract[key_bin_col]["target_term"])})"
        res_dict[key_bin_col] = (regex_negative, regex_pos)
    return res_dict


dict_regex_extract_binary_value = gen_list_regex_binary_extract(dict_extract_binary_value)

def get_bin_column(text):
    res_list = list()
    for key_regex in dict_regex_extract_binary_value.keys():
        tem_regex_negative = re.search(dict_regex_extract_binary_value[key_regex][0], text)
        tem_regex_positive = re.search(dict_regex_extract_binary_value[key_regex][1], text)
        if not(tem_regex_negative is None):
            res_list.append((key_regex, 0))
        elif not(tem_regex_positive is None):
            res_list.append((key_regex, 1))
    return res_list

dict_regex_extract_value = {
    "Absences": ([r"[Aa]+[Bb]+[Ss]+[Ee]+[nN]+[Tt]+\s+(?:[fF]+[oO]+[rR]+\s+)?(?P<Absences>\d+)", r"(?P<Absences>\d+)\s*[Aa]+[Bb]+[Ss]+[Ee]+[nN]+[cC]+[eE]+[Ss]*"], lambda a: int(a)),
    "Age": ([r"(?P<Age>\d+)\s+[Yy]+[Ee]+[Aa]+[Rr]+[Ss]*\s+[Oo]+[Ll]+[dD]+", r"[Aa]*[mM]+\s+(?P<Age>\d+)"], lambda a: int(a)),
    "StudyTimeWeekly": ([r"(?:[sS][Tt][Uu][Dd])([Yy]+|[Ii][Ee][Dd])\s+(?:[fF]+[oO]+[rR]+|[Dd]+[Uu]+[Rr]+[Ii]+[nN]+[gG]+)?\s+(?:[eE]+[vV]+[eE]+[rR]+\s+[wW]+[eE]+[kK]+\s+)?(?P<StudyTimeWeekly>\d+\.?\d*)"], lambda a: float(a)),
    "GPA":([r"[Gg]+[Pp]+[Aa]+\s+(?:[Ii]+[Ss]+|[Ee]+[Qq]+[Uu]+[Aa]+[Ll]+\s+[Tt]+[Oo]+|[Oo]+[Ff]+)\s+(?P<GPA>\d+\.?\d*)"], lambda a: float(a)),
}

def get_value_column(text):
    res_list = list()
    for key_regex in dict_regex_extract_value.keys():
        for extract_regex in dict_regex_extract_value[key_regex][0]:
            tem_search = re.search(extract_regex, text)
            if not(tem_search is None):
                res_list.append((key_regex, dict_regex_extract_value[key_regex][1](tem_search.group(key_regex))))
    print(f"get_value_column: {res_list}")
    return res_list

def get_values_activies(text):
    list_new_acitivity = list()
    list_acitivity_to_be_removed = list()
    for sub_regex in regex_activites[1]:
        list_new_acitivity = list_new_acitivity + re.findall(sub_regex, text)
    for sub_regex in regex_activites[0]:
        print(f"list_acitivity_to_be_removed: {list_acitivity_to_be_removed}")
        list_acitivity_to_be_removed =  list_acitivity_to_be_removed + re.findall(sub_regex, text)
    list_new_acitivity = list(filter(lambda a: not(a in list_acitivity_to_be_removed), list_new_acitivity))
    print([("Activity", (list_acitivity_to_be_removed, list_new_acitivity))])
    return [("Activity", (list_acitivity_to_be_removed, list_new_acitivity))]

def get_list_update(text):
    return get_bin_column(text) + get_value_column(text) + get_values_activies(text)

# dataframe value update inpired from https://sqlpey.com/python/top-4-ways-to-update-row-values-in-pandas/#solution-3-the-power-of-update

def update_user_once(user_id, df, name_col, new_value):
    if name_col == "Activity":
        up_date_col = lambda a: "|".join(list(set(list(filter(lambda a: not(a in new_value[0]), a.split("|"))) + new_value[1])))
    else:
        up_date_col = lambda a: new_value
    print("old value")
    print(df.loc[df["StudentID"].apply(str) == user_id, name_col])
    df.loc[df["StudentID"].apply(str) == user_id, name_col] = df.loc[df["StudentID"].apply(str) == user_id, name_col].apply(up_date_col)
    print("new value")
    print(df.loc[df["StudentID"].apply(str) == user_id, name_col])
    print("expcted result")
    print(df.loc[df["StudentID"].apply(str) == user_id, name_col].apply(up_date_col))
    df = df
    return df

def update_user(user_id, text, df):
    #df = pd.read_csv(CSV_ACTIVITY)
    list_update = get_list_update(text)
    for tuple_name_value in list_update:
        print(f"tuple_name_value: {tuple_name_value}")
        df = update_user_once(user_id, df, tuple_name_value[0], tuple_name_value[1])
    #df.to_csv(CSV_ACTIVITY, index=False)
    return df


def recommand_activity(studentID, top=3):
    studentID = str(studentID)
    try:
        ### Load the csv ###
        df = pd.read_csv(CSV_ACTIVITY)
        ### seperate the user from the rest of user ###
        df_input = df.loc[df["StudentID"].apply(str) != studentID, df.columns != "StudentID"]
        array_input = df_input.to_numpy()
        row_student = df.loc[df["StudentID"].apply(str) == studentID, df.columns  != "StudentID"].to_numpy()
        user_already_hobby = row_student[0,-1]
        ### compute the sumilarity between the user and the rest ###
        list_similary_activity = list(zip(cosine_similarity(array_input[:,:-1], row_student[:,:-1]), array_input[:,-1]))
        list_similary_activity.sort(key=lambda a: a[0])
        tem_list_activity_recommanded = list()
        for i in range(top):
            list_from_ind = list_similary_activity[i][1].split("|")
            tem_list_activity_recommanded = tem_list_activity_recommanded + list_from_ind
        tem_list_activity_recommanded = list(set(tem_list_activity_recommanded))
        recommanded_new_activity = list(filter(lambda a: a != "_" and re.search(convert_word_regex(a), user_already_hobby) is None, tem_list_activity_recommanded))
        if len(recommanded_new_activity) > 0:
            res = {
                "success":True,
                "action":"recommendation_success",
                "message": f"[SUCESS] Here {'are' if len(recommanded_new_activity) > 1 else 'is'} recommended {'' if len(recommanded_new_activity) > 1 else 'a '}activite{'s' if len(recommanded_new_activity) > 1 else ''} : {' ,'.join(recommanded_new_activity)}."
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