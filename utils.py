import requests
import zipfile
import io
import numpy as np
import pystablemotifs as sm
import os

CELL_COLLECTIVE_BASIC_PATH = "https://research.cellcollective.org/api/"
DATA_PATH = "old/new/Networks"
SPECIAL_WORDS_KEYS = {"AND": "&&", "OR": "||", "NOT": "~", "=": "*=", "(": "(", ")": ")"}


def get_model_boolean_expressions(model_id, model_name):
    print(f"Try to get boolean expressions for model Id {model_id} name {model_name}")
    response = requests.get(f"{CELL_COLLECTIVE_BASIC_PATH}/model/{model_id}/export/version/1?type=EXPR", stream=True)
    if not response.ok:
        print(f"Failed to get boolean expressions for model {model_name}")
        return
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extract("expr/expressions.ALL.txt", get_network_data_path(model_name))
        zf.extract("expr/external_components.ALL.txt", get_network_data_path(model_name))


def get_model_truth_tables(model_id, model_name):
    print(f"Try to get truth tables for model Id {model_id} name {model_name}")
    response = requests.get(f"{CELL_COLLECTIVE_BASIC_PATH}/model/{model_id}/export/version/1?type=TT", stream=True)
    if not response.ok:
        print(f"Failed to get boolean expressions for model {model_name}")
        return
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for f in zf.infolist():
            zf.extract(f.filename, get_network_truth_table_data_path(model_name))


def parser_line(line):
    line = line.replace("/", "_").replace(";", "_").replace("+", "_plus_").replace("-", "_minus_")\
                .replace("NAD(P)H", "NAD_P_H").replace(",", "_").replace("C", "C_").replace("(i_o)", "i_o").replace("S", "S_")\
                .replace("(outtermb)", "_outtermb_").replace("(innermb)", "_innermb_").replace("L", "L_")\
                .replace("&", "_with_").replace(".", "_point_").replace("/", "_frac_").replace("G", "G_").replace("3", "n3")\
                .replace(" AND ", " & ").replace(" OR ", " | ").replace(" NOT ", "~ ").replace("=", "*=")
    line = line.strip()
    line = line.replace(" ", "")
    return f"{line}\n"


def parser_data(network_name):
    path = get_network_data_path(network_name)
    no_external_expression_file = open(f"{path}/expressions_clean_no_eternal.txt", "w")
    all_expression_file = open(f"{path}/expressions_clean.txt", "w")

    with open(f"{path}/expr/expressions.ALL.txt", "r") as fr:
        for line in fr.read().split('\n'):
            if line == "":
                continue
            no_external_expression_file.write(parser_line(line))
            all_expression_file.write(parser_line(line))

    with open(f"{path}/expr/external_components.ALL.txt", "r") as fr:
        for line in fr.read().split('\n'):
            if line == "":
                continue
            line = line.replace("/", "_").replace(";", "_").replace("+", "_plus_").replace("-", "_minus_")\
                .replace("NAD(P)H", "NAD_P_H").replace(",", "_").replace("C", "C_").replace("(i_o)", "i_o").replace("S", "S_")\
                .replace("(outtermb)", "_outtermb_").replace("(innermb)", "_innermb_").replace("L", "L_")\
                .replace("&", "_with_").replace(".", "_point_").replace("/", "_frac_").replace("G", "G_").replace("3", "n3")
            line = line.strip()
            line = line.replace(" ", "")
            all_expression_file.write(f"{line}*={line}\n")
    no_external_expression_file.close()
    all_expression_file.close()


def get_model_primes(network_name, path=None):
    print(f"Get primes for model {network_name}.")
    # path = f"{get_network_data_path(network_name)}/primes.npy"
    # if os.path.exists(path):
    #     return np.load(path, allow_pickle=True).item()
    path = f"{get_network_data_path(network_name)}/expressions_clean_no_eternal.txt" if path is None else path
    primes = sm.format.import_primes(path)
    np.save(path, primes, allow_pickle=True)
    return primes


def get_network_data_path(network_name):
    return f"{DATA_PATH}/{network_name.replace('-', '').replace('.', '').replace(' ', '_')}/Boolean_Functions"


def get_network_truth_table_data_path(network_name):
    return f"{DATA_PATH}/{network_name.replace('-', '').replace('.', '').replace(' ', '_')}/Truth_Tables"


def get_all_models_boolean_expressions():
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    path = CELL_COLLECTIVE_BASIC_PATH + "model/cards/research?category=published&modelTypes=boolean&cards=79"
    response = requests.get(path, stream=True)
    if not response.ok:
        print("Failed to get all cell collective boolean models information")
        return
    models_list = response.json()
    for model in models_list:
        model_name = model["model"]["name"]
        model_id = model["model"]["id"]
        if not os.path.exists(get_network_data_path(model_name)):
            get_model_boolean_expressions(model_id, model_name)
            parser_data(model_name)


def get_all_models_primes():
    for model_name in os.listdir(DATA_PATH):
        if not os.path.exists(f"{DATA_PATH}/{model_name}/Boolean_Functions/primes.npy"):
            get_model_primes(model_name)


