# 清洗 TrainingAnnotation.csv -> training-set.csv

import csv

after_clean_fileds = [
    "patient ID",
    "gender",
    "age",
    "diagnosis",
    "preVA",
    "anti-VEGF",
    "preCST",
    "preIRF",
    "preSRF",
    "prePED",
    "preHRF",
    "VA",
    "continue injection",
    "CST",
    "IRF",
    "SRF",
    "PED",
    "HRF",
]


def process_row(row):
    try:
        # 原始的 gender, 1 表示 男, 2 表示 女
        # 当前的 gender, 0 表示 男, 1 表示 女
        row["gender"] = "0" if row["gender"] == "1" else "1"

        if row["preVA"] == "nan" or float(row["preVA"]) < 0:
            return True, []

        fields = "preCST, preIRF, preSRF, prePED, preHRF".split(", ")
        for field in fields:
            if row[field] == "nan":
                return True, []

        if row["VA"] == "nan" or float(row["VA"]) < 0:
            return True, []

        if row["continue injection"] != "0" and row["continue injection"] != "1":
            return True, []

        fields = "CST, IRF, SRF, PED, HRF".split(", ")
        for field in fields:
            if row[field] == "nan":
                return True, []
    except:
        pass

    return False, row


with open("training-set.csv", "w", newline="") as training_set_file:
    with open("./raw-data/TrainingAnnotation.csv", newline="") as raw_training_set_file:
        reader = csv.DictReader(raw_training_set_file)
        writer = csv.DictWriter(training_set_file, fieldnames=reader.fieldnames)

        writer.writeheader()
        for row in reader:
            should_skip, result_row = process_row(row)

            if not should_skip:
                writer.writerow(row)

