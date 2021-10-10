# 清洗 PreliminaryValidationSet_Info.csv -> test-set.csv

import csv


def process_row(row):
    # 原始的 gender, 1 表示 男, 2 表示 女
    # 当前的 gender, 0 表示 男, 1 表示 女
    row["gender"] = "0" if row["gender"] == "1" else "1"

    return False, row


with open("test-set.csv", "w", newline="") as test_set_file:
    with open(
        "./data/PreliminaryValidationSet_Info.csv", newline=""
    ) as raw_test_set_file:
        reader = csv.DictReader(raw_test_set_file)
        writer = csv.DictWriter(test_set_file, fieldnames=reader.fieldnames)

        writer.writeheader()
        for row in reader:
            should_skip, result_row = process_row(row)

            if not should_skip:
                writer.writerow(row)

