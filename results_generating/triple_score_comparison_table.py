import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from functools import partial
import bisect
import math

y = np.zeros(51)
log_range = np.logspace(-8, 3, 51).tolist()

# plt.plot(log_range, y, 'o')
# plt.show()

fes_dim_10_dict = {k: int(10 ** (k / 5 - 3) * 200000) for k in range(16)}
fes_dim_10_dict[16] = "FEterm"
FES_DIM_10 = pd.DataFrame(fes_dim_10_dict.values(), columns=["FES ="])
FES_DIM_10[[i for i in range(1, 31)]] = 0
FES_DIM_10.iloc[16, 1:] = 200000

fes_dim_20_dict = {k: int(20 ** (k / 5 - 3) * 1000000) for k in range(16)}
fes_dim_20_dict[16] = "FEterm"
FES_DIM_20 = pd.DataFrame(fes_dim_20_dict.values(), columns=["FES ="])
FES_DIM_20[[i for i in range(1, 31)]] = 0
FES_DIM_20.iloc[16, 1:] = 1000000

FES_RESULT_DICT = {
    "10": FES_DIM_10,
    "20": FES_DIM_20
}

FILES_TEMPLATES = {
    "mvmo": {
        "10": "mvmo_1_40_1_1_75__{func}_50_200000_10_{run}.txt",
        "20": "mvmo_1_40_1_1_75__{func}_50_1000000_20_{run}.txt"
    },
    "tlbo": {
        "10": "tlbo__{func}_60_1666_10_{run}.txt",
        "20": "tlbo__{func}_60_8333_20_{run}.txt"
    },
    "hs": {
        "10": "hs_0.93_0.25_0.18__{func}_50_200000_10_{run}.txt",
        "20": "hs_0.93_0.25_0.18__{func}_50_1000000_20_{run}.txt"
    }
}


def get_result_temp(algorithm, row):
    file = FILES_TEMPLATES[algorithm][row["dim"]].format(func=row["func"], run=row["run"])
    f = open(f"./tests/final_test/{algorithm}/{file}")
    content = f.read()
    result = content.rstrip().split('\n')

    if '.' in result:
        return round(float(result), 2)
    else:
        return 0

RESULTS = {
    "mvmo": {},
    "tlbo": {},
    "hs": {}
}

results = {
    "mvmo": {
        "10": [],
        "20": [],
        "all": []
    },
    "tlbo": {
        "10": [],
        "20": [],
        "all": []
    },
    "hs": {
        "10": [],
        "20": [],
        "all": []
    },
}

ALL_RUNS = {
    "10": 360,
    "20": 360,
    "all": 720
}

ALL_TRESHOLDS = {
    "10": 18360,
    "20": 18360,
    "all": 36720
}

ALL_EVALUATIONS = {
    "10": 12 * 30 * 200000,
    "20": 12 * 30 * 1000000,
}
ALL_EVALUATIONS["all"] = ALL_EVALUATIONS["10"] + ALL_EVALUATIONS["20"]

for algorithm in ["mvmo", "tlbo", "hs"]:
    for dim in ["10", "20"]:
        for func in range(1, 13):
            result_df = FES_RESULT_DICT[dim].copy(deep=True)
            for run in range(1, 31):
                file = FILES_TEMPLATES[algorithm][dim].format(func=func, run=run)
                f = open(f"./tests/final_test/{algorithm}/{file}")
                content = f.read()
                result = content.rstrip().split('\n')
                if '.' not in result[-1]:
                    result_df.iloc[16, run] = result[-1]
                    result = result[: -1]
                for i in range(len(result)):
                    result_df.iloc[i, run] = round(float(result[i]), 2)

            results[algorithm][dim].append(result_df)
            results[algorithm]["all"].append(result_df)
            # result_df.to_excel(f"./tests/result_tables/{algorithm}_{func}_{dim}.xlsx", index=False)

for algorithm in ["mvmo", "tlbo", "hs"]:
    for dim in ["10", "20", "all"]:
        found_solution = 0
        tresholds_achieved = 0
        evaluations_used = 0
        for result in results[algorithm][dim]:
            found_solution += (result.iloc[15] == 0).astype(int).sum()
            for run in range(1, 31):
                sol = result[run].iloc[15]
                bisect.insort(log_range, sol)
                ind = log_range.index(sol)
                tresholds_achieved += (51 - ind)
                del log_range[ind]
                evaluations_used += int(result[run].iloc[16])

        print(f"found solutions {algorithm} {dim} -> {math.ceil(found_solution / ALL_RUNS[dim] * 100)}%")
        print(f"achieved tresholds {algorithm} {dim} -> {math.ceil(tresholds_achieved / ALL_TRESHOLDS[dim] * 100)}%")
        print(f"evaluations left {algorithm} {dim} -> {math.ceil((ALL_EVALUATIONS[dim] - evaluations_used) / ALL_EVALUATIONS[dim] * 100)}%")
        RESULTS[algorithm][dim] = [math.ceil(found_solution / ALL_RUNS[dim] * 100), math.ceil(tresholds_achieved / ALL_TRESHOLDS[dim] * 100), math.ceil((ALL_EVALUATIONS[dim] - evaluations_used) / ALL_EVALUATIONS[dim] * 100)]

# copy RESULT to Dokumentacja / pierwsza tabela
# download data from gsheet as first_table.csv

result_df = pd.read_csv('first_table.csv')
result_df.columns = ["Algorithm"] + list(range(9))
result_df = result_df.applymap(lambda x: f'0{x}' if len(str(x)) == 1 else x)

print("""
\\begin{table}[!h] \label{tab:tabela1} \centering
\caption{TODO tytuł do uzupełnienia z artykułu Pana Promotora}
\\begin{tabular} {| c | c | c | c |} \hline
Algorytm & 10D & 20D & Wszystkie  \\\\ \hline
""")


for row in result_df.iterrows():
    print(f"{row[1]['Algorithm']} & {row[1][0]}\space{row[1][1]}\space{row[1][2]} & {row[1][3]}\space{row[1][4]}\space{row[1][5]} & {row[1][6]}\space{row[1][7]}\space{row[1][8]} \\\\ \hline")

print("""
\end{tabular}
\end{table}
""")

