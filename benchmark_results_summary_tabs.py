import os
import pandas as pd
from functools import partial


LATEX_TABLE_START = """
\\begin{{table}}[!h] \label{{tab:tabela1}} \centering
\caption{{Rezultat dla {dim} wymiarów dla {algorithm}}}
\\begin{{tabular}} {{| c | c | c | c | c | c |}} \hline
    Func. & Najlepszy & Najsłabszy & Mediana & Średni & Odchylenie \\\\ \hline

"""

LATEX_TABLE_BODY = """
{func} & {min} & {max} & {median} & {mean} & {std} \\\\ \hline"""

LATEX_END_TABLE = f"""

\end{{tabular}}
\end{{table}}
"""

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
    result = content.rstrip().split('\n')[-1]

    if '.' in result:
        return round(float(result), 2)
    else:
        return 0


def gen_latex_table(algorithm, stats_df):

    for dim in ["10", "20"]:
        latex_table = ""
        latex_table += LATEX_TABLE_START.format(dim=dim, algorithm=algorithm.upper())
        dim_stats_df = stats_df[stats_df["dim"] == dim]

        for row in dim_stats_df.iterrows():
            latex_table += LATEX_TABLE_BODY.format(
                func=int(row[1]["func"]),
                min=str(round(row[1]["min"], 2)).replace(".", ","),
                max=str(round(row[1]["max"], 2)).replace(".", ","),
                median=str(round(row[1]["median"], 2)).replace(".", ","),
                mean=str(round(row[1]["mean"], 2)).replace(".", ","),
                std=str(round(row[1]["std"], 2)).replace(".", ","),
            )
        latex_table += LATEX_END_TABLE

        print(latex_table)


def stats(algorithm):

    get_result = partial(get_result_temp, algorithm)

    test_results = os.listdir(f'./tests/final_test/{algorithm}')
    done = []

    for result in test_results:
        result = result.split("__")[1].split("_")
        func = result[0]
        dim = result[3]
        run = result[4].split(".")[0]
        done.append((dim, func, run))

    result_df = pd.DataFrame(done)
    result_df.columns = ["dim", "func", "run"]
    result_df["result"] = result_df.apply(get_result, axis=1)

    stats_df = result_df[["dim", "func", "result"]].groupby(["dim", "func"]).min()
    stats_df = stats_df.merge(result_df[["dim", "func", "result"]].groupby(["dim", "func"]).max(), left_index=True, right_index=True)
    stats_df = stats_df.merge(result_df[["dim", "func", "result"]].groupby(["dim", "func"]).median(), left_index=True, right_index=True)
    stats_df = stats_df.merge(result_df[["dim", "func", "result"]].groupby(["dim", "func"]).mean(), left_index=True, right_index=True)
    stats_df = stats_df.merge(result_df[["dim", "func", "result"]].groupby(["dim", "func"]).std(), left_index=True, right_index=True)
    stats_df.columns = ["min", "max", "median", "mean", "std"]
    stats_df = stats_df.reset_index()
    stats_df.func = stats_df.func.astype(int)
    stats_df = stats_df.sort_values(["dim", "func"])
    gen_latex_table(algorithm, stats_df)


stats("mvmo")
stats("tlbo")
stats("hs")
