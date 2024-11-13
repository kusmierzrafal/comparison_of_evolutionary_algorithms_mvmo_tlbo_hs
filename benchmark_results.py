import os
import pandas as pd
from functools import partial

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
    result = content.rstrip().split('\n')

    if '.' in result:
        return round(float(result), 2)
    else:
        return 0


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
                    try:
                        result_df.iloc[i, run] = round(float(result[i]), 2)
                    except:
                        import pdb; pdb.set_trace()
            result_df.to_excel(f"./tests/result_tables/{algorithm}_{func}_{dim}.xlsx", index=False)
