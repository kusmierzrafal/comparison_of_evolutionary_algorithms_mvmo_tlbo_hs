import bisect
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log_range = np.logspace(-8, 3, 51).tolist()
mpl.use("TkAgg")
x = list(range(1, 17, 1))
ALGORITHMS_OUR = ["MVMO", "TLBO", "HS"]
ALGORITHMS_SOTA = [
    "CEC2022_EA4eig_CORRECTED",
    "jSObinexpEig",
    "MTT_SHADE_CEC_CORR",
    "NL-SHADE-LBC",
    "NL-SHADE-RSP-MID",
]
ALGORITHMS = ALGORITHMS_OUR + ALGORITHMS_SOTA
ALGORITHMS_MARKERS = dict(zip(ALGORITHMS, ['.', 'v', '1', '8', 's', 'p', '*', '+']))

PATH_DICT = {
    "MVMO": "result_tables",
    "TLBO": "result_tables",
    "HS": "result_tables",
    "CEC2022_EA4eig_CORRECTED": "other_best_results/Results",
    "jSObinexpEig": "other_best_results/Results",
    "MTT_SHADE_CEC_CORR": "other_best_results/Results",
    "NL-SHADE-LBC": "other_best_results/Results",
    "NL-SHADE-RSP-MID": "other_best_results/Results",
}

test_results = os.listdir(f"../tests/final_test/mvmo")
done = []

for result in test_results:
    result = result.split("__")[1].split("_")
    func = result[0]
    dim = result[3]
    run = result[4].split(".")[0]
    done.append((dim, func, run))
RESULT_DF = pd.DataFrame(done)
RESULT_DF.columns = ["dim", "func", "run"]

ALGORITHMS_LABELS = {
    "MVMO": "MVMO",
    "TLBO": "TLBO",
    "HS": "HS",
    "CEC2022_EA4eig_CORRECTED": "EA4eig",
    "jSObinexpEig": "jSObinexpEig",
    "MTT_SHADE_CEC_CORR": "MTT_SHADE",
    "NL-SHADE-LBC": "NL-SHADE-LBC",
    "NL-SHADE-RSP-MID": "NL-SHADE-RSP-MID",
}

RESULTS_DICT = {
    "MVMO": RESULT_DF.copy(deep=True),
    "TLBO": RESULT_DF.copy(deep=True),
    "HS": RESULT_DF.copy(deep=True),
    "CEC2022_EA4eig_CORRECTED": RESULT_DF.copy(deep=True),
    "jSObinexpEig": RESULT_DF.copy(deep=True),
    "MTT_SHADE_CEC_CORR": RESULT_DF.copy(deep=True),
    "NL-SHADE-LBC": RESULT_DF.copy(deep=True),
    "NL-SHADE-RSP-MID": RESULT_DF.copy(deep=True),
}

ALGORITHMS_NAME_DICT = {
    "MVMO": "mvmo",
    "TLBO": "tlbo",
    "HS": "hs",
    "CEC2022_EA4eig_CORRECTED": "CEC2022_EA4eig_CORRECTED",
    "jSObinexpEig": "jSObinexpEig",
    "MTT_SHADE_CEC_CORR": "MTT_SHADE_CEC_CORR",
    "NL-SHADE-LBC": "NL-SHADE-LBC",
    "NL-SHADE-RSP-MID": "NL-SHADE-RSP-MID",
}

LATEX_FIG = """
\\begin{{figure}}[H]
    \label{{fig:tradycyjne-logo-pw}}
    \centering \includegraphics[width=0.88\linewidth]{{eiti/agg_{dims}.png}}
    \caption{{Wykres przedstawiający zagregowane krzywe ECDF dla porównywanych algorytmów dla {display_dims}.}}
\end{{figure}}

"""


def display_dims(dims):
    if len(dims) == 1:
        return f"{dims[0]} wymiarów"
    else:
        return f"{dims[0]} oraz {dims[1]} wymiarów"


def log_range_achieved(val):
    bisect.insort(log_range, val)
    ind = log_range.index(val)
    del log_range[ind]
    return 51 - ind


def get_log_range_result(algorithm, dims):
    agg_result_df = pd.DataFrame()
    for dim in dims:
        for func in range(1, 13, 1):
            result_df = pd.read_csv(
                f"../tests/{PATH_DICT[algorithm]}/{ALGORITHMS_NAME_DICT[algorithm]}_{func}_{dim}.txt",
                delim_whitespace=True,
                header=None,
            )
            result_df = result_df.iloc[:16, :]
            result_df = result_df.applymap(log_range_achieved)
            agg_result_df = pd.concat([agg_result_df, result_df], axis=1)
    return agg_result_df.apply(lambda x: sum(x) / (len(x) * 51), axis=1).tolist()


for dims in [['10', '20'], ['10'], ['20']]:
    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(6)
    for algorithm in ALGORITHMS:
        ax.plot(x, get_log_range_result(algorithm, dims), label=ALGORITHMS_LABELS[algorithm], marker=ALGORITHMS_MARKERS[algorithm])
    ax.legend()
    plt.ylabel("proporcja zdobytych progów celu")
    plt.xlabel(
        "Numer zrzutu - k z wymiar ^ (k / 5 - 3) * maksymalna liczba ewaluacji"
    )
    plt.title(f"Zagregowany wykres dla {display_dims(dims)}")
    plt.savefig(f"../func_dim_all_algs_png/agg_{dims}.png")

    print(LATEX_FIG.format(dims=dims, display_dims=display_dims(dims)))
