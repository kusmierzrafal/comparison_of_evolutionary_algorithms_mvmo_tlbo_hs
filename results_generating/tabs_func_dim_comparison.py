import os
import pandas as pd
from functools import partial
from decimal import Decimal

ALGORITHMS_OUR = ["MVMO", "TLBO", "HS"]
ALGORITHMS_SOTA = ["CEC2022_EA4eig_CORRECTED", "jSObinexpEig", "MTT_SHADE_CEC_CORR", "NL-SHADE-LBC", "NL-SHADE-RSP-MID"]
ALGORITHMS = ALGORITHMS_OUR + ALGORITHMS_SOTA

LATEX_TABLE_HEADER_2 = """
\\begin{{table}}[H] \label{{tab:tabela1}} \centering
\caption{{Porównanie algorytmu {algorithm} z innymi w oparciu o CEC2022 dla {dim} wymiarów.}}
\\begin{{tabular}} {{| c | c | c | c | c | c | c | c | c |}} \hline

"""

LATEX_TABLE_HEADER_1 = """
\\begin{{table}}[H] \label{{tab:tabela1}} \centering
\caption{{Porównanie algorytmu {algorithm} z innymi w oparciu o CEC2022 dla {dim} wymiarów.}}
\\begin{{tabular}} {{| c | c | c | c | c | c |}} \hline

"""

LATEX_TABLE_HEADER_DICT = {
    1: LATEX_TABLE_HEADER_1,
    2: LATEX_TABLE_HEADER_2
}

LATEX_TABLE_COLUMN_NAMES_2 = """
     & \multicolumn{{2}}{{|c|}}{{{first_alg}}} & \multicolumn{{3}}{{|c|}}{{{second_alg}}} & \multicolumn{{3}}{{|c|}}{{{third_alg}}} \\\\ \hline
     Funkcja & średnia & odchylenie & średnia & odchylenie & & średnia & odchylenie &  \\\\ \hline
     """

LATEX_TABLE_COLUMN_NAMES_1 = """
     & \multicolumn{{2}}{{|c|}}{{{first_alg}}} & \multicolumn{{3}}{{|c|}}{{{second_alg}}} \\\\ \hline
     Funkcja & średnia & odchylenie & średnia & odchylenie & \\\\ \hline
     """


LATEX_TABLE_COLUMN_NAMES_DICT = {
    1: LATEX_TABLE_COLUMN_NAMES_1,
    2: LATEX_TABLE_COLUMN_NAMES_2
}

LATEX_END_TABLE_1 = """
& \multicolumn{{2}}{{|c|}}{{+/-/=}} & \multicolumn{{3}}{{|c|}}{{{second_alg_sum}}} \\\\ \hline
\end{{tabular}}
\end{{table}}
"""

LATEX_END_TABLE_2 = """
& \multicolumn{{2}}{{|c|}}{{+/-/=}} & \multicolumn{{3}}{{|c|}}{{{second_alg_sum}}} & \multicolumn{{3}}{{|c|}}{{{third_alg_sum}}} \\\\ \hline
\end{{tabular}}
\end{{table}}
"""

LATEX_END_TABLE_DICT = {
    1: LATEX_END_TABLE_1,
    2: LATEX_END_TABLE_2
}

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

test_results = os.listdir(f'./tests/final_test/mvmo')
done = []

for result in test_results:
    result = result.split("__")[1].split("_")
    func = result[0]
    dim = result[3]
    run = result[4].split(".")[0]
    done.append((dim, func, run))
RESULT_DF = pd.DataFrame(done)
RESULT_DF.columns = ["dim", "func", "run"]


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


def get_result_temp(algorithm, row):
    result_df = pd.read_csv(f"./tests/{PATH_DICT[algorithm]}/{algorithm.lower()}_{row['func']}_{row['dim']}.txt", delim_whitespace=True, header=None)
    result = result_df[int(row['run']) - 1].iloc[15]
    return result


def stats(algorithm, dim):
    result_df = RESULTS_DICT[algorithm][RESULTS_DICT[algorithm].dim == dim]
    stats_df = result_df[["func", "result"]].groupby(["func"]).mean()
    stats_df = stats_df.merge(result_df[["func", "result"]].groupby(["func"]).std(), left_index=True, right_index=True)

    stats_df.columns = ["mean", "std"]
    stats_df = stats_df.reset_index()
    stats_df.func = stats_df.func.astype(int)
    stats_df = stats_df.sort_values(["func"])
    return stats_df


for algorithm in ALGORITHMS:
    get_result = partial(get_result_temp, algorithm)
    RESULTS_DICT[algorithm]["result"] = RESULTS_DICT[algorithm].apply(get_result, axis=1)


# for algorithm in ALGORITHMS:
#     for dim in ["20"]:
#         stats(algorithm, dim)

def wst(first_alg, second_alg, row):
    if row[f"{second_alg} mean"] * 1.05 < row[f"{first_alg} mean"]:
        return '+'
    elif row[f"{second_alg} mean"] > row[f"{first_alg} mean"] * 1.05:
        return '-'
    else:
        return '='


def get_wst(df, wst):
    try:
        return df[wst]
    except:
        return 0


def gen_table(first, all, dim):
    try:
        all.remove(first)
    except ValueError:
        pass
    num = len(all)
    comparison_df = stats(first, dim)
    comparison_df.columns = ['func'] + [first + ' ' + col for col in comparison_df.columns[1:]]
    for algorithm in all:
        to_compare_df = stats(algorithm, dim)
        to_compare_df = to_compare_df.applymap(lambda x: 0 if x <= 10**-8 else x)
        to_compare_df.columns = ['func'] + [algorithm + ' ' + col for col in to_compare_df.columns[1:]]
        comparison_df = comparison_df.merge(to_compare_df, left_on="func", right_on="func")
        comparison_df[algorithm + ' ' + 'WST'] = comparison_df.apply(partial(wst, first, algorithm), axis=1)

    print(LATEX_TABLE_HEADER_DICT[num].format(algorithm=first, dim=dim))
    if num == 1:
        print(LATEX_TABLE_COLUMN_NAMES_DICT[num].format(
            first_alg=first,
            second_alg=all[0],
        ).replace('_', '\\_')
              )
    elif num == 2:
        print(LATEX_TABLE_COLUMN_NAMES_DICT[num].format(
            first_alg=first,
            second_alg=all[0],
            third_alg=all[1],
        ).replace('_', '\\_')
              )
    cols_to_cast = [col for col in comparison_df.columns if col.endswith("std") or col.endswith("mean")]
    comparison_df[cols_to_cast] = comparison_df[cols_to_cast].applymap(lambda x: '%.2E' % Decimal(x))

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 9999)
    print(comparison_df.apply(lambda row: " & ".join([str(x) for x in row]) + "\\\\ \hline", axis=1).to_string(index=False).replace('.', ','))
    if num == 1:
        all_0_wst = comparison_df[all[0] + " WST"].value_counts()
        print(LATEX_END_TABLE_1.format(
            second_alg_sum=f"{get_wst(all_0_wst, '+')}/{get_wst(all_0_wst, '-')}/{get_wst(all_0_wst, '=')}",
        ))
    elif num == 2:
        all_0_wst = comparison_df[all[0] + " WST"].value_counts()
        all_1_wst = comparison_df[all[1] + " WST"].value_counts()
        print(LATEX_END_TABLE_2.format(
            second_alg_sum=f"{get_wst(all_0_wst, '+')}/{get_wst(all_0_wst, '-')}/{get_wst(all_0_wst, '=')}",
            third_alg_sum=f"{get_wst(all_1_wst, '+')}/{get_wst(all_1_wst, '-')}/{get_wst(all_1_wst, '=')}",
        ))


alg = "HS"
for dim in ["10", "20"]:
    gen_table(alg, ALGORITHMS_OUR, dim)
for dim in ["10", "20"]:
    gen_table(alg, ALGORITHMS_SOTA[:2], dim)
for dim in ["10", "20"]:
    gen_table(alg, ALGORITHMS_SOTA[2:4], dim)
for dim in ["10", "20"]:
    gen_table(alg, ALGORITHMS_SOTA[4:], dim)
