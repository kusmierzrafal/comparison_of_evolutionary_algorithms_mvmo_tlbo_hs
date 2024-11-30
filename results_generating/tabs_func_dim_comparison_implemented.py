import os
import pandas as pd
from functools import partial
from decimal import Decimal
from scipy.stats import wilcoxon

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 9999)


ALGORITHMS_OUR = ["MVMO", "TLBO", "HS"]
ALGORITHMS_SOTA = ["CEC2022_EA4eig_CORRECTED", "jSObinexpEig", "MTT_SHADE_CEC_CORR", "NL-SHADE-LBC", "NL-SHADE-RSP-MID"]
ALGORITHMS = ALGORITHMS_OUR + ALGORITHMS_SOTA

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

LATEX_TABLE_HEADER = """
\\begin{{table}}[H] \\tiny \label{{tab:tabela1}} \centering
\caption{{Porównanie zaimplementowanych algorytmów w oparciu o CEC2022 dla {dim} wymiarów.}}
\\begin{{tabular}} {{| c | r | r | r | r | r | r | c | c | c |}} \hline

"""

LATEX_TABLE_COLUMN_NAMES = """
     & \multicolumn{{2}}{{c|}}{{{first_alg}}} & \multicolumn{{2}}{{c|}}{{{second_alg}}} & \multicolumn{{2}}{{c|}}{{{third_alg}}} & {comp_1} & {comp_2} & {comp_3} \\\\ \hline
     Funkcja & średnia & odchylenie & średnia & odchylenie & średnia & odchylenie & wst & wst & wst \\\\ \hline
     """

LATEX_END_TABLE = """
& \multicolumn{{6}}{{c|}}{{+/-/=}} & \multicolumn{{1}}{{c|}}{{{comp_1_sum}}} & \multicolumn{{1}}{{c|}}{{{comp_2_sum}}} & \multicolumn{{1}}{{c|}}{{{comp_3_sum}}}  \\\\ \hline
\end{{tabular}}
\end{{table}}
"""

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

test_results = os.listdir(f'../tests/final_test/mvmo')
done = []

for result in test_results:
    result = result.split("__")[1].split("_")
    func = result[0]
    dim = result[3]
    run = result[4].split(".")[0]
    done.append((dim, func, run))
RESULT_DF = pd.DataFrame(done)
RESULT_DF.columns = ["dim", "func", "run"]
RESULT_DF.run = RESULT_DF.run.astype(int)
RESULT_DF.func = RESULT_DF.func.astype(int)


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
    result_df = pd.read_csv(f"../tests/{PATH_DICT[algorithm]}/{algorithm.lower()}_{row['func']}_{row['dim']}.txt", delim_whitespace=True, header=None)
    result = result_df[int(row['run']) - 1].iloc[15]
    return 0 if result <= 10 ** -8 else result


def stats(algorithm, dim):
    result_df = RESULTS_DICT[algorithm][RESULTS_DICT[algorithm].dim == dim]
    stats_df = result_df[["func", "result"]].groupby(["func"]).mean()
    stats_df = stats_df.merge(result_df[["func", "result"]].groupby(["func"]).std(), left_index=True, right_index=True)

    stats_df.columns = ["mean", "std"]
    stats_df = stats_df.reset_index()
    stats_df = stats_df.sort_values(["func"])
    return stats_df


for algorithm in ALGORITHMS:
    get_result = partial(get_result_temp, algorithm)
    RESULTS_DICT[algorithm]["result"] = RESULTS_DICT[algorithm].apply(get_result, axis=1)


def wst(first_alg, second_alg, dim, row):
    first_alg_results = RESULTS_DICT[first_alg][(RESULTS_DICT[first_alg].dim == dim) & (RESULTS_DICT[first_alg].func == int(row['func']))].sort_values('run').result.tolist()
    second_alg_results = RESULTS_DICT[second_alg][(RESULTS_DICT[second_alg].dim == dim) & (RESULTS_DICT[second_alg].func == int(row['func']))].sort_values('run').result.tolist()

    if first_alg_results == second_alg_results:
        pvalue = 1
    else:
        pvalue = wilcoxon(first_alg_results, second_alg_results).pvalue

    if pvalue >= 0.05:
        return '='
    elif row[f"{second_alg} mean"] < row[f"{first_alg} mean"]:
        return '+'
    elif row[f"{second_alg} mean"] > row[f"{first_alg} mean"]:
        return '-'


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
    comparison_df = stats(first, dim)
    comparison_df.columns = ['func'] + [first + ' ' + col for col in comparison_df.columns[1:]]
    for algorithm in all:
        to_compare_df = stats(algorithm, dim)
        to_compare_df.columns = ['func'] + [algorithm + ' ' + col for col in to_compare_df.columns[1:]]
        comparison_df = comparison_df.merge(to_compare_df, left_on="func", right_on="func")

    comp_1 = 'TLBO vs MVMO'
    comp_2 = 'HS vs MVMO'
    comp_3 = 'TLBO vs HS'
    comparison_df[comp_1] = comparison_df.apply(partial(wst, "MVMO", "TLBO", dim), axis=1)
    comparison_df[comp_2] = comparison_df.apply(partial(wst, "MVMO", "HS", dim), axis=1)
    comparison_df[comp_3] = comparison_df.apply(partial(wst, "HS", "TLBO", dim), axis=1)
    print(LATEX_TABLE_HEADER.format(algorithm=ALGORITHMS_LABELS[first], dim=dim))

    print(LATEX_TABLE_COLUMN_NAMES.format(
        first_alg=ALGORITHMS_LABELS[first],
        second_alg=ALGORITHMS_LABELS[all[0]],
        third_alg=ALGORITHMS_LABELS[all[1]],
        comp_1=comp_1,
        comp_2=comp_2,
        comp_3=comp_3,
    ).replace('_', '\\_')
          )
    cols_to_cast = [col for col in comparison_df.columns if col.endswith("std") or col.endswith("mean")]
    comparison_df[cols_to_cast] = comparison_df[cols_to_cast].applymap(lambda x: '%.2E' % Decimal(x))

    print(comparison_df.apply(lambda row: " & ".join([str(x) for x in row]) + "\\\\ \hline", axis=1).to_string(index=False).replace('.', ','))

    comp_1_sum = comparison_df[comp_1].value_counts()
    comp_2_sum = comparison_df[comp_2].value_counts()
    comp_3_sum = comparison_df[comp_3].value_counts()

    print(LATEX_END_TABLE.format(
        comp_1_sum=f"{get_wst(comp_1_sum, '+')}/{get_wst(comp_1_sum, '-')}/{get_wst(comp_1_sum, '=')}",
        comp_2_sum=f"{get_wst(comp_2_sum, '+')}/{get_wst(comp_2_sum, '-')}/{get_wst(comp_2_sum, '=')}",
        comp_3_sum=f"{get_wst(comp_3_sum, '+')}/{get_wst(comp_3_sum, '-')}/{get_wst(comp_3_sum, '=')}",
    ))


alg = "MVMO"
for dim in ["10", "20"]:
    gen_table(alg, ALGORITHMS_OUR, dim)
