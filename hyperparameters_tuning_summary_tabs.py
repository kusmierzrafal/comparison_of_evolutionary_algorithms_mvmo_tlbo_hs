import os
import pandas as pd

files_to_check = os.listdir('./tests/hyperparameters_tuning/')
mvmo_done = []
tlbo_done = []

for file in files_to_check:
    if file.startswith('mvmo'):
        mvmo_done.append(file)
    elif file.startswith('tlbo'):
        tlbo_done.append(file)


mvmo_done = [(int(mvmo.split('_')[2]), int(mvmo.split('_')[-1].split('.')[0])) for mvmo in mvmo_done]
tlbo_done = [(int(tlbo.split('_')[3]), int(tlbo.split('_')[-1].split('.')[0])) for tlbo in tlbo_done]

mvmo_all = []
for best_size in [2, 10, 20, 30, 40, 50, 60, 70, 80]:
    for run in range(1, 31, 1):
        mvmo_all.append((best_size, run))

tlbo_all = []
for pop_size in range(10, 80, 10):
    for run in range(1, 31, 1):
        tlbo_all.append((pop_size, run))


def get_result_tlbo(row):
    pop_size = row["pop_size"]
    run = row["run"]
    f = open(f'./tests/hyperparameters_tuning/tlbo__6_{pop_size}_{int(200000 / 2 // pop_size)}_20_{run}.txt')
    content = f.read()
    result = content.rstrip().split('\n')[-1]
    return round(float(result), 2)

def get_result_mvmo(row):
    n_best_size = row["n_best_size"]
    run = row["run"]
    f = open(f'./tests/hyperparameters_tuning/mvmo_1_{n_best_size}_1_1_75__6_50_200000_20_{run}.txt')
    content = f.read()
    result = content.rstrip().split('\n')[-1]
    return round(float(result), 2)


tlbo_result_df = pd.DataFrame(tlbo_all)
tlbo_result_df.columns = ["pop_size", "run"]
tlbo_result_df["result"] = tlbo_result_df.apply(get_result_tlbo, axis=1)


tlbo_stats_df = tlbo_result_df[["pop_size", "result"]].groupby("pop_size").min()
tlbo_stats_df = tlbo_stats_df.merge(tlbo_result_df[["pop_size", "result"]].groupby("pop_size").max(), left_index=True, right_index=True)
tlbo_stats_df = tlbo_stats_df.merge(tlbo_result_df[["pop_size", "result"]].groupby("pop_size").mean().round(2), left_index=True, right_index=True)
tlbo_stats_df = tlbo_stats_df.merge(tlbo_result_df[["pop_size", "result"]].groupby("pop_size").std().round(2), left_index=True, right_index=True)
tlbo_stats_df.columns = ["min_result", "max_result", "mean_result", "std_result"]
tlbo_stats_df = tlbo_stats_df.reset_index()


tlbo_latex_table_start = f"""
\\begin{{table}}[H] \label{{tab:tabela1}} \centering
\caption{{Dobór parametrów TLBO}}
\\begin{{tabular}} {{| c | c | c | c | c |}} \hline
    rozmiar populacji & najlepszy wynik & najsłabszy wynik & średni wynik & odchylenie \\\\ \hline
    
"""

tlbo_latex_body_table = """
{pop_size} & {min_result} & {max_result} & {mean_result} & {std_result} \\\\ \hline
"""

latex_table_end = f"""

\end{{tabular}}
\end{{table}}
"""

tlbo_latex_table = ""
tlbo_latex_table += tlbo_latex_table_start
for row in tlbo_stats_df.iterrows():
    tlbo_latex_table += tlbo_latex_body_table.format(pop_size=int(row[1]["pop_size"]), min_result=row[1]["min_result"], max_result=row[1]["max_result"], mean_result=row[1]["mean_result"], std_result=row[1]["std_result"])
tlbo_latex_table += latex_table_end

print(tlbo_latex_table)


mvmo_result_df = pd.DataFrame(mvmo_all)
mvmo_result_df.columns = ["n_best_size", "run"]
mvmo_result_df["result"] = mvmo_result_df.apply(get_result_mvmo, axis=1)


mvmo_stats_df = mvmo_result_df[["n_best_size", "result"]].groupby("n_best_size").min()
mvmo_stats_df = mvmo_stats_df.merge(mvmo_result_df[["n_best_size", "result"]].groupby("n_best_size").max(), left_index=True, right_index=True)
mvmo_stats_df = mvmo_stats_df.merge(mvmo_result_df[["n_best_size", "result"]].groupby("n_best_size").mean().round(2), left_index=True, right_index=True)
mvmo_stats_df = mvmo_stats_df.merge(mvmo_result_df[["n_best_size", "result"]].groupby("n_best_size").std().round(2), left_index=True, right_index=True)
mvmo_stats_df.columns = ["min_result", "max_result", "mean_result", "std_result"]
mvmo_stats_df = mvmo_stats_df.reset_index()

mvmo_latex_table_start = f"""
\\begin{{table}}[H] \label{{tab:tabela1}} \centering
\caption{{Dobór parametrów MVMO}}
\\begin{{tabular}} {{| c | c | c | c | c |}} \hline
    rozmiar archiwum & najlepszy wynik & najsłabszy wynik & średni wynik & odchylenie \\\\ \hline

"""

mvmo_latex_body_table = """
{n_best_size} & {min_result} & {max_result} & {mean_result} & {std_result} \\\\ \hline
"""

mvmo_latex_table = ""
mvmo_latex_table += mvmo_latex_table_start
for row in mvmo_stats_df.iterrows():
    mvmo_latex_table += mvmo_latex_body_table.format(n_best_size=int(row[1]["n_best_size"]), min_result=row[1]["min_result"],
                                                     max_result=row[1]["max_result"], mean_result=row[1]["mean_result"],
                                                     std_result=row[1]["std_result"])
mvmo_latex_table += latex_table_end

print(mvmo_latex_table)
