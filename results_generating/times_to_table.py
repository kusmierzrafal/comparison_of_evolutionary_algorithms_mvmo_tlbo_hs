algorytm = "TLBO"
log_file = f"../tests/algorithms_comparison/{algorytm.lower()}_comparison.log"


def get_value(ind):
    val = float(data[ind].replace("\t", "").replace("\n", "").split(" -> ")[-1])
    return ('%.2E' % val).replace('.', ',')


f = open(log_file, "r")
data = f.readlines()


title = "Funkcja Levy'ego"
offset = 0
table = f"""
\\begin{{table}}[H]
\\centering
\\caption{{Porównanie czasów optymalizacji dla algorytmu {algorytm}}}
    \\begin{{tabular}}{{|c|c|r|r|r|r|}}
    \\hline
    \\multicolumn{{2}}{{|c|}}{{}} & \\multicolumn{{4}}{{c|}}{{{title}}}\\\\ \\cline{{3-6}}
    \\multicolumn{{2}}{{|c|}}{{}} & \\multicolumn{{2}}{{c|}}{{Przygotowana}}& \\multicolumn{{2}}{{c|}}{{Istniejąca}}\\\\ \\hline
    Populacja & Wymiary & \\multicolumn{{1}}{{c|}}{{Średni}}& \\multicolumn{{1}}{{c|}}{{Najlepszy}}& \\multicolumn{{1}}{{c|}}{{Średni}}& \\multicolumn{{1}}{{c|}}{{Najlepszy}}\\\\ \\hline
    10 & 10 &  {get_value(offset + 14)} &  {get_value(offset + 15)} &  {get_value(offset + 19)} &  {get_value(offset + 20)} \\\\ \\hline
    40 & 10 &  {get_value(offset + 38)} &  {get_value(offset + 39)} &  {get_value(offset + 43)} &  {get_value(offset + 44)} \\\\ \\hline
    70 & 10 &  {get_value(offset + 62)} &  {get_value(offset + 63)} &  {get_value(offset + 67)} &  {get_value(offset + 68)} \\\\ \\hline
    10 & 20 &  {get_value(offset + 86)} &  {get_value(offset + 87)} &  {get_value(offset + 91)} &  {get_value(offset + 92)} \\\\ \\hline
    40 & 20 &  {get_value(offset + 110)} &  {get_value(offset + 111)} &  {get_value(offset + 115)} &  {get_value(offset + 116)} \\\\ \\hline
    70 & 20 &  {get_value(offset + 134)} &  {get_value(offset + 135)} &  {get_value(offset + 139)} &  {get_value(offset + 140)} \\\\ \\hline
    \\end{{tabular}}
 \\end{{table}}
"""
print(table)

title = "Funkcja Rosenbrock'a"
offset = 144
table = f"""
\\begin{{table}}[H]
\\centering
    \\caption{{Porównanie czasów optymalizacji dla algorytmu {algorytm}}}

    \\begin{{tabular}}{{|c|c|r|r|r|r|}}
    \\hline
    \\multicolumn{{2}}{{|c|}}{{}} & \\multicolumn{{4}}{{c|}}{{{title}}}\\\\ \\cline{{3-6}}
    \\multicolumn{{2}}{{|c|}}{{}} & \\multicolumn{{2}}{{c|}}{{Przygotowana}}& \\multicolumn{{2}}{{c|}}{{Istniejąca}}\\\\ \\hline
    Populacja & Wymiary & \\multicolumn{{1}}{{c|}}{{Średni}}& \\multicolumn{{1}}{{c|}}{{Najlepszy}}& \\multicolumn{{1}}{{c|}}{{Średni}}& \\multicolumn{{1}}{{c|}}{{Najlepszy}}\\\\ \\hline
    10 & 10 &  {get_value(offset + 14)}  &  {get_value(offset + 15)}  &  {get_value(offset + 19)}  &  {get_value(offset + 20)} \\\\ \\hline
    40 & 10 &  {get_value(offset + 38)}  &  {get_value(offset + 39)}  &  {get_value(offset + 43)}  &  {get_value(offset + 44)} \\\\ \\hline
    70 & 10 &  {get_value(offset + 62)}  &  {get_value(offset + 63)}  &  {get_value(offset + 67)}  &  {get_value(offset + 68)} \\\\ \\hline
    10 & 20 &  {get_value(offset + 86)}  &  {get_value(offset + 87)}  &  {get_value(offset + 91)}  &  {get_value(offset + 92)} \\\\ \\hline
    40 & 20 &  {get_value(offset + 110)}  &  {get_value(offset + 111)}  &  {get_value(offset + 115)}  &  {get_value(offset + 116)} \\\\ \\hline
    70 & 20 &  {get_value(offset + 134)}  &  {get_value(offset + 135)}  &  {get_value(offset + 139)}  &  {get_value(offset + 140)} \\\\ \\hline
    \\end{{tabular}}
 \\end{{table}}
"""
print(table)


title = "Funkcja Zakharov'a"
offset = 288
table = f"""
\\begin{{table}}[H]
\\centering
    \\caption{{Porównanie czasów optymalizacji dla algorytmu {algorytm}}}

    \\begin{{tabular}}{{|c|c|r|r|r|r|}}
    \\hline
    \\multicolumn{{2}}{{|c|}}{{}} & \\multicolumn{{4}}{{c|}}{{{title}}}\\\\ \\cline{{3-6}}
    \\multicolumn{{2}}{{|c|}}{{}} & \\multicolumn{{2}}{{c|}}{{Przygotowana}}& \\multicolumn{{2}}{{c|}}{{Istniejąca}}\\\\ \\hline
    Populacja & Wymiary & \\multicolumn{{1}}{{c|}}{{Średni}}& \\multicolumn{{1}}{{c|}}{{Najlepszy}}& \\multicolumn{{1}}{{c|}}{{Średni}}& \\multicolumn{{1}}{{c|}}{{Najlepszy}}\\\\ \\hline
    10 & 10 &  {get_value(offset + 14)}  &  {get_value(offset + 15)}  &  {get_value(offset + 19)}  &  {get_value(offset + 20)} \\\\ \\hline
    40 & 10 &  {get_value(offset + 38)}  &  {get_value(offset + 39)}  &  {get_value(offset + 43)}  &  {get_value(offset + 44)} \\\\ \\hline
    70 & 10 &  {get_value(offset + 62)}  &  {get_value(offset + 63)}  &  {get_value(offset + 67)}  &  {get_value(offset + 68)} \\\\ \\hline
    10 & 20 &  {get_value(offset + 86)}  &  {get_value(offset + 87)}  &  {get_value(offset + 91)}  &  {get_value(offset + 92)} \\\\ \\hline
    40 & 20 &  {get_value(offset + 110)}  &  {get_value(offset + 111)}  &  {get_value(offset + 115)}  &  {get_value(offset + 116)} \\\\ \\hline
    70 & 20 &  {get_value(offset + 134)}  &  {get_value(offset + 135)}  &  {get_value(offset + 139)}  &  {get_value(offset + 140)} \\\\ \\hline
    \\end{{tabular}}
 \\end{{table}}
"""
print(table)
