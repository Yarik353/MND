import math
import numpy as np
from numpy import average, transpose
from numpy.linalg import solve
from prettytable import PrettyTable
from scipy.stats import f
from scipy.stats import t as t_criterium
from functools import partial
from random import randint

m, N, d = 3, 8, 8

x0_factor = [1, 1, 1, 1, 1, 1, 1, 1]
x1_factor = [-1, -1, 1, 1, -1, -1, 1, 1]
x2_factor = [-1, 1, -1, 1, -1, 1, -1, 1]
x3_factor = [-1, 1, 1, -1, 1, -1, -1, 1]
x1x2_factor = [a * b for a, b in zip(x1_factor, x2_factor)]
x1x3_factor = [a * b for a, b in zip(x1_factor, x3_factor)]
x2x3_factor = [a * b for a, b in zip(x2_factor, x3_factor)]
x1x2x3_factor = [a * b * c for a, b, c in zip(x1_factor, x2_factor, x3_factor)]

x1_list = []
x2_list = []
x3_list = []
x1x2_list = []
x1x3_list = []
x2x3_list = []
x1x2x3_list = []
x_main_list = [x0_factor, x1_list, x2_list, x3_list, x1x2_list, x1x3_list, x2x3_list, x1x2x3_list]
x_factor_list = [x0_factor, x1_factor, x2_factor, x3_factor, x1x2_factor, x1x3_factor, x2x3_factor, x1x2x3_factor]

list_bi = []

F1 = m - 1
F2 = N
F3 = F1 * F2
F4 = N - d

x1, x2, x3 = (-20, 30), (5, 40), (5, 10)
x_tuple = (x1, x2, x3)
x_max_average = average([i[1] for i in x_tuple])
x_min_average = average([i[0] for i in x_tuple])
y_max = int(200 + x_max_average)
y_min = int(200 + x_min_average)
y_min_max = [y_min, y_max]
mat_Y = [[randint(y_min_max[0], y_min_max[1]) for _ in range(m)] for _ in range(N)]

def get_average_y():
    return [round(sum(mat_Y[k1]) / m, 3) for k1 in range(N)]

def get_dispersion():
    return [round(sum([((k1 - get_average_y()[j]) ** 2) for k1 in mat_Y[j]]) / m, 3) for j in
            range(N)]

def fill_x_matrix():
    [x1_list.append(x1[0 if i == -1 else 1]) for i in x1_factor]
    [x2_list.append(x2[0 if i == -1 else 1]) for i in x2_factor]
    [x3_list.append(x3[0 if i == -1 else 1]) for i in x3_factor]
    [x1x2_list.append(a * b) for a, b in zip(x1_list, x2_list)]
    [x1x3_list.append(a * b) for a, b in zip(x1_list, x3_list)]
    [x2x3_list.append(a * b) for a, b in zip(x2_list, x3_list)]
    [x1x2x3_list.append(a * b * c) for a, b, c in zip(x1_list, x2_list, x3_list)]

def cohren():
    q = 0.05
    Gp = max(get_dispersion()) / sum(get_dispersion())
    q1 = q / F1
    fisher_value = f.ppf(q=1 - q1, dfn=F2, dfd=(F1 - 1) * F2)
    Gt = fisher_value / (fisher_value + F1 - 1)
    return Gp < Gt

def fisher():
    fisher_teor = partial(f.ppf, q=1 - 0.05)
    Ft = fisher_teor(dfn=F4, dfd=F3)
    return Ft


fill_x_matrix()
dispersion = get_dispersion()
sum_dispersion = sum(dispersion)
y_average = get_average_y()

column_names1 = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "Y1", "Y2", "Y3", "Y", "S^2"]
trans_y_mat = transpose(mat_Y).tolist()

list_for_solve_a = list(zip(*x_main_list))
list_for_solve_b = x_factor_list

for k in range(N):
    S = 0
    for i in range(N):
        S += (list_for_solve_b[k][i] * y_average[i]) / N
    list_bi.append(round(S, 5))

pt = PrettyTable()
cols = x_factor_list
[cols.extend(ls) for ls in [trans_y_mat, [y_average], [dispersion]]]
[pt.add_column(column_names1[coll_id], cols[coll_id]) for coll_id in range(13)]
print("Матриця планування")
print(pt, "\n")


pt = PrettyTable()
cols = x_main_list
[cols.extend(ls) for ls in [trans_y_mat, [y_average], [dispersion]]]
[pt.add_column(column_names1[coll_id], cols[coll_id]) for coll_id in range(13)]
print("Нормована матриця")
print(pt, "\n")
list_ai = [round(i, 5) for i in solve(list_for_solve_a, y_average)]
print("Критерій Кохрена")
if cohren():
    print("Дисперсія однорідна!\n")
    Dispersion_B = sum_dispersion / N
    Dispersion_beta = Dispersion_B / (m * N)
    S_beta = math.sqrt(abs(Dispersion_beta))
    beta_list = np.zeros(8).tolist()
    for i in range(N):
        beta_list[0] += (y_average[i] * x0_factor[i]) / N
        beta_list[1] += (y_average[i] * x1_factor[i]) / N
        beta_list[2] += (y_average[i] * x2_factor[i]) / N
        beta_list[3] += (y_average[i] * x3_factor[i]) / N
        beta_list[4] += (y_average[i] * x1x2_factor[i]) / N
        beta_list[5] += (y_average[i] * x1x3_factor[i]) / N
        beta_list[6] += (y_average[i] * x2x3_factor[i]) / N
        beta_list[7] += (y_average[i] * x1x2x3_factor[i]) / N
    t_list = [abs(beta_list[i]) / S_beta for i in range(0, N)]
    print("Критерій Стьюдента")
    for i, j in enumerate(t_list):
        print(f't{i}={beta_list[i]}')
        if j < t_criterium.ppf(q=0.975, df=F3):
            beta_list[i] = 0
            d -= 1
    print()
    print('Рівняння регресії з коефіцієнтами від нормованих значень факторів')
    print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3".format(*list_bi))
    print('Рівняння регресії з коефіцієнтами від натуральних значень факторів')
    print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3".format(*list_ai))
    print()
    Y_counted = [sum([beta_list[0], *[beta_list[i] * x_main_list[1:][j][i] for i in range(N)]])
                 for j in range(N)]
    Dispersion_ad = 0
    for i in range(len(Y_counted)):
        Dispersion_ad += ((Y_counted[i] - y_average[i]) ** 2) * m / (N - d)
    Fp = Dispersion_ad / Dispersion_beta
    Ft = fisher()
    print("Критерій Фіішера")
    
    if (Ft > Fp) and list(beta_list).count(0) < len(list(beta_list))-list(beta_list).count(0) : # Додаткова перевірка, якщо кількість незначимих менша за кількість значимих - то модель вважається неадекватною
        print("Рівняння регресії адекватне!")
    else:
        print("Рівняння регресії неадекватне.")
else:
    print("Дисперсія неоднорідна")