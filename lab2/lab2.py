from random import randint
from math import sqrt
from numpy.linalg import det
import prettytable

romanovsky_table = {(2, 3, 4): 1.71, (5, 6, 7): 2.1, (8, 9): 2.27, (10, 11): 2.41,
                    (12, 13): 2.52, (14, 15, 16, 17): 2.64, (18, 19, 20): 2.78}



#---------Дані за варіантом-------
n = 119
x1_min = 15
x1_max = 45
x2_min = -35
x2_max = 15
y_max = (30 - n) * 10
y_min = (20 - n) * 10

#---------Шукані величини-------
offset = 0
romanovsky = 0
averages_y = list()
dispersion_y = list()
f_uv = list()
sigma_uv = list()
r_uv = list()

x1 = [-1, -1, 1]
x2 = [-1, 1, -1]

nx1 = [x1_min if x1[i] == -1 else x1_max for i in range(3)]
nx2 = [x2_min if x2[i] == -1 else x2_max for i in range(3)]

m = 5
y_1 = [randint(y_min, y_max) for i in range(m)]
y_2 = [randint(y_min, y_max) for i in range(m)]
y_3 = [randint(y_min, y_max) for i in range(m)]

def get_dispersion(average, y_list):
    dispersion = 0
    for i in range(len(y_list)):
        dispersion += (y_list[i]-average)**2
    return dispersion/m


def romanovsky_criterion():
    global offset, romanovsky, averages_y, dispersion_y, f_uv, sigma_uv, r_uv, m
    averages_y = [sum(y_1)/m, sum(y_2)/m, sum(y_3)/m]
    dispersion_y = [get_dispersion(averages_y[0], y_1), get_dispersion(averages_y[1], y_2), get_dispersion(averages_y[2], y_3)]
    offset = sqrt((2*(2*m-2))/(m*(m-4)))
    uv_pairs = [[dispersion_y[0], dispersion_y[1]], [dispersion_y[1], dispersion_y[2]], [dispersion_y[2], dispersion_y[0]]]
    for i in range(3):
        f_uv.append(max(uv_pairs[i]) / min(uv_pairs[i]))
    sigma_uv = [((m-2)/m)*f_uv[0], ((m-2)/m)*f_uv[1], ((m-2)/m)*f_uv[2]]
    r_uv = [abs(sigma_uv[0]-1)/offset, abs(sigma_uv[1]-1)/offset, abs(sigma_uv[2]-1)/offset]
    for key in romanovsky_table.keys():
        if m in key:
            romanovsky = romanovsky_table[key]
            break


    if(max(r_uv)>=romanovsky):
        m+=1
        y_1.append(randint(y_min, y_max))
        y_2.append(randint(y_min, y_max))
        y_3.append(randint(y_min, y_max))
        romanovsky_criterion()

romanovsky_criterion()
#-------------Розрахунок нормованих коефіцієнтів рівняння регресії

mx1, mx2, my = sum(x1) / 3, sum(x2) / 3, sum(averages_y) / 3
a1, a2, a3 = (x1[0]**2 + x1[1]**2+ x1[2]**2)/3,  (x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2])/3, (x2[0]**2 + x2[1]**2+ x2[2]**2)/3
a11, a22 = (x1[0]*averages_y[0] + x1[1]*averages_y[1] + x1[2]*averages_y[2])/3,  (x2[0]*averages_y[0] + x2[1]*averages_y[1] + x2[2]*averages_y[2])/3

deter = det([[1, mx1, mx2], [mx1, a1, a2], [mx2, a2, a3]])
b0 = det([[my, mx1, mx2], [a11, a1, a2], [a22, a2, a3]]) / deter
b1 = det([[1, my, mx2], [mx1, a11, a2], [mx2, a22, a3]]) / deter
b2 = det([[1, mx1, my], [mx1, a1, a11], [mx2, a2, a22]]) / deter

delta_x1, delta_x2, x10, x20 = abs(x1_max - x1_min) / 2, abs(x2_max - x2_min) / 2, (x1_max + x1_min) / 2, (x2_max + x2_min) / 2
a0 =((b0 - b1 * x10 / delta_x1) - (b2 * x20 / delta_x2))
a1 = (b1 / delta_x1)
a2 = (b2 / delta_x2)

table = prettytable.PrettyTable()
table.field_names = ["№", "X1", "X2", *[f"Y{i+1}" for i in range(m)]]

table.add_row([1, x1[0], x2[0], *y_1])
table.add_row([2, x1[1], x2[1], *y_2])
table.add_row([3, x1[2], x2[2], *y_3])

print("Критерій Романовського: " + str(romanovsky))
print("Головне відхилення: " + str(offset))
print(table)

table2 = prettytable.PrettyTable()
table2.field_names = ["№", "Average Y", "Dispersion Y", "Fuv", "σuv", "Ruv"]
for i in range(3):
    table2.add_row([i+1, round(averages_y[i], 3), round(dispersion_y[i], 3), round(f_uv[i], 3), round(sigma_uv[i], 3), round(r_uv[i], 3)])

print(table2)

print(f"Нормоване рівняння регресії: {round(b0, 3)} + {round(b1, 3)} * x1 + {round(b2, 3)} * x2")

table3 = prettytable.PrettyTable()
table3.field_names = ["№", "X1", "X2", "Experimental Y", "Average Y"]
for i in range(3):
    table3.add_row([i+1, x1[i], x2[i], round(averages_y[i], 3), round(b0+b1*x1[i]+b2*x2[i], 3)])
print(table3)

print(f"Натуралізоване рівняння регресії: {round(a0, 3)} + {round(a1, 3)} * x1 + {round(a2, 3)} * x2")

table4 = prettytable.PrettyTable()
table4.field_names = ["№", "NX1", "NX2", "Experimental Y", "Average Y"]
for i in range(3):
    table4.add_row([i+1, nx1[i], nx2[i], round(averages_y[i], 3), round(a0+a1*nx1[i]+a2*nx2[i], 3)])
print(table4)