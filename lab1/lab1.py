import numpy as np
from random import randint
import prettytable

MIN, MAX = 0, 20
a0, a1, a2, a3 = 1, 2, 2, 3
#################################
X = np.empty((8, 3), dtype=float)
Y = np.empty(8)
X0 = np.empty(3)
DX = np.empty(3)
XNormalized = np.empty((8, 3), dtype=float)
############################################
for i in range(8):
    for j in range(3):
        X[i, j] = randint(MIN, MAX)

for i in range(8):
    Y[i] = a0 + a1 * X[i, 0] + a2 * X[i, 1] + a3 * X[i, 2]

for i in range(3):
    X0[i] = (X[:, i].max() + X[:, i].min()) / 2
    DX[i] = X[:, i].max() - X0[i]

Y_et = a0 + a1 * X0[0] + a2 * X0[1] + a3 * X0[2]

for i in range(8):
    for j in range(3):
        XNormalized[i, j] = ((X[i, j] - X0[j]) / DX[j]).round(3)

bigger_lst = [i for i in Y if i >= Y_et]
number = np.where(Y==(min(bigger_lst)))[0][0]

Y2 = a0 + a1 * X[number, 0] + a2 * X[number, 1] + a3 * X[number, 2]

table = prettytable.PrettyTable()
table.field_names = ["№", "X1", "X2", "X3", "Y", "XН1", "XН2", "XH3"]

for i in range(8):
    table.add_row([i+1, X[i][0], X[i][1], X[i][2], Y[i], XNormalized[i][0], XNormalized[i][1], XNormalized[i][2]])

table.add_row(["X0", X0[0], X0[1], X0[2], "-", "-", "-", "-"])
table.add_row(["Dx", DX[0], DX[1], DX[2], "-", "-", "-", "-"])

print(table)
print("Yet = ", Y_et)
print("Вираз який задовольняє критерій вибору 'Yet <-':", )
print( "{} + {} * {} + {} * {} + {} * {} = {}".format(a0,a1,X[number][0], a2, X[number][1], a3, X[number][2], Y2))
