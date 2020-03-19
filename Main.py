import math
import numpy as np
from numpy import *
from matplotlib.pyplot import *

from matplotlib import pyplot as plt
# ----------------------------------------------------
# This is a simple model of the COVID-19 affectation in Mexico
# its purpose is to determine the quantity of people who will
# suffer this decease in and quantify the importance of the
# self-quarantine and its impact on the number of infected persons
#
# Nd = cases in a given day
# E = Average of people and infected is exposed each day
# P = probability of a healthy person to get infected
# C = contant of proporcionality
# d = number of days the decease
#
# --------------------------------------------------


E = 1;
N = 93;
P = 0.2;
d = 18;
d_array = [];
Nd = N * (1 + E * P) ** (d);

for i in range(1, 16, 1):
    d_array.append(i)



print(d_array)

n_cases = [0,1,2,4,5,6,8,11,15,26,41,53,82,91,93]



#print(n_cases)
plt.plot(d_array, [0,1,2,4,5,6,8,11,15,26,41,53,82,91,93])
plt.show()
#plt.plot[d_array, n_cases]
#plt.show()

#t = linspace(0, 3, 51)


