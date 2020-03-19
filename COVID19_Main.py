import math
from tkinter import font
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from matplotlib.pyplot import *

from matplotlib import pyplot as plt
import math
import numpy as np
from numpy import *
from matplotlib.pyplot import *

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

from COVID19_functions import nd_n, n_d, plotting,func_exp,dataestimation,exponential_regression,x_data,y_data
# ----------------------------------------------------
# This is a simple model of the COVID-19 affectation in Mexico
# its purpose is to determine the quantity of people who will
# suffer this decease in and quantify the importance of the
# self-quarantine and its impact on the number of infected persons
#
# Nd = cases in a given day
# E = Average of people and infected is exposed each day
# p = probability of a healthy person to get infected
# C = contant of proporcionality
# d = number of days the decease
#
#--------------------------------------------------
#
#Last Update March 18 2020 
Ep1 = 0.297;
Ep2 = 0.307;
Ep3 = 0.287;
N_0 = 0.77;
d = 30;

days=str(d);
suptitle('Expectations for COVID-19 MXN For ' + days +' days')
plt.subplot(2, 2, 1)
plotting(N_0, Ep1, 0)
plt.subplot(2, 2, 2)
plotting(N_0, Ep1, d)
plt.subplot(2, 2, 3)
plotting(N_0, Ep2, d)
plt.subplot(2, 2, 4)
plotting(N_0, Ep3, d)

show()
suptitle('Expectations for COVID-19 MXN For ' + days +' days')
plt.subplot(2, 1, 1)
coeficients=exponential_regression(x_data, y_data)
print(coeficients[0])
plt.subplot(2, 1, 2)
dataestimation(d)
show()

