import math
from tkinter import font
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
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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
#Data regresion method obtain from https://stackoverflow.com/questions/50706092/exponential-regression-function-python
# --------------------------------------------------
font = {'family': 'serif',
        'color': 'red',
        'weight': 'normal',
        'size': 10,
        }
font2 = {'family': 'serif',
         'color': 'blue',
         'weight': 'normal',
         'size': 10,
         }
font3 = {'family': 'serif',
         'color': 'black',
         'weight': 'normal',
         'size': 10,
         }
font4 = {'family': 'serif',
         'color': 'black',
         'weight': 'bold',
         'size': 8.5,
         }
font5 = {'family': 'serif',
         'color': 'black',
         'weight': 'bold',
         'size': 10,
         }
# ========================Real data=============================
quantity_of_infected = [0, 1, 2, 4, 5, 6, 7, 11, 15, 26, 41, 53, 82, 91, 93, 108]  # number of infected
days_elapse = [0, 1, 2, 3, 4, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # days elapsed from day 0

y_data=np.array([0,1,2,4,5,6,7,11,15,26,41,53,82,91,93,108]) #number of infected
x_data=np.array([0,1,2,3,4,9,10,14,15,16,17,18,19,20,21,22])#days elapsed from day 0
# =======================Function definition===========================#
# =================Model Functions========
def nd_n(N_0, Ep, d):
    n_given_day = ((1 + Ep) ** d) * N_0
    # print(n_given_day)
    # print('------------------')
    ndn = ((1 + Ep) * n_given_day)
    print(ndn)


def n_d(N_0, Ep, d):
    nd = N_0 * (1 + Ep) ** d;
    nd = round(nd)
    return nd;


def plotting(N_0, Ep, d):
    if d == 0:
        print('The Maximum Number of Cases is : 0')
        print('In ', d, ' days since day 0')

        # ===========================================================================
        # ================================PLOT PART==================================
        # =======Arranging data=====#

        plt.title('Predictive Exponential Model COVID-19 MXN')
        plt.ylabel('Numer of cases (N)')
        plt.xlabel('Time (Days)')
        plt.grid(True)

        # ====================Texts===================
        plt.text(0.1, 0.8, "Red is the math model",
                 fontdict=font)
        plt.text(0.5, 0.8, r'$Nd_1 = (1 +E*p)*Nd$',
                 fontdict=font3)
        plt.text(0.40, 0.73, "Where ",
                 fontdict=font3, )
        plt.text(0.52, 0.73, r'$Nd = ((1+E*p)^d)*N_0$',
                 fontdict=font3)
        plt.text(0.1, 0.65, "Blue squares are real data",
                 fontdict=font2)
        # plt.plot(x_Axis, y_Axis, 'r', r, t, 'bs')  # PLOTTING DATA
        plt.text(0.1, 0.55, "Ep: Average of people and infected is exposed each day times probability of get infected",
                 fontdict=font4)
        plt.text(0.1, 0.45, "Nd = Number of cases on a given day",
                 fontdict=font4)
        plt.text(0.1, 0.35, "d: number of days the decease",
                 fontdict=font4)
        plt.text(0.1, 0.25, r'$N_0$',
                 fontdict=font4)
        plt.text(0.15, 0.25, ": Number of cases on day 0",
                 fontdict=font4)
    elif d > 0:
        y_Axis = []
        x_Axis = []
        for i in range(0, d, 1):
            y_Axis.append(n_d(N_0, Ep, (i + 1)))
            # print(i)
        # print('fin clico')

        #print('The Maximum Number of Cases is :', y_Axis[d])
        for i in range(0, d, 1):
            x_Axis.append(i)
        print('In ', d, ' days since day 0')
        # print(y_Axis)
        # print(x_Axis)

        # ===========================================================================
        # ================================PLOT PART==================================
        # =======Arranging data=====#
        x_Axis = np.array(x_Axis)
        y_Axis = np.array(y_Axis)

        t = np.array(quantity_of_infected)
        r = np.array(days_elapse)
        ymax = max(y_Axis)
        print('The Maximum Number of Cases is :',ymax)
        # print(ymax)
        xmax = max(x_Axis)
        plt.title('Predictive Exponential Model COVID-19 MXN')
        plt.ylabel('Numer of cases (N)')
        plt.xlabel('Time (Days)')
        plt.grid(True)

        # ====================Texts===================
        plt.text(0 * (xmax), 0.9 * (ymax), "Red is the math model",
                 fontdict=font)
        plt.text(0.50 * (xmax), 0.82 * (ymax), r'$Nd_1 = (1 +E*p)*Nd$',
                 fontdict=font3)
        plt.text(0.45 * (xmax), 0.75 * (ymax), "Where ",
                 fontdict=font3, )
        plt.text(0.550 * (xmax), 0.75 * (ymax), r'$Nd = ((1+E*p)^d)*N_0$',
                 fontdict=font3)
        plt.text(0, 0.75 * (ymax), "Blue squares are real data",
                 fontdict=font2)
        plt.text(0, 0.65 * (ymax), "E =",
                 fontdict=font4)
        plt.text(0.03 * (xmax), 0.65 * (ymax), Ep,
                 fontdict=font4)
        plt.text(0 * (xmax), 0.55 * (ymax), "N_0 = ",
                 fontdict=font4)
        plt.text(0.07 * (xmax), 0.55 * (ymax), N_0,
                 fontdict=font4)
        plt.text(0 * (xmax), 0.45 * (ymax), "d =",
                 fontdict=font4)
        plt.text(0.05 * (xmax), 0.45 * (ymax), d,
                 fontdict=font4)
        # max number of cases
        plt.text(0.25 * (xmax), 0.55 * (ymax), "Max number of cases = ",
                 fontdict=font5)
        plt.text(0.60 * (xmax), 0.55 * (ymax), ymax,
                 fontdict=font5)

        plt.plot(x_Axis, y_Axis, 'r', r, t, 'bs')  # PLOTTING DATA

        print('\n')


# ===============Regresion Function=======================
def func_exp(x, a, b, c):
    c = 0
    return a * np.exp(b * x) + c


def exponential_regression(x_data, y_data):
    popt, pcov = curve_fit(func_exp, x_data, y_data, p0=(-1, 0.01, 1))
    print(popt)
    puntos = plt.plot(x_data, y_data, 'x', color='xkcd:maroon', label="Real Cases")
    curva_regresion = plt.plot(x_data, func_exp(x_data, *popt), color='xkcd:teal',
                               label="Coeficients: {:.3f}, {:.3f}, {:.3f}".format(*popt))
    plt.ylabel('Numer of cases (N)')
    plt.xlabel('Time (Days)')
    xmax = max(x_data)
    ymax = max(y_data)
    plt.text(0.1 * (xmax), 0.73 * (ymax), "Model Equation",
             fontdict=font3)
    plt.text(0.1 * (xmax), 0.63 * (ymax), r'$Nd = (a*e^(bx))+c$',
             fontdict=font3)
    plt.grid(True)
    plt.legend()
    #plt.show()
    #print(popt[1])
    return func_exp(x_data, *popt)

def dataestimation(desiredTimeinDays):
    a=1.12389
    b=0.2117
    c=1
    realdata =[]
    x_data2 = np.arange(0, desiredTimeinDays, 1)
    for i in range(0, desiredTimeinDays, 1):
        temp=a * e ** (b * x_data2[i]) + c
        realdata.append(temp)
    plot(x_data2,realdata)
    xmax = max(x_data)
    ymax = max(realdata)
    plt.text(0.1 * (xmax), 0.73 * (ymax), "Model Equation",
             fontdict=font3)
    plt.text(0.1 * (xmax), 0.63 * (ymax), r'$Nd = (a*e^(bx))+c$',
             fontdict=font3)
    plt.ylabel('Numer of cases (N)')
    plt.xlabel('Time (Days)')
    #print(vectora[1]);
# =====================================End of functions============================================


