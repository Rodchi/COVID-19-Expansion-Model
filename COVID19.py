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
font = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 10,
        }
font2 = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 10,
        }
font3 = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 10,
        }
font4 = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 8.5,
        }
font5 = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 10,
        }
#========================Real data=============================
quantity_of_infected=[0,1,2,4,5,6,7,11,15,26,41,53,82,91,93,108] #number of infected
days_elapse=[0,1,2,3,4,9,10,14,15,16,17,18,19,20,21,22] #days elapsed from day 0


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

def plotting(N_0,Ep,d):
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
       #plt.plot(x_Axis, y_Axis, 'r', r, t, 'bs')  # PLOTTING DATA
        plt.text(0.1, 0.55, "E: Average of people and infected is exposed each day ",
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

        print('The Maximum Number of Cases is :', y_Axis[d - 1])
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
        # print(ymax)
        xmax = max(x_Axis)
        plt.title('Predictive Exponential Model COVID-19 MXN')
        plt.ylabel('Numer of cases (N)')
        plt.xlabel('Time (Days)')
        plt.grid(True)

        # ====================Texts===================
        plt.text(0*(xmax), 0.9*(ymax), "Red is the math model",
                 fontdict=font)
        plt.text(0.50*(xmax), 0.82*(ymax), r'$Nd_1 = (1 +E*p)*Nd$',
                 fontdict=font3)
        plt.text(0.45*(xmax), 0.75*(ymax), "Where ",
                 fontdict=font3, )
        plt.text(0.550*(xmax), 0.75*(ymax), r'$Nd = ((1+E*p)^d)*N_0$',
                 fontdict=font3)
        plt.text(0, 0.75*(ymax), "Blue squares are real data",
                 fontdict=font2)
        plt.text(0, 0.65*(ymax), "E =",
                 fontdict=font4)
        plt.text(0.03*(xmax), 0.65 * (ymax), Ep,
                 fontdict=font4)
        plt.text(0*(xmax),0.55*(ymax), "N_0 = ",
                 fontdict=font4)
        plt.text(0.07*(xmax), 0.55*(ymax), N_0,
                 fontdict=font4)
        plt.text(0*(xmax),0.45*(ymax), "d =",
                 fontdict=font4)
        plt.text(0.05*(xmax), 0.45*(ymax), d,
                 fontdict=font4)
        #max number of cases
        plt.text(0.25 * (xmax), 0.55 * (ymax), "Max number of cases = ",
                 fontdict=font5)
        plt.text(0.60* (xmax), 0.55 * (ymax), ymax,
                 fontdict=font5)

        plt.plot(x_Axis, y_Axis, 'r', r, t, 'bs')  # PLOTTING DATA


        print('\n')





Ep1= 0.297;
Ep2 = 0.307;
Ep3 = 0.287;
N_0 = 0.77;
d = 30;

suptitle('Expectations for COVID-19 MXN ')
plt.subplot(2,2,1)
plotting(N_0, Ep1, 0)
plt.subplot(2,2,2)
plotting(N_0, Ep1, d)
plt.subplot(2,2,3)
plotting(N_0, Ep2, d)
plt.subplot(2,2,4)
plotting(N_0, Ep3, d)
show()
#plt.subplot(421)
#plotting(d)
#plt.subplot(412)
#plotting(45)
#subplot(422)
#plotting(60)
#plt.show()

