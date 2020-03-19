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
        'weight': 'normal',
        'size': 8.5,
        }

E = 0.5;
p = 0.2;
d = 10;
n_cases = [0, 1, 2, 4, 5, 6, 8, 11, 15, 26, 41, 53, 82, 91, 93, 118]
quantity_of_infected=[0,1,2,4,5,6,7,11,15,26,41,53,82,91,93,108]
days_elapse=[0,1,2,3,4,9,10,14,15,16,17,18,19,20,21,22]


# Nd1 = N * (1 + E * P) ** (Nd0);

def ndplus1(E, p):
    temp = 0;
    if temp <= len(n_cases):
        for i in range(0, len(n_cases), 1):
            nd0 = ((1 + E * p) ** d) * n_cases[i]
            temp = temp + 1;
            # print(temp)
            print(nd0)
    nd1 = nd0
    if temp >= len(n_cases):
        ndn = ((1 + E * p) ** d) * nd1
        print('temp mayor a 16')
        print(ndn)
def nd_n(N_0, Ep, d):
    n_given_day = ((1 + Ep) ** d) * N_0
    # print(n_given_day)
    # print('------------------')
    ndn = ((1 + Ep) * n_given_day)
    print(ndn)

# def n_d(N_0, Ep, d):
#   temp = 1
#   for i in range(0, d, 1):
#       nd = N_0 * (1 + Ep) ** temp
#       nd=round(nd)
#        temp = temp + 1
# print(nd)

def n_d(N_0, Ep, d):
    nd = N_0 * (1 + Ep) ** d;
    nd = round(nd)
    return nd;


def nd_n(N_0, Ep, d):
    n_given_day = ((1 + Ep) ** d) * N_0
    # print(n_given_day)
    # print('------------------')
    ndn = ((1 + Ep) * n_given_day)
    print(ndn)

# def n_d(N_0, Ep, d):
#   temp = 1
#   for i in range(0, d, 1):
#       nd = N_0 * (1 + Ep) ** temp
#       nd=round(nd)
#        temp = temp + 1
# print(nd)

def n_d(N_0, Ep, d):
    nd = N_0 * (1 + Ep) ** d;
    nd = round(nd)
    return nd;


def plotting(d):
    y_Axis = []
    x_Axis = []
    for i in range(0, d, 1):
        y_Axis.append(n_d(N_0, Ep, (i+1)))
        #print(i)
    #print('fin clico')

    print('The Maximum Number of Cases is :',y_Axis[d-1])
    for i in range(0, d , 1):
        x_Axis.append(i)
    print('In ', d ,' days since day 0')
    # print(y_Axis)
    # print(x_Axis)

    #===========================================================================
    #================================PLOT PART==================================
    #=======Arranging data=====#
    x_Axis = np.array(x_Axis)
    y_Axis = np.array(y_Axis)
    t = np.array(quantity_of_infected)
    r = np.array(days_elapse)
    ymax = max(y_Axis)
    #print(ymax)
    xmax = max(x_Axis)
    plt.title('Predictive Exponential Model COVID-19 MXN')
    plt.ylabel('Numer of cases (N)')
    plt.xlabel('Time (Days)')
    plt.grid(True)

    #====================Texts===================
    plt.text(0, 1700, "Red is the math model",
             fontdict=font)
    plt.text(15, 1700, r'$Nd_1 = (1 +E*p)*Nd$',
             fontdict=font3)
    plt.text(12.5, 1550, "Where ",
             fontdict=font3, )
    plt.text(16, 1550, r'$Nd = ((1+E*p)^d)*N_0$',
             fontdict=font3)
    plt.text(0, 1400, "Blue squares are real data",
             fontdict=font2)
    plt.plot(x_Axis, y_Axis,'r', r, t, 'bs') #PLOTTING DATA
    plt.text(0, 1200, "E:Average of people and infected is exposed each day ",
             fontdict=font4)
    plt.text(0, 1000, "C = contant of proporcionality",
             fontdict=font4)
    plt.text(0, 800, "d = number of days the decease",
             fontdict=font4)
    print('\n')



#nd_n(N_0,Ep,d)
#print(n_d(N_0,Ep,d))

#t= [0,1,2,3,4,5,6,7,8,9]
#t = np.array(t)

#quantity_of_infected=[0,1,2,4,5,6,7,11,15,26,41,53,82,91,93,108]
#days_elapse=[0,1,2,3,4,9,10,14,15,16,17,18,19,20,21,22]

#t =np.array(quantity_of_infected)
#r =np.array(days_elapse)
#plt.plot(r, t, 'bs')
#plt.show()

Ep = 0.297;
N_0 = 0.77;
d = 30;


#plotting(d)
plt.subplot(221)
plotting(d)
plt.subplot(222)
plotting(40)
plt.show()



