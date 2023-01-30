#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:15:49 2023

@author: anniecadanie
"""

# This file is used for PS0 in Programming for Economics

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Exercise 1

import os
os.chdir("/Users/anniecadanie/Desktop/University/3rd Year/Programming for Economics")

import this
#In the good coding practices we also saw that readability counts, simple is better than complex, and that breaks are necessary. 
#The rest are new recommenadations!


# Exercise 2

str_x = '''Key facts about the distribution of wealth have been highlighted in a large number of studies,
 including Wolff (1992, 1998), Cagetti and De Nardi (2008), and Moritz and Rios-Rull (2015).
 A striking aspect of the wealth distribution in the US is its degree of concentration. 
 Over the past 30 years or so, for instance, households in the top 1% of the wealth distribution have held about one-third of the total wealth in the economy, and those in the top 5% have held more than half.
 At the other extreme, more than 10% of households have little or no assets. 
 While there is agreement that the share held by the richest few is very high, 
 the extent to which the shares of the richest have changed over time (and why) 
 is still the subject of some debate (Piketty 2014, Kopczuk 2014, Saez and Zucman 2014, and Bricker et al. 2015).'''
 
wealth_count = str_x.count('wealth')
print('Wealth occurs', wealth_count, 'times.')

distribution_count = str_x.count('distribution')
print('Distribution occurs', distribution_count, 'times.')

households_count = str_x.count('households')
print('Households occurs', households_count, 'times.')

assets_count = str_x.count('assets')
print('Assets occurs', assets_count, 'times.')

# A more efficient implementation through a for-loop: remember DRY, be lazy, automate.
list_words = ['wealth','distribution','household','assets']

for word in list_words:   
    count = str_x.count(word)
    print(word,'occurs', count,'times.')


# Exercise 3

# Make data
x = np.linspace(0,10,100)
y = np.log(x)
z = np.sin(x)

# Plot graphs
fig, ax = plt.subplots() # Plot for log(x)
ax.plot(x, y, linewidth=2.0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('log function')
plt.show()

fig, ax = plt.subplots() # Plot for sin(x)
ax.plot(x, z, linewidth=2.0, color='r')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title('sinus function')
plt.show()


# Exercise 4

# Make data
x = np.linspace(0,10,100)
y = np.linspace(0,10,100)
X, Y = np.meshgrid(x, y)   # create the cartesian product of x and y
Z = np.log(X) + np.sin(Y)

# Plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('f(x,y) = log(x) + sin(y)', fontsize=14)
plt.show()


# Exercise 5

# Parameters values
alpha = 0.5
y = 10
p1 = 1
p2 = 2

# Define utility function
def utility_function(x1,alpha,y,p1,p2):
    x2 = (y-p1*x1)/p2 # Income not spent first good is spent on the second
    utility = x1**alpha * x2**(1-alpha)
    return utility

# Make utility_function as a function of only x1 using the lambda expression
obj_func = lambda x1: -utility_function(x1,alpha,y,p1,p2) #Note the minus in front so that we are maximizing!


# Minimizee function. Note we use bounds to x1 (the min and max value x1 can take) to solve the minimization problem
res = minimize_scalar(obj_func,bounds=(0,y/p1))

x1_star = res.x
x2_star = (y-p1*x1_star)/p2

print('The consumer would buy', x1_star, 'units of x1 and', x2_star, 'units of x2.')