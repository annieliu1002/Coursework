# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:41:03 2022

@author: rodri
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


# =========  EX1: Preliminaries =========
import os
# os.chdir('my folder path')
import this

#In the good gooding practices we also saw that readability counts, simple is better than complex, and that breaks are necessary. 
#The rest are new recommenadations!

# ========= EX 2: Counting the number of times a word is repeated in a string. =========

str_x = '''Key facts about the distribution of wealth have been highlighted in a large number of studies,
 including Wolff (1992, 1998), Cagetti and De Nardi (2008), and Moritz and Rios-Rull (2015).
 A striking aspect of the wealth distribution in the US is its degree of concentration. 
 Over the past 30 years or so, for instance, households in the top 1% of the wealth distribution have held about one-third of the total wealth in the economy, and those in the top 5% have held more than half.
 At the other extreme, more than 10% of households have little or no assets. 
 While there is agreement that the share held by the richest few is very high, 
 the extent to which the shares of the richest have changed over time (and why) 
 is still the subject of some debate (Piketty 2014, Kopczuk 2014, Saez and Zucman 2014, and Bricker et al. 2015).'''


cnt = str_x.count("wealth") # use count method of a str class
print('The word wealth is repeated', cnt, 'times.')

# A more efficient implementation through a for-loop: remember DRY, be lazy, automate.
list_words = ['wealth','distribution','household','assets']

for word in list_words:   
    cnt = str_x.count(word)
    print(word,'is repeated', cnt,'times.')





# ========= EX 3: Compute and plot 1-D functions =========

# 1a
x = np.linspace(0,10,100)  # create a linear space
y= np.log(x)   #note that functions in numpy are vectorized. They act element-wise.
z=np.sin(x)

#What type is y? what size is y? and z?


# 1b
# Plot for log(x)
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('log function')
plt.show()

# Plot for sin(x)
fig, ax = plt.subplots()
ax.plot(x, z, linewidth=2.0, color='r')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title('sinus function')
plt.show()



# ========= EX 4: a 2-dimensional function =========

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



# ========= EX 5: Solving the consumer problem =========

# Parameters values
alpha = 0.5
y = 10
p1 = 1
p2 = 2


def utility_function(x1,alpha,y,p1,p2):  
    x2 = (y-p1*x1)/p2  # Income not spent first good is spent on the second
    utility = x1**alpha * x2**(1-alpha)
    
    return utility


# Make utility_function as a function of only x1 using the lambda expression
obj_func = lambda x1: -utility_function(x1,alpha,y,p1,p2)  #Note the minus in front so that we are maximizing!


# Minimize function. Note we use bounds to x1 (the min and max value x1 can take) to solve the minimization problem
res = minimize_scalar(obj_func,bounds=(0,y/p1))

x1_star = res.x
x2_star =(y-p1*x1_star)/p2 

print('the solution is x1=',x1_star,'x2=',x2_star)



