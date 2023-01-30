#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:12:29 2023

@author: anniecadanie
"""

# This file is used for PS0 in Programming for Economics

# Importing relevant libraries
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

# Exercise 1

l1 = [2, 5, 6, 4, 5 ,9, 3, 2, 2]

# a) Extract and print the first 3 elements of the list ([2,5,6]). Extract and print the last 2 elements.
print(l1[:3]) # Extracting first 3 elements
print(l1[-2:]) # Extracting last 2 elements

# b) Replace the 3rd element in the list for number 4. Replace the 7th element in the list for number 7. Print the new list.
l1[2] = 4
l1[6] = 7
print(l1)

# c) Replace the number 9 for 8 using list.index()
l1 = [2, 5, 6, 4, 5 ,9, 3, 2, 2]
l1[l1.index(9)] = 8
print(l1)

# d) Replace number 9 for number 8 using NumPy argmax.
l1 = [2, 5, 6, 4, 5 ,9, 3, 2, 2]
l1[np.argmax(l1)] = 8
print(l1)


# Exercise 2

l1 = [2, 5, 6, 4, 5 ,9, 3, 2, 2]

# a) Convert the list into a 3 × 3 matrix (2-d array)
flat_A = np.array(l1)
shape = (3,3)
A = flat_A.reshape(shape)
print(A)

# b) Find the maximum of the matrix A. Find the index of the maximum.
A_max = A.max()    # maximum element
print(A_max)

A_max_index = A.argmax()   # index of the maximum element
print(A_max_index)

# c) Transpose the matrix A
A_transposed = A.transpose()
print(A_transposed)

# d) Squared the matrix (i.e. AA′). Raise to the power all the elements in matrix A
A_squared = A@A_transposed # Squaring matrix A
print(A_squared)

A_elements_equared = A**2 # Squaring all the elements in matrix A

# e) Compute the eigenvalues of matrix A
w,v=eig(A)
print('Eigenvalues:', w)
print('Eigenvector', v)

# f) Multiply matrix A by a matrix of zeros (f1), by the identity matrix (f2),and by a matrix of ones (f3).
A_times_zero = A@np.zeros((3,3)) # f1
print (A_times_zero)

A_times_identity = A@np.identity(3) # f2
print(A_times_identity)

A_times_ones = A@np.ones((3,3)) # f3
print(A_times_ones)      
      
# g) Create an even grid from 0 to 9 with 9 elements. Convert the grid into a 3x3 matrix called B. Multiply matrix A by matrix B.
grid_b = np.linspace(0,9,9) # Creating grid

shape = (3,3) # Converting grid into 3x3 matrix
B = grid_b.reshape(shape)

AB = A@B # Multiplying matrix A by matrix B
print(AB)


# Exercise 3 
# Write a code to iterate the first 10 numbers and in each iteration, print the sum of the current and previous number.

print('Printing current number, previous number, and sum of first 10 numbers')
previous_number = 0
for i in range(1,11):
    sum_numbers = previous_number + i
    print('Current number: ', i, 'Previous number: ', previous_number,'Sum: ', sum_numbers)
    previous_number = i


# Exercise 4

l1 = [2, 5, 6, 4, 5 ,9, 3, 2, 2]

# a) Create a new list that contains only the elements in list l1 that are smaller than 5.
smaller_than_5 = [] # Create empty list

for i in l1: # Create loop to add elements under condition
    if i < 5:
        smaller_than_5.append(i)

print(smaller_than_5)

# b) Create a new list that contains only the elements in list l1 bigger or equal than 3 and smaller than 7.
between_3_7 = [] # Create empty list

for i in l1: # Create loop to add elements under condition
    if i >= 3 and i < 7:
        between_3_7.append(i)

print(between_3_7)

# c) Given matrix A from exercise 2, write a code that checks whether 5 belongs to A.
(A == 5).any() # Checks whether 5 belongs to A

# d) Create a new matrix B that is equal to matrix A but where numbers below 4 are replaced by zeroes.

list_B = [] # Creating an empty list

for row in A: # For loop to add values to list
    for element in row:
        if element >= 4:
            list_B.append(element)
        else:
            list_B.append(0)
          
flat_B = np.array(list_B) # Turning list into array B
shape = (3,3)
B = flat_B.reshape(shape)
print(B)

# e) Write a code that counts the number of zeros in matrix B.
zero_count = 0

for row in B:
    for element in row:
        if element == 0:
            zero_count += 1

print(zero_count)


# Exercise 5

# a) Create a function that given the arguments K, L, A, α, σ, returns output Y. You can have inside the function an if statement for 
#    when σ = 1 the output Y comes from Cobb-Douglas production function, else from the CES function.
#    From now on work with the following parameterization: A = 1.5, α = 0.33.

# Parameters values
A = 1.5
alpha = 0.33

# Creating function
def Y(K, L, A, alpha, sigma):
    if sigma == 1:
        cobb_douglas = A * K**alpha * L**(1-alpha)
        return cobb_douglas
    else:
        CES = A * (alpha * K**((sigma-1)/sigma) + (1-alpha) * L**((sigma-1)/sigma))**(sigma/(sigma-1))
        return CES

# b) Cobb-Douglass production function. First consider the Cobb-Douglass case with σ = 1. Compute output Y for an even-spaced grid of K, 
#    G_k = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, and a fixed L = 3. Plot the resuls—make an x-y plot with the grid of K in the x and output 
#    Y in the axis y.

K = np.linspace(0,10,11) # Creating K
L = 3 # Defining fixed L = 3
sigma = 1 # Defining fixed sigma = 1

output_Y = Y(K,L,A,alpha,sigma)

# Creating x-y plot
plt.plot(K, output_Y)

# c) From b recompute output Y for the 3 cases α = 0.25, α = 0.5, α = 0.75. Make an x − y plot with the 3 production functions in the same graph.

# Recomputing output Y for the 3 Cobb-Douglas cases
case1_sigma = 0.25 # Defining Case 1: α = 0.25
case1_output_Y = Y(K,L,A,alpha,case1_sigma)

case2_sigma = 0.5 # Defining Case 2: α = 0.5
case2_output_Y = Y(K,L,A,alpha,case2_sigma)

case3_sigma = 0.75 # Defining Case 3: α = 0.75
case3_output_Y = Y(K,L,A,alpha,case3_sigma)

# Plotting all 3 production functions together
plt.plot(K, case1_output_Y, K, case2_output_Y, K, case3_output_Y)
plt.show()

# d) CES production function. Redo exercise b but for σ = 0.33

sigma_CES = 0.33 # Defining new sigma
output_Y_CES = Y(K,L,A,alpha,sigma_CES)

# Creating x-y plot
plt.plot(K, output_Y_CES)
plt.show()

# e) Keeping α = 0.33, plot output Y vs the grid of capital for the cases of σ = 0.25, σ = 0.5, σ = 1, σ = 2, σ = 4

# Recomputing output Y for the 5 CES cases
CEScase1_sigma = 0.25 # Defining Case 1: α = 0.25
CEScase1_output_Y = Y(K,L,A,alpha,case1_sigma)

CEScase2_sigma = 0.5 # Defining Case 2: α = 0.5
CEScase2_output_Y = Y(K,L,A,alpha,case2_sigma)

CEScase3_sigma = 1 # Defining Case 3: α = 1
CEScase3_output_Y = Y(K,L,A,alpha,case3_sigma)

CEScase4_sigma = 2 # Defining Case 4: α = 2
CEScase4_output_Y = Y(K,L,A,alpha,CEScase4_sigma)

CEScase5_sigma = 4 # Defining Case 5: α = 4
CEScase5_output_Y = Y(K,L,A,alpha,CEScase5_sigma)

# Plotting all 5 production functions together
plt.plot(K, CEScase1_output_Y, K, CEScase2_output_Y, K, CEScase3_output_Y, K, CEScase4_output_Y, K, CEScase5_output_Y)
plt.show()

# f) How does output Y changes along K for the different σ specifications? Can you provide the economic interpretation? Hint: σ captures 
#    the relative degree of substitutability/complementarity between the two inputs K, L.





