# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 21:36:23 2020

@author: shubh
"""

import numpy as np

a= np.array([1,2,3])
for item in a:
    print (item)\

#two D array
b = np.array([[1,2,3,3],[4,4,5,6]])
for item in b:
    for i in item:
        print(i)

#no of dimensions
print(a.ndim)
print(b.ndim)

#shape
print(a.shape)
print(b.shape)

#type
print(a.dtype)
c = np.array([2,3,4],dtype='float32')
print(c.dtype)

#size
print(a.itemsize)
print(b.itemsize)
print(c.itemsize)

#no of elements
print(c.size)
print(c.nbytes)

#array splice, fetch operation
arr = np.array([[1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16]])
print(arr[1,2])
print(arr[1,-2])
print(arr[:,3])  #this is printing just the column 3
print(arr[0,:])#this is printing all columns in the first row

#startIndex, endIndex, steps
print(arr[0,-2:-6:-2])

#replacing an entire column
arr[:,2]=1000
print(arr)

#3D array
brr = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(brr.ndim)
print(brr)

#getting elements in 3D array
print(brr[0,:,2])
#replace
brr[:,1,:]=[[9,9,9],[8,8,8]]
print(brr)

#matrix of all Zeroes
crr = np.zeros(5)
print(crr)
err = np.zeros((2,4))
print(err)

#matrix of all 1s
frr = np.ones(5)
print(frr)
grr = np.ones((4,2,3), dtype='int32')
print(grr)

#create a matrix of any other number
hrr = np.full((3,3),1)
print(hrr)

#full-like, take a shape thats already built
irr = np.full_like(hrr,5)
print(irr)
#full using array.shape
jrr = np.full(hrr.shape,6)
print(jrr)

#generating random numbers 2D array
print(np.random.rand(4,2))
#generating random numbers 3D array
print(np.random.rand(4,2,3))
#generating random numbers 2D array using shape
print(np.random.random_sample(grr.shape))

#getting a random integer within a number
print(np.random.randint(8))
print(np.random.randint(7,size=(3,3)))
#getting a random number within a range
print(np.random.randint(4,6,size=(2,3)))

#getting identity matrix
print(np.identity(5, dtype='int32'))
#repeating elements inside a array
zrr = np.array([1,2,3,4])
print(np.repeat(zrr, 3))

#repeating 2D array
xrr = np.array([[1,2,3],[4,5,6]])
print(np.repeat(xrr,3,axis=0))

#repeating a 3D array
yrr = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(np.repeat(yrr,2,axis=1))

#replacing within a 2D array
srr = np.ones((5,5))
trr = np.zeros((3,3))
trr[1,1]=44
srr[1:4,1:4]=trr
print(srr)

#be careful when copying arrays
a= np.array([1,2,3])
b =a.copy()
b[0] = 100
print(b)
print(a)

b= a
b[1]=1000
print(b)
print(a)

#Mathematics
a = np.array([1,2,3,4])
a+=2
print(a+2)
print(a-2)
print(a)
print(a**2)
print(np.sin(a ))

#LinearAlgebra
a = np.ones((2,3))
b = np.full((3,4),5)
print(np.matmul(a,b))

#determinant of a matrix
c = np.full((5,5),9, dtype='int32')
print(np.linalg.det(c))

#statistics with numpy
#min of an array, 2D
srr = np.array([[1,2,3],[4,5,6]])
print(np.min(srr))
print(np.max(srr))
print(np.min(srr,axis=0))
print(np.min(srr, axis =1))
print(np.sum(srr, axis=0))

#reorganizing arrays
before = np.array([[1,2,3],[4,5,6]])
print(before.shape)
after = before.reshape(6,1)
print(after)
after = np.resize(before, (3,3))
print(after)

#stacking arrays vertically
v1 = np.array([1,2,3,4])
v2 = np.array([4,5,6,7])
v3=np.vstack([v1,v2])
print(v3)

#stacking arrays horizontally
h1 = np.ones((2,4))
h2 = np.zeros((2,2))
h3 = np.hstack([h1,h2])
print(h3)


#loading data from a file
fileData = np.genfromtxt('data.txt',delimiter=',')
print(fileData)
print(fileData.astype('int32'))

#advanced indexing and boolean masking
print(fileData>5)
print(fileData[fileData>5])

#indexing with a list in numpy
a = np.array([1,2,3,4,5,6,7,8,9])
print(a[[0,4,6]])

print(np.any(fileData>20,axis=1))
print(np.all(fileData>20,axis=0))
print((fileData>50)& (fileData<100))

