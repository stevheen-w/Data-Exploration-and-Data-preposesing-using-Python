#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
oneDim = np.array([1.0,2,3,4,5]) # a 1-dimensional array (vector)
print(oneDim)
print("#Dimensions =", oneDim.ndim)
print("Dimension =", oneDim.shape)
print("Size =", oneDim.size)
print("Array type =", oneDim.dtype)

twoDim = np.array([[1,2],[3,4],[5,6],[7,8]]) # a two-dimensional array (matrix)
print(twoDim)
print("#Dimensions =", twoDim.ndim)
print("Dimension =", twoDim.shape)
print("Size =", twoDim.size)
print("Array type =", twoDim.dtype)

arrFromTuple = np.array([(1,'a',3.0),(2,'b',3.5)]) # create ndarray from tuple
print(arrFromTuple)
print("#Dimensions =", arrFromTuple.ndim)
print("Dimension =", arrFromTuple.shape)
print("Size =", arrFromTuple.size)


# In[2]:


print(np.random.rand(5)) # random numbers from a uniform distribution␣
("→between", "[0,1]")
print(np.random.randn(5)) # random numbers from a normal distribution
print(np.arange(-10,10,2)) # similar to range, but returns ndarray instead␣
("→of", "list")
print(np.arange(12).reshape(3,4)) # reshape to a matrix
print(np.linspace(0,1,10)) # split interval [0,1] into 10 equally separated␣
("→values")
print(np.logspace(-3,3,7)) # create ndarray with values from 10^-3 to 10^3


# In[3]:


print(np.zeros((2,3))) # a matrix of zeros
print(np.ones((3,2))) # a matrix of ones
print(np.eye(3)) # a 3 x 3 identity matrix


# In[4]:


x = np.array([1,2,3,4,5])
print(x + 1) # addition
print(x - 1) # subtraction
print(x * 2) # multiplication
print(x // 2) # integer division
print(x ** 2) # square
print(x % 2) # modulo
print(1 / x) # division


# In[5]:


x = np.array([2,4,6,8,10])
y = np.array([1,2,3,4,5])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x // y)
print(x ** y)


# In[6]:


x = np.arange(-5,5)
print(x)
y = x[3:5] # y is a slice, i.e., pointer to a subarray in x
print(y)
y[:] = 1000 # modifying the value of y will change x
print(y)
print(x)
z = x[3:5].copy() # makes a copy of the subarray
print(z)
z[:] = 500 # modifying the value of z will not affect x
print(z)
print(x)


# In[7]:


my2dlist = [[1,2,3,4],[5,6,7,8],[9,10,11,12]] # a 2-dim list
print(my2dlist)
print(my2dlist[2]) # access the third sublist
print(my2dlist[:][2]) # can't access third element of each sublist
# print(my2dlist[:,2]) # this will cause syntax error
my2darr = np.array(my2dlist)
print(my2darr)
print(my2darr[2][:]) # access the third row
print(my2darr[2,:]) # access the third row
print(my2darr[:][2]) # access the third row (similar to 2d list)
print(my2darr[:,2]) # access the third column
print(my2darr[:2,2:]) # access the first two rows & last two columns


# In[8]:


my2darr = np.arange(1,13,1).reshape(3,4)
print(my2darr)
divBy3 = my2darr[my2darr % 3 == 0]
print(divBy3, type(divBy3))
divBy3LastRow = my2darr[2:, my2darr[2,:] % 3 == 0]
print(divBy3LastRow)


# In[9]:


my2darr = np.arange(1,13,1).reshape(4,3)
print(my2darr)
indices = [2,1,0,3] # selected row indices
print(my2darr[indices,:])
rowIndex = [0,0,1,2,3] # row index into my2darr
columnIndex = [0,2,0,1,2] # column index into my2darr
print(my2darr[rowIndex,columnIndex])


# In[10]:


y = np.array([-1.4, 0.4, -3.2, 2.5, 3.4]) # generate a random vector
print(y)
print(np.abs(y)) # convert to absolute values
print(np.sqrt(abs(y))) # apply square root to each element
print(np.sign(y)) # get the sign of each element
print(np.exp(y)) # apply exponentiation
print(np.sort(y))


# In[11]:


x = np.arange(-2,3)
y = np.random.randn(5)
print(x)
print(y)
print(np.add(x,y)) # element-wise addition x + y
print(np.subtract(x,y)) # element-wise subtraction x - y
print(np.multiply(x,y)) # element-wise multiplication x * y
print(np.divide(x,y)) # element-wise division x / y
print(np.maximum(x,y)) # element-wise maximum max(x,y)


# In[12]:


y = np.array([-3.2, -1.4, 0.4, 2.5, 3.4]) # generate a random vector
print(y)
print("Min =", np.min(y)) # min
print("Max =", np.max(y)) # max
print("Average =", np.mean(y)) # mean/average
print("Std deviation =", np.std(y)) # standard deviation
print("Sum =", np.sum(y)) # sum


# In[13]:


X = np.random.randn(2,3) # create a 2 x 3 random matrix
print(X)
print(X.T) # matrix transpose operation X^T
y = np.random.randn(3) # random vector
print(y)
print(X.dot(y)) # matrix-vector multiplication X * y
print(X.dot(X.T)) # matrix-matrix multiplication X * X^T
print(X.T.dot(X)) # matrix-matrix multiplication X^T * X


# In[14]:


X = np.random.randn(5,3)
print(X)
C = X.T.dot(X) # C = X^T * X is a square matrix
invC = np.linalg.inv(C) # inverse of a square matrix
print(invC)
detC = np.linalg.det(C) # determinant of a square matrix
print(detC)
S, U = np.linalg.eig(C) # eigenvalue S and eigenvector U of a square matrix
print(S)
print(U)


# In[15]:


from pandas import Series
s = Series([3.1, 2.4, -1.7, 0.2, -2.9, 4.5]) # creating a series from a list
print(s)
print('Values=', s.values) # display values of the Series
print('Index=', s.index) # display indices of the Series


# In[16]:


import numpy as np
s2 = Series(np.random.randn(6)) # creating a series from a numpy ndarray
print(s2)
print('Values=', s2.values) # display values of the Series
print('Index=', s2.index) # display indices of the Series


# In[17]:


s3 = Series([1.2,2.5,-2.2,3.1,-0.8,-3.2],

index = ['Jan 1','Jan 2','Jan 3','Jan 4','Jan 5','Jan 6',])

print(s3)
print('Values=', s3.values) # display values of the Series
print('Index=', s3.index) # display indices of the Series


# In[20]:


s3 = Series([1.2,2.5,-2.2,3.1,-0.8,-3.2],

index = ['Jan 1','Jan 2','Jan 3','Jan 4','Jan 5','Jan 6',])

print(s3)
# Accessing elements of a Series
print('\ns3[2]=', s3[2]) # display third element of the Series
print('s3[\'Jan 3\']=', s3['Jan 3']) # indexing element of a Series
print('\ns3[1:3]=') # display a slice of the Series
print(s3[1:3])
print('s3.iloc([1:3])=') # display a slice of the Series
print(s3.iloc[1:3])


# In[21]:


print('shape =', s3.shape) # get the dimension of the Series
print('size =', s3.size) # get the # of elements of the Series


# In[22]:


print(s3[s3 > 0]) # applying filter to select elements of the Series


# In[23]:


print(s3 + 4) # applying scalar operation on a numeric Series
print(s3 / 4)


# In[24]:


print(np.log(s3 + 4)) # applying numpy math functions to a numeric Series


# In[26]:


from pandas import DataFrame
cars = {'make': ['Ford', 'Honda', 'Toyota', 'Tesla'],
'model': ['Taurus', 'Accord', 'Camry', 'Model S'],
'MSRP': [27595, 23570, 23495, 68000]}
carData = DataFrame(cars) # creating DataFrame from dictionary
carData # display the table


# In[27]:


print(carData.index) # print the row indices
print(carData.columns) # print the column indices


# In[29]:


carData2 = DataFrame(cars, index = [1,2,3,4]) # change the row index
carData2['year'] = 2018 # add column with same value
carData2['dealership'] = ['Courtesy Ford','Capital Honda','Spartan Toyota','N/A']
carData2 # display table


# In[30]:


tuplelist = [(2011,45.1,32.4),(2012,42.4,34.5),(2013,47.2,39.2),
(2014,44.2,31.4),(2015,39.9,29.8),(2016,41.5,36.7)]

columnNames = ['year','temp','precip']
weatherData = DataFrame(tuplelist, columns=columnNames)
weatherData


# In[31]:


import numpy as np
npdata = np.random.randn(5,3) # create a 5 by 3 random matrix
columnNames = ['x1','x2','x3']
data = DataFrame(npdata, columns=columnNames)
data


# In[32]:


# accessing an entire column will return a Series object
print(data['x2'])
print(type(data['x2']))


# In[33]:


# accessing an entire row will return a Series object
print('Row 3 of data table:')
print(data.iloc[2]) # returns the 3rd row of DataFrame
print(type(data.iloc[2]))
print('\nRow 3 of car data table:')
print(carData2.iloc[2]) # row contains objects of different types


# In[34]:


# accessing a specific element of the DataFrame
print(carData2.iloc[1,2]) # retrieving second row, third column
print(carData2.loc[1,'model']) # retrieving second row, column named 'model'
# accessing a slice of the DataFrame
print('carData2.iloc[1:3,1:3]=')
print(carData2.iloc[1:3,1:3])


# In[35]:


print('carData2.shape =', carData2.shape)
print('carData2.size =', carData2.size)


# In[36]:


# selection and filtering
print('carData2[carData2.MSRP > 25000]')
print(carData2[carData2.MSRP > 25000])


# In[37]:


print(data)
print('Data transpose operation:')
print(data.T) # transpose operation
print('Addition:')
print(data + 4) # addition operation
print('Multiplication:')
print(data * 10) # multiplication operation


# In[38]:


print('data =')
print(data)
columnNames = ['x1','x2','x3']
data2 = DataFrame(np.random.randn(5,3), columns=columnNames)
print('\ndata2 =')
print(data2)
print('\ndata + data2 = ')
print(data.add(data2))
print('\ndata * data2 = ')
print(data.mul(data2))


# In[39]:


print(data.abs()) # get the absolute value for each element
print('\nMaximum value per column:')
print(data.max()) # get maximum value for each column
print('\nMinimum value per row:')
print(data.min(axis=1)) # get minimum value for each row
print('\nSum of values per column:')
print(data.sum()) # get sum of values for each column
print('\nAverage value per row:')
print(data.mean(axis=1)) # get average value for each row
print('\nCalculate max - min per column')
f = lambda x: x.max() - x.min()
print(data.apply(f))
print('\nCalculate max - min per row')
f = lambda x: x.max() - x.min()
print(data.apply(f, axis=1))


# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
s3 = Series([1.2,2.5,-2.2,3.1,-0.8,-3.2,1.4],

index = ['Jan 1','Jan 2','Jan 3','Jan 4','Jan 5','Jan 6','Jan 7'])

s3.plot(kind='line', title='Line plot')


# In[41]:


s3.plot(kind='bar', title='Bar plot')


# In[42]:


s3.plot(kind='hist', title = 'Histogram')


# In[43]:


tuplelist = [(2011,45.1,32.4),(2012,42.4,34.5),(2013,47.2,39.2),
(2014,44.2,31.4),(2015,39.9,29.8),(2016,41.5,36.7)]

columnNames = ['year','temp','precip']
weatherData = DataFrame(tuplelist, columns=columnNames)
weatherData[['temp','precip']].plot(kind='box', title='Box plot')


# In[ ]:




