#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Study group 08
"""

#%% Exercise 1
#%% IMPORT AND READ DATASET
from multiprocessing.connection import wait
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



data = np.load('megmag_data.npy')



#%% Describe data structure
dshape  = data.shape
print(dshape)

#%% Add time offset (ms)
times = np.arange(-200, 801, 4)
print(times.shape)



#%% Create covariance matrix
output = []
for i in range(len(data)):
  X = data[i]
  Xt = X.T
  output.append(np.matrix(np.dot(X,Xt)))

cov_mat = 1  / dshape[0] * np.sum(i for i in output)

print(cov_mat.shape)

plt.imshow(cov_mat)
plt.show()

#%% Create signal averages across repititions
avg_for_reps = np.mean(data, axis=0)

plt.plot(avg_for_reps)
plt.axvline()
plt.axhline()
plt.show()

#%% Find and plot maximum response channel
max_resp = np.unravel_index(np.argmax(avg_for_reps), avg_for_reps.shape)
print(max_resp)
plt.plot(data[:,max_resp[0],:])
plt.axvline()
plt.axvline(x = max_resp[1], color='r', label='max response')
plt.axhline()
plt.show()

#%% load pas_vector.npy (call it y)
y = np.load('pas_vector.npy')

# same as number of repititions
print(y.shape)
pas_data = []
pas_index = {}
for i in np.unique(y):
  print(i)
  pas_slice = np.argwhere(y == i)
  data_slice = data[pas_slice].squeeze()

  pas_data.append({
    'paslevel': i,
    'data': data_slice,
  })
  pas_index[i] = len(pas_data) - 1

for i in pas_data:
  print('Pas {} shape: {}'.format(i['paslevel'], i['data'].shape))
  avg_for_pas = np.mean(i['data'], axis=0)
  print('Pas {} average shape: {}'.format(i['paslevel'], avg_for_pas.shape))
  plt.plot(times, avg_for_pas[0], label='Pas {}'.format(i['paslevel']))

plt.axvline()
plt.axhline()
plt.axvline(x = max_resp[1], color='r', label='max response')
plt.legend(loc="upper left")
plt.title('Average response for each pas level')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.show()


#%% Exercise 2
# create a new array called data_1_2 that only contains PAS responses 1 and 2

data_1_2 = np.concatenate((pas_data[pas_index[1]]['data'], pas_data[pas_index[2]]['data']), axis = 0)
print(data_1_2.shape)
# Similarly, create a y_1_2 for the target vector
y_1_2 = y[np.where((y == 1) | (y == 2))]

#Our data_1_2 is a three-dimensional array. Our strategy will be to collapse our two last dimensions (sensors and time) into one dimension, while keeping the first dimension as it is (repetitions). Use np.reshape to create a variable X_1_2 that fulfils these criteria.
X_1_2 = np.reshape(data_1_2, (data_1_2.shape[0], data_1_2.shape[1] * data_1_2.shape[2]))

print(X_1_2.shape)
#  and scale X_1_2
sc = StandardScaler()
Y_1_2_fit = sc.fit(X_1_2)
X_1_2_std = sc.transform(X_1_2)

# Do a standard LogisticRegression - can be imported from sklearn.linear_model - make sure there is no penalty applied
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
print(lr.fit(X_1_2_std, y_1_2))

print(lr.score(X_1_2_std, y_1_2))
