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
for i in np.unique(y):
  print(i)
  pas_slice = np.unravel_index(np.argwhere(y == i), y.shape)
  pas_data.append({
    'paslevel': i,
    'data': data[pas_slice[0],max_resp[0],:],
  })

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
data_1_2 = data[np.argwhere(y == 1) | np.argwhere(y == 2)]
# Similarly, create a y_1_2 for the target vector
y_1_2 = y[np.where(y == 1) | np.where(y == 2)]
