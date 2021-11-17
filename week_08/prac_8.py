from multiprocessing.connection import wait
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

data = np.load('megmag_data.npy')
pas_vec = np.load('pas_vector.npy')

dshape  = data.shape

times = np.arange(-200, 801, 4)
times.shape


output = []
for i in range(len(data)):
  X = data[i]
  Xt = X.T
  output.append(np.matrix(np.dot(X,Xt)))

cov_mat = 1  / dshape[0] * np.sum(i for i in output)

print(cov_mat.shape)

plt.imshow(cov_mat)
plt.show()

print("hello")

avg_for_reps = np.mean(data, axis=0)

plt.plot(avg_for_reps)
plt.axvline()
plt.axhline()
plt.show()

max_resp = np.unravel_index(np.argmax(avg_for_reps), avg_for_reps.shape)

plt.plot(avg_for_reps[73])
plt.axvline()
plt.axvline(x = max_resp[1], color='r')
plt.axhline()
plt.show()

print("hi")