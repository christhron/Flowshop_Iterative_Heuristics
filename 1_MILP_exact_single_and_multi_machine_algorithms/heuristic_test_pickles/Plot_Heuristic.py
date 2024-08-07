# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 21:28:25 2023

@author: matth
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

timesFilename = 'heuristic_times.pickle'

with open(timesFilename, 'rb') as f:
      job_count=pickle.load(f)
      time_count=pickle.load(f)

tMean = np.mean(time_count, axis=1)

# Plotting the line plot
plt.plot(job_count, tMean, marker='o', linestyle='-', color='b', label='Average Runtime')

# Adding labels and title
plt.xlabel('Number of Jobs')
plt.ylabel('Average Runtime (Seconds)')
plt.title('Average Runtime for Each Job Count')
plt.xticks(job_count.flatten())

# Display the plot
plt.show()

orderCountFilename = 'orderCount.pickle'
num_orders=np.empty([6,15000])

with open(orderCountFilename, 'rb') as f:
    for i in range(6):
      num_orders[i]=pickle.load(f).flatten()

ordMean = np.mean(num_orders, axis=1)

# Plotting the line plot
plt.plot(job_count, ordMean, marker='o', linestyle='-', color='r', label='Average Number of Orders')

# Adding labels and title
plt.xlabel('Number of Jobs')
plt.ylabel('Average Number of Saved Orders')
plt.title('Average Number of Saved Orders for Each Job Count')
plt.xticks(job_count.flatten())

# Display the plot
plt.show()

plt.hist(time_count[5], bins=100)