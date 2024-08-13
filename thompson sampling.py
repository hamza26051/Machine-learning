import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random

data=pd.read_csv('ads.csv')

ads=[]
d=10
n=10000
total_reward=0
numberofselection0=[0]*d
numberofselection1=[0]*d
for n in range(0,10000):
  ad=0
  maxrandom=0
  for i in range(0,10):
    beta=random.betavariate(numberofselection0[i]+1,numberofselection1[i]+1)
    if beta>maxrandom:
      maxrandom=beta
      ad=i
    ads.append(ad)
    reward=data.values(n,ad)
    if reward==1:
      numberofselection1[ad]=numberofselection1[ad]+1
    else:
      numberofselection0[ad]=numberofselection0[ad]+1
    total_reward=total_reward+reward

plt.hist(ads)
plt.title('Histogram of ads selections (Thompson Sampling)')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

print(f'Total Reward: {total_reward}')