import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ads.csv')

n = 10000
d = 10
number_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0

for round in range(0, n):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if number_of_selections[i] > 0:
            avg_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(round + 1) / number_of_selections[i])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400  

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i

    ads_selected.append(ad)
    number_of_selections[ad] += 1
    reward = data.values[round, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

print(f'Total Reward: {total_reward}')
