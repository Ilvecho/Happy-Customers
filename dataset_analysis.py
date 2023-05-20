import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- LOAD AND SEPARATION ----------
data = pd.read_csv('ACME-HappinessSurvey2020.csv')

attributes = data.drop('Y', axis=1).to_numpy()
labels = data['Y'].to_numpy()

# Separate the attributes according to the label
happy = attributes[labels == 1, :]
unhappy = attributes[labels == 0, :]


## ---------- SCATTER PLOT -----------
## add some noise for visualization purposes
#happy = happy + np.random.normal(loc=0.0, scale=0.1, size=happy.shape)
#unhappy = unhappy + np.random.normal(loc=0.0, scale=0.1, size=unhappy.shape)

#plt.figure()

#for i in range(attributes.shape[1]):
#  x = np.random.normal(loc=float(i+1), scale=0.1, size=happy.shape[0])
#  plt.scatter(x, happy[:,i], c='g', marker='X', alpha=0.7)
#  x = np.random.normal(loc=float(i+1), scale=0.1, size=unhappy.shape[0])
#  plt.scatter(x, unhappy[:,i], c='r', marker='o', alpha=0.7)

#plt.xlabel("Question #")
#plt.ylabel("Answer")
#plt.show()


#  ---------- HISTOGRAM -----------
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#yticks = np.arange(6)

#for ytick in yticks:
## ytick = 0
#    col_happy = happy[:, ytick]
#    col_unhappy = unhappy[:, ytick]

#    hist_happy, edges_happy = np.histogram(col_happy, bins = 5, range=(0,6))
#    hist_unhappy, edges_unhappy = np.histogram(col_unhappy, bins = 5, range=(0,6))

#    centers = np.array([1, 2, 3, 4, 5])
#    xwidth = 0.4 
#    happy_centers = centers - 0.2 * np.ones_like(centers)
#    unhappy_centers = centers + 0.2 * np.ones_like(centers)

#    ax.bar(happy_centers, height=hist_happy, width=0.4, zs=ytick, zdir="y", color='g')
#    ax.bar(unhappy_centers, height=hist_unhappy, width=0.4, zs=ytick, zdir="y", color='r')

#plt.show()


# -----------STATISTICAL ANALYSIS ----------
# OVERALL MEAN & VARIANCE ACROSS THE SAMPLES AND ALL THE QUESTIONS
full_mean_happy = np.mean(happy)                # 3.758 +/- 1.146
full_std_happy = np.std(happy)                  
full_mean_unhappy = np.mean(unhappy)            # 3.491 +/- 1.120
full_std_happy = np.std(unhappy)

# MEAN AND VARIANCE BY QUESTION ACROSS THE SAMPLES
ans_mean_happy = np.mean(happy, axis=0)         # [4.536 2.507 3.449 3.797 3.884 4.376]
ans_std_happy = np.std(happy, axis=0)           # [0.693 1.098 1.014 0.894 1.056 0.763]
ans_mean_unhappy = np.mean(unhappy, axis=0)     # [4.087 2.561 3.140 3.684 3.368 4.105]
ans_std_unhappy = np.std(unhappy, axis=0)       # [0.843 1.124 0.998 0.841 1.179 0.831]

## histogram
#centers = np.array(np.arange(6))
#xwidth = 0.4 
#happy_centers = centers - 0.2 * np.ones_like(centers)
#unhappy_centers = centers + 0.2 * np.ones_like(centers)

#fig = plt.figure()
#plt.bar(happy_centers, ans_mean_happy, width=xwidth, color='g', yerr=ans_std_happy)
#plt.bar(unhappy_centers, ans_mean_unhappy, width=xwidth, color='r', yerr=ans_std_unhappy)
#plt.title('Histogram of the average by question across all the samples')
#plt.show()

# MEAN BY SAMPLE ACROSS ALL THE QUESTIONS
unit_mean_happy = np.mean(happy, axis=1)         
unit_mean_unhappy = np.mean(unhappy, axis=1)

## histogram
#hist_unit_mean_happy, edges_unit_mean_happy = np.histogram(unit_mean_happy, bins = 25, range=(0,5))
#hist_unit_mean_unhappy, edges_unit_mean_unhappy = np.histogram(unit_mean_unhappy, bins = 25, range=(0,5))

#center = np.convolve(edges_unit_mean_happy, np.ones(2), "valid")/2
#xwidth = np.diff(edges_unit_mean_happy)
#fig = plt.figure()
#plt.bar(center, height=hist_unit_mean_happy, width=xwidth, color='g', alpha=0.5)
#plt.bar(center, height=hist_unit_mean_unhappy, width=xwidth, color='r', alpha=0.5)
#plt.title('Histogram of the average by sample across all the questions')
#plt.show()


