import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('ACME-HappinessSurvey2020.csv')

attributes = data.drop('Y', axis=1).to_numpy()
labels = data['Y'].to_numpy()

happy = attributes[labels == 1, :]
unhappy = attributes[labels == 0, :]

# add some noise for visualization purposes
happy = happy + np.random.normal(loc=0.0, scale=0.1, size=happy.shape)
unhappy = unhappy + np.random.normal(loc=0.0, scale=0.1, size=unhappy.shape)

plt.figure()

for i in range(attributes.shape[1]):
  x = np.random.normal(loc=float(i+1), scale=0.1, size=happy.shape[0])
  plt.scatter(x, happy[:,i], c='g', marker='X', alpha=0.7)
  x = np.random.normal(loc=float(i+1), scale=0.1, size=unhappy.shape[0])
  plt.scatter(x, unhappy[:,i], c='r', marker='o', alpha=0.7)

plt.xlabel("Question #")
plt.ylabel("Answer")
plt.show()