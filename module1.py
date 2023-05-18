import numpy as np
import pandas as pd
import torch

data = pd.read_csv('ACME-HappinessSurvey2020.csv')
labels = torch.tensor(data['Y'].values)
attributes = torch.tensor(data.drop('Y', axis=1).values)

print(attributes.shape)