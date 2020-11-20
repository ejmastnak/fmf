import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

filename = "0.html"
data = pd.read_html(filename, skiprows=0)[0]
np_data = data.to_numpy()
coincidences = np.sum(np_data[:,1])
print(coincidences)
