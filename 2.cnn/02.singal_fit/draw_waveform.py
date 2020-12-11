import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_table("waveform.data", header=None)
print(data)
data.rename(columns={0: "label"}, inplace=True)

data.plot()
plt.show()
