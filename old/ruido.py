import numpy as np

import matplotlib.pyplot as plt

sample_x = np.random.normal(4, 0.1, 100)
sample_y = np.random.normal(4, 0.1, 100)

fig, ax = plt.subplots()
ax.plot(sample_x, sample_y, '.')
fig.show()

plt.show()