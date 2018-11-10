import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

# Generate some test data
x = np.random.randint(0, 19, 100)
y = np.random.randint(0, 19, 100)
print(x, y)
img, extent = myplot(x, y, 8)

plt.imshow(img)
plt.show()