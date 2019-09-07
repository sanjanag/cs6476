#!/usr/bin/env python
# coding: utf-8

# In[98]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# In[99]:


# im = Image.open("q1.jpg")
# im = im.resize((100,100)).convert('L')
# A = np.asarray(im)
# np.save("inputAPS0Q1.npy", A)
A = np.load("inputAPS0Q1.npy")

# In[102]:


x = np.sort(A.reshape(-1))[::-1]
intensity_values = np.sin(x)
fig, (ax0, ax1) = plt.subplots(
    nrows=2, gridspec_kw={'height_ratios':[7, 1],}, sharex=True)
ax0.plot(x)
ax0.set_title("Intensities")
ax1.imshow(np.atleast_2d(x), cmap='gray', extent=(0, 10000, 0, 300))
plt.yticks([])
plt.savefig("4a.png", format="PNG")
plt.show()


# In[104]:


plt.hist(A.reshape(-1), bins = 20)
plt.title("Histogram of intensities with 20 bins")
plt.savefig("4b.png", format="PNG")
plt.show()


# In[105]:


X = A[50:, :50]
plt.imshow(X, interpolation='none', cmap='gray')
plt.title("Bottom left quadrant")
plt.savefig("X.png", format="PNG")
np.save("outputXPS0Q1.npy", X)
plt.show()


# In[106]:


mean_A = int(np.mean(A))
Y = A.astype('int') - mean_A
Y = np.clip(Y, a_min = 0, a_max=255).astype('uint8')
plt.imshow(Y, interpolation='none', cmap='gray')
plt.title("Mean intensity subtracted")
plt.savefig("Y.png", format="PNG")
np.save("outputYPS0Q1.npy", Y)
plt.show()


# In[107]:


Z = np.zeros((100,100,3), dtype = 'uint8')
x,y = np.where(A > mean_A)
Z[x,y,0] = 255
plt.imshow(Z, interpolation='none')
plt.title("Red where Intensity greater than threshold")
plt.savefig("outputZPS0Q1.png", format="PNG")
plt.show()

