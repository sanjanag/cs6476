#!/usr/bin/env python
# coding: utf-8

# In[59]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# In[37]:


output_imgs = []
output_arrays = []


# In[38]:


im = Image.open("inputPS0Q2.jpg")
imarray = np.asarray(im)


# In[39]:


r = imarray[:,:,0].copy()
g = imarray[:,:,1].copy()
grb_imarray = imarray.copy()
grb_imarray[:,:,0] = g
grb_imarray[:,:,1] = r
grb_im = Image.fromarray(grb_imarray)
output_imgs.append(grb_im)
output_arrays.append(grb_imarray)


# In[40]:


grayscale = im.copy().convert('L')
output_imgs.append(grayscale)
gray_array = np.asarray(grayscale)
output_arrays.append(gray_array)


# In[41]:


neg_imarray = 255 - gray_array.copy()
neg_im = Image.fromarray(neg_imarray)
output_imgs.append(neg_im)
output_arrays.append(neg_imarray)


# In[42]:


mirror_array = gray_array.copy()[:,::-1]
mirror_im = Image.fromarray(mirror_array)
output_imgs.append(mirror_im)
output_arrays.append(mirror_array)


# In[43]:


avg_array = (gray_array+mirror_array)/2
avg_array = avg_array.astype('uint8')
avg_im = Image.fromarray(avg_array)
output_imgs.append(avg_im)
output_arrays.append(avg_array)


# In[44]:


# mean = 0
# var = 0.1
# sigma = var**0.5
# gauss = np.random.normal(mean, sigma, gray_array.shape)
# gauss = np.clip(gauss, a_min = 0, a_max = 1) * 255
# gauss = gauss.astype('uint8')
# np.save("noise.npy", gauss)
gauss = np.load("noise.npy")
noisy_array = gray_array + gauss
noisy_array = np.clip(noisy_array, a_min = 0, a_max = 255) 
noisy_im = Image.fromarray(noisy_array)
output_imgs.append(noisy_im)
output_arrays.append(noisy_array)


# In[45]:


fig = plt.figure(figsize=(10,10))
# fig.suptitle('Programming Question')
plt.subplot(321)
plt.axis('off')
plt.title('Red and Green swapped')
plt.imshow(grb_im)

plt.subplot(322)
plt.axis('off')
plt.title('Grayscale')
plt.imshow(grayscale, cmap='gray')

plt.subplot(323)
plt.axis('off')
plt.title('Negative image')
plt.imshow(neg_im, cmap='gray')

plt.subplot(324)
plt.axis('off')
plt.title('Mirror image')
plt.imshow(mirror_im, cmap='gray')

plt.subplot(325)
plt.axis('off')
plt.title('Mirror and Grayscale average image')
plt.imshow(avg_im, cmap='gray')

plt.subplot(326)
plt.axis('off')
plt.title('Image with added noise')
plt.imshow(noisy_im, cmap='gray')

plt.subplots_adjust(wspace=.05, hspace=.2)

plt.show()


# In[46]:


grb_im.save("swapImgPS0Q2.png", format='PNG')
grayscale.save("grayImgPS0Q2.png", format='PNG')
neg_im.save("negativeImgPS0Q2.png", format="PNG")
mirror_im.save("mirrorImgPS0Q2.png", format="PNG")
avg_im.save("avgImgPS0Q2.png", format="PNG")
noisy_im.save("addNoiseImgPS0Q2.png", format="PNG")

fig.savefig("q2.png", format="PNG")

