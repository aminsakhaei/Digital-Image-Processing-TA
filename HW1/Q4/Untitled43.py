#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# A
img = cv.imread('object-LE.bmp', 0)
upper = cv.imread('fullscale-LE.bmp', 0)
lower = cv.imread('background-LE.bmp', 0)

# B
upper_avg = np.mean(upper, axis=1,  keepdims=True)
lower_avg = np.mean(lower, axis=1,  keepdims=True)

# C
normalized_image = ((img - lower_avg) / (upper_avg - lower_avg)) * 255

# D
fix, ax = plt.subplots(1, 2, figsize=(10, 5))
ax = plt.subplot(1, 2, 1)
ax.imshow(img, cmap='gray', vmin = 0, vmax = 255)
ax.set_title('Original')
ax.axis('off')

ax = plt.subplot(1, 2, 2)
ax.imshow(normalized_image, cmap = 'gray', vmin = 0, vmax = 255)
ax.set_title('Normalized')
ax.axis('off')

plt.show()

