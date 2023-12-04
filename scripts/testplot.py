
#pip install pyqt5
# sudo apt-get install libqt5gui5
#dpkg -l | grep libqt5  
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

fruits = ['apple', 'blueberry', 'cherry', 'orange']
counts = [40, 100, 30, 55]
bar_labels = ['red', 'blue', '_red', 'orange']
bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('fruit supply')
ax.set_title('Fruit supply by kind and color')
ax.legend(title='Fruit color')

plt.show()



#import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import LogNorm 
      
dx, dy = 0.015, 0.05
y, x = np.mgrid[slice(-4, 4 + dy, dy), 
                slice(-4, 4 + dx, dx)] 
z = (1 - x / 3. + x ** 5 + y ** 5) * np.exp(-x ** 2 - y ** 2) 
z = z[:-1, :-1] 
z_min, z_max = -np.abs(z).max(), np.abs(z).max() 
  
c = plt.imshow(z, cmap ='Greens', vmin = z_min, vmax = z_max, 
                 extent =[x.min(), x.max(), y.min(), y.max()], 
                    interpolation ='nearest', origin ='lower') 
plt.colorbar(c) 
  
plt.title('matplotlib.pyplot.imshow() function Example',  
                                     fontweight ="bold") 
plt.show() 


# Python code to read image
import cv2
 
# To read image from disk, we use
# cv2.imread function, in below method,
img = cv2.imread("sampledata/sjsuimag1.jpg", cv2.IMREAD_COLOR)
 
# Creating GUI window to display an image on screen
# first Parameter is windows title (should be in string format)
# Second Parameter is image array
cv2.imshow("image", img)
 
# To hold the window on screen, we use cv2.waitKey method
# Once it detected the close input, it will release the control
# To the next line
# First Parameter is for holding screen for specified milliseconds
# It should be positive integer. If 0 pass an parameter, then it will
# hold the screen until user close it.
cv2.waitKey(0)
 
# It is for removing/deleting created GUI window from screen
# and memory
cv2.destroyAllWindows()
