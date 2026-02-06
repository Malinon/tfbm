# SHOWCASE TO PRESENT HOW TFBM PACKAGE CAN BE USED IN FLUID FLOW SIMULATIONS.
# SOLVES NAVIER-STOKES EQUATIONS IN A MODEL FRACTURE WHOSE GEOMETRY IS OBTAINED
# FROM THE TFBM

import matplotlib.pyplot as plt
import numpy as np
import sys
# adjust the path to import TFB files
sys.path.append("/home/barman/tfbm")

from tfbm import TFBM1, TFBM2, TFBM3

# function for checking if a given coordinate is inside the fracture
def in_fracture(x,y,x_wall,y_top,y_bottom):
    for i in range(len(y_top)-1):
        
        if x<=x_wall[i] or x>x_wall[i+1]:
            continue

        if (y-y_top[i])/(x-x_wall[i]) > (y_top[i+1]-y_top[i]) / (x_wall[i+1]-x_wall[i]):
            return 0
        elif (y-y_bottom[i])/(x-x_wall[i]) < (y_bottom[i+1]-y_bottom[i]) / (x_wall[i+1]-x_wall[i]):
            return 0
        else:
            return 1

    return 0

L = 20.
N = 200
h = L / (N*2)

tfbm = TFBM1(1.,N,.5,L)

# generate the top and the bottom walls
tfbm_samples = tfbm.generate_samples(2)
y_top_wall = tfbm_samples[0]
y_bottom_wall = tfbm_samples[1]

# translate the walls vertically to create a fracture
y_top_min = np.min(y_top_wall)
y_bottom_max = np.max(y_bottom_wall)
y_bottom_min = np.min(y_bottom_wall) # this will be useful for setting up the lattice
min_aperture = 0.1
for i in range(len(y_bottom_wall)):
    y_top_wall[i] += min_aperture - y_top_min + y_bottom_max - y_bottom_min
    y_bottom_wall[i] += -y_bottom_min

# create an array of x-coordinates
x_wall = np.arange(0,L,L/N)
x_wall=np.concatenate([x_wall,np.array([L])])

# create the lattice
x_vals = np.arange(0, L, h)
y_vals = np.arange(-h, np.max(y_top_wall)+h, h)
x, y = np.meshgrid(x_vals, y_vals, indexing='ij')
wall = np.zeros(x.shape)
for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        wall[i,j] = in_fracture(i*h,j*h,x_wall,y_top_wall,y_bottom_wall)

# plot the fracture
f,a = plt.subplots()
plt.plot(x_wall,y_top_wall)
plt.plot(x_wall,y_bottom_wall)
plt.scatter(x,y,c=wall,s=1)
a.set_aspect("equal")
plt.show()