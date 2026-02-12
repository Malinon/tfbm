# SHOWCASE TO PRESENT HOW TFBM PACKAGE CAN BE USED IN FLUID FLOW SIMULATIONS.
# SOLVES NAVIER-STOKES EQUATIONS IN A MODEL FRACTURE WHOSE GEOMETRY IS OBTAINED
# FROM THE TFBM

import math
import matplotlib.pyplot as plt
import numpy as np
import sys

# adjust the path to import TFB files
sys.path.append("/home/barman/tfbm")

from tfbm import TFBM1, TFBM2, TFBM3

# function for checking if a given coordinate is inside the fracture
def is_wall(x,y,x_wall,y_top,y_bottom):
    for i in range(len(y_top)-1):

        if x<x_wall[i] or x>x_wall[i+1]:
            continue

        if abs(x-x_wall[i]) < 1e-10 and y>y_bottom[i] and y<y_top[i]:
            return 0
        else:
            if (y-y_top[i])/(x-x_wall[i]) > (y_top[i+1]-y_top[i]) / (x_wall[i+1]-x_wall[i]):
                return 1
            elif (y-y_bottom[i])/(x-x_wall[i]) < (y_bottom[i+1]-y_bottom[i]) / (x_wall[i+1]-x_wall[i]):
                return 1
            else:
                return 0

    return 1

L = 20.
N = 800
lattice_sites_per_tfbm_step = 4
h = L / N

tfbm = TFBM1(1.,int(N/lattice_sites_per_tfbm_step),.5,L)

# generate the top and the bottom walls
tfbm_samples = tfbm.generate_samples(2)
y_top_wall_tfbm = tfbm_samples[0]
y_bottom_wall_tfbm = tfbm_samples[1]

# translate the walls vertically to create a fracture
y_top_min = np.min(y_top_wall_tfbm)
y_bottom_max = np.max(y_bottom_wall_tfbm)
y_bottom_min = np.min(y_bottom_wall_tfbm) # this will be useful for setting up the lattice
min_aperture = 0.1
for i in range(len(y_bottom_wall_tfbm)):
    y_top_wall_tfbm[i] += min_aperture - y_top_min + y_bottom_max - y_bottom_min
    y_bottom_wall_tfbm[i] += -y_bottom_min

# create an array of x-coordinates
x_wall_tfbm = np.arange(0,L,h*lattice_sites_per_tfbm_step)
x_wall_tfbm=np.concatenate([x_wall_tfbm,np.array([L])])
print(x_wall_tfbm)

# create the lattice
x_vals = np.arange(0, L, h)
y_vals = np.arange(-h, np.max(y_top_wall_tfbm)+h, h)
x, y = np.meshgrid(x_vals, y_vals, indexing='ij')
wall = np.zeros(x.shape)
for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        wall[i,j] = is_wall(i*h,j*h,x_wall_tfbm,y_top_wall_tfbm,y_bottom_wall_tfbm)

# plot the fracture
# f,a = plt.subplots()
# plt.plot(x_wall_tfbm,y_top_wall_tfbm)
# plt.plot(x_wall_tfbm,y_bottom_wall_tfbm)
# plt.scatter(x,y,c=wall,s=1)
# a.set_aspect("equal")
# plt.show()



# ================= ACTUAL LBM SIMULATION =====================
import pystencils as ps
from lbmpy.session import *

# 1. Simulation Parameters
width, height = wall.shape
print("Domain size is ",end=''); print(wall.shape)
velocity_inlet = 1e-3
tau_p = 0.502
relaxation_rate_e = 1./tau_p
Lambda = 1/4
tau_m = Lambda / (tau_p - 0.5) + 0.5
relaxation_rate_o = 1./tau_m

print("Running with:\ntau+ = "+str(tau_p)+"\ntau- = "+str(tau_m)+"\nLambda = "+str(Lambda))

lbm_config = LBMConfig(stencil=Stencil.D2Q9, method=Method.TRT, relaxation_rates=[relaxation_rate_e, relaxation_rate_o])
config = CreateKernelConfig(target=Target.CPU)

sc = LatticeBoltzmannStep(domain_size=(width,height),
                           lbm_config=lbm_config,
                           config=config)


def fracture_geometry_callback(x,y):
    ret = np.ones((wall.shape[0]+2,wall.shape[1]+2), dtype=bool)
    for i in range(wall.shape[0]):
        for j in range(wall.shape[1]):
            if wall[i,j] == 0:
                ret[i+1,j+1] = False
    for j in range(ret.shape[1]):
        ret[0,j] = ret[1,j]
        ret[-1,j] = ret[-2,j]
    return ret

def inflow_vel_callback(boundary_data, activate=True, **_):
    boundary_data['vel_0'] = velocity_inlet
    boundary_data['vel_1'] = 0

inflow = UBB(inflow_vel_callback, dim=sc.method.dim)
outflow = ExtrapolationOutflow((0,1), sc.method)

sc.boundary_handling.set_boundary(inflow, make_slice[0, :])
sc.boundary_handling.set_boundary(outflow, make_slice[-1, :])

no_slip_obj = NoSlip()
sc.boundary_handling.set_boundary(no_slip_obj, mask_callback=fracture_geometry_callback)

sc.run(int(2e4))

plt.figure(dpi=200)
plt.scalar_field(sc.velocity[:, :, 0])
plt.xlim(-1,width+1)
plt.ylim(-1,height+1)
plt.colorbar()
plt.show()