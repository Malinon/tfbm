# SHOWCASE TO PRESENT HOW TFBM PACKAGE CAN BE USED IN FLUID FLOW SIMULATIONS.
# SOLVES NAVIER-STOKES EQUATIONS IN A MODEL FRACTURE WHOSE GEOMETRY IS OBTAINED
# FROM THE TFBM

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import pystencils as ps
from lbmpy.session import *

# adjust the path to import TFB files
# sys.path.append("/home/barman/tfbm")
sys.path.append("/home/dstrzelczyk/tfbm")

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

os.environ["OMP_NUM_THREADS"] = "8"

H = 0.5
Lam = 20.
L = 20.
N = 800
lattice_sites_per_tfbm_step = 4
h = L / N

N_SAMPLES = 1
save_path = "USECASES/fracture_flow/res/H"+str(H)+"_Lam"+str(Lam)+"/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i_SAMPLE in range(N_SAMPLES):

    # --- clear the results file for the current sample
    file_clear = open(save_path+"SAMPLE_"+str(i_SAMPLE)+".dat", "w")
    file_clear.close()

    # --- define the tfbm object
    tfbm = TFBM1(1.,int(N/lattice_sites_per_tfbm_step),H,Lam)

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

    # create the lattice
    x_vals = np.arange(0, L, h)
    y_vals = np.arange(-h, np.max(y_top_wall_tfbm)+h, h)
    x, y = np.meshgrid(x_vals, y_vals, indexing='ij')
    wall = np.zeros(x.shape)
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            wall[i,j] = is_wall(i*h,j*h,x_wall_tfbm,y_top_wall_tfbm,y_bottom_wall_tfbm)

    # plot the fracture
    # print("plotting the fracturte")
    # f,a = plt.subplots()
    # plt.plot(x_wall_tfbm,y_top_wall_tfbm)
    # plt.plot(x_wall_tfbm,y_bottom_wall_tfbm)
    # plt.scatter(x,y,c=wall,s=1)
    # a.set_aspect("equal")
    # plt.show()

    

    # ================= ACTUAL LBM SIMULATION =====================

    # 1. Simulation Parameters
    width, height = wall.shape
    print("Domain size is ",end=''); print(wall.shape)
    tau_p = 0.505
    relaxation_rate_e = 1./tau_p
    Lambda = 1/4
    tau_m = Lambda / (tau_p - 0.5) + 0.5
    relaxation_rate_o = 1./tau_m

    print("Running with:\ntau+ = "+str(tau_p)+"\ntau- = "+str(tau_m)+"\nLambda = "+str(Lambda))

    lbm_config = LBMConfig(stencil=Stencil.D2Q9, method=Method.TRT, relaxation_rates=[relaxation_rate_e, relaxation_rate_o])
    config = CreateKernelConfig(target=ps.Target.CPU,cpu_openmp=True,cpu_vectorize_info=None)

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

    def inflow_vel_callback(boundary_data, vx_in=.0, **_):
            boundary_data['vel_0'] = vx_in
            boundary_data['vel_1'] = 0

    outflow = ExtrapolationOutflow((0,1), sc.method)
    inflow = UBB(inflow_vel_callback, dim=sc.method.dim)
    
    sc.boundary_handling.set_boundary(inflow, make_slice[0, :])
    sc.boundary_handling.set_boundary(outflow, make_slice[-1, :])

    no_slip_obj = NoSlip()
    sc.boundary_handling.set_boundary(no_slip_obj, mask_callback=fracture_geometry_callback)

    # dummy timestep for the inflow BC to be successfully reinitialized for the first Re
    sc.run(1)
    
    # --- run sims for a range of Re
    Re_range = [pow(10.0,a) for a in np.arange(-3,4,0.125)]
    for Re in Re_range:

        velocity_inlet = Re * (tau_p-0.5)/3. / ( len(wall[0,:]) - sum(wall[0,:]) )

        sc.boundary_handling.trigger_reinitialization_of_boundary_data(vx_in=velocity_inlet)

        print("\n----> Running i_SAMPLE = "+str(i_SAMPLE)+" for Re = "+str(Re)+" (v_in = "+str(velocity_inlet)+")")
        
        q_prev = 1e10
        q = 1
        q_diff = abs((q_prev-q)/q)
        it = 0
        it_max = 2e6
        it_interval = int(1e4)
        # --- TIMESTEPPING FOR CURRENT RAYNOLDS NUMBER
        while q_diff > 1e-6 and it < it_max:
            sc.run(it_interval)
            it+=it_interval
            q_prev = q
            q = sum([sc.velocity[int(width/2), i, 0].compressed()[0] for i in range(len(sc.velocity[int(width/2), :, 0])) if len(sc.velocity[int(width/2), i, 0].compressed())==1])
            q_diff = abs((q_prev-q)/q)
            print("      it = "+str(it)+" | q_diff = "+str(q_diff))

        # -- calcualte the final measures
        vx_integral = .0
        vmag_integral = .0
        vx_minus_integral = .0
        n_total = .0
        for i in range(int(width*0.25), int(width*0.75)):
            for j in range(len(sc.velocity[i, :, 0])):
                if len(sc.velocity[i, j, 0].compressed())==1:
                    vx = sc.velocity[i, j, 0].compressed()[0]
                    vy = sc.velocity[i, j, 1].compressed()[0]

                    vx_integral += vx
                    vmag_integral += math.sqrt(vx*vx + vy*vy)
                    if vx < .0:
                        vx_minus_integral += 1
                    n_total += 1 

        T_omega = vmag_integral / vx_integral
        rho_minus = vx_minus_integral / n_total
        if it >= it_max:
            T_omega = -1
            rho_minus = -1

        # --- save the final measures
        file = open(save_path+"SAMPLE_"+str(i_SAMPLE)+".dat", "a")
        file.write(
            str(Re)+" "+\
            str(T_omega)+" "+\
            str(rho_minus)+"\n"
        )
        file.close()

        # --- save the macro fields to a file
        file = open(save_path+"SAMPLE_"+str(i_SAMPLE)+"_Re"+str(Re)+".dat", "w")
        for i in range(sc.velocity[:, :, :].shape[0]):
            for j in range(len(sc.velocity[i, :, 0])):
                if len(sc.velocity[i, j, 0].compressed())==1:
                    file.write(\
                    str(i) + " " +\
                    str(j) + " " +\
                    str(sc.velocity[i, j, 0].compressed()[0]) + " " +\
                    str(sc.velocity[i, j, 1].compressed()[0]) + " " +\
                    str(sc.density[i, j].compressed()[0]) + "\n"
                    )
        file.close()
        

        # --- plot the velocity field and save
        plt.figure(dpi=200,figsize=(10.5,1.5))
        plt.scalar_field(sc.velocity[:, :, 0])
        plt.xlim(-1,width+1)
        plt.ylim(-1,height+1)
        plt.colorbar()
        plt.savefig(save_path+"SAMPLE_"+str(i_SAMPLE)+"_Re"+str(Re)+".pdf")
        plt.close()