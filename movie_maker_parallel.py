import sys
import os
import numpy as np
from datetime import datetime

import matplotlib as mpl

import matplotlib.pyplot as plt
import postcactus as pc
from postcactus.simdir import SimDir
from postcactus import visualize as viz
from postcactus import grid_data as gd
from postcactus.fourier_util import *

from postcactus import cactus_grid_h5 as cgr
from postcactus import cactus_grid_ascii as cgra

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

import matplotlib.colors as colors

import pickle

from mpi4py import MPI
from mpi4py.util import pkl5 # overcome 2GiB message count limit

# See also:
# https://mpi4py.readthedocs.io/en/stable/mpi4py.util.pkl5.html#
# https://github.com/mpi4py/mpi4py/issues/119

# ----------------------------------------------------------------------

comm = pkl5.Intracomm(MPI.COMM_WORLD)
size = comm.Get_size()
rank = comm.Get_rank()

# ----------------------------------------------------------------------

sd_dict = {"tnt" : SimDir("/mnt/raarchive/chabanov/direct_Urca_runs/rates_dU_10_15_tnt_HR/"),
           "low" : SimDir("/mnt/raarchive/chabanov/direct_Urca_runs/dU_10_15_linear_HR/"),
           "med" : SimDir("/mnt/raarchive/chabanov/direct_Urca_runs/dU_10_15_pwl_0.5_HR/"),
           "high" : SimDir("/mnt/raarchive/chabanov/direct_Urca_runs/dU_10_15_pwl_0.75_HR/")}

key_list = ["tnt","low","med","high"]

t_merge_dict_true = {'low': 4254.444444444444,
                     'high': 4255.555555555556,
                     'tnt': 4252.222222222223,
                     'med': 4254.444444444444}

# ----------------------------------------------------------------------

factor = 1.61887093132742e-18
rhosat = 2.7e14 * factor

millis = 203.01930744592713
msun_to_km = 1.4766696910334391 

# Note that I am using the neutron mass here
# That shouldn't be a problem
kB_over_mb = 8.617332478e-5 / 939.565e6
convert_temp = pow(kB_over_mb,-1.)*8.617332478e-11

my_vmax=1
my_vmin=1e-4

#RdBu_r
#RdGy_r
#Greys
cm1     = viz.get_color_map("PRGn")

string_dens1 = "green"

clrs   = ["black","black","black"]

dens_stream0 = 2.0

lw_0 = 1.3

rho_levels = [(1.-0.9975)*0.00073157182561,0.00073157182561,0.0013]

styler = ["dashed","solid","dotted"]

font_size = 15

conversion_visc = 2.7338395629592487e+33

direct_dict = {'low':  2.4024024024024024*0.0004029425588922455, 
              'high': 1.5529899348727056*0.0004029425588922455,
              'tnt':  8.206669603088548*0.0004029425588922455,
              'med':  1.857667584940312*0.0004029425588922455}

# ----------------------------------------------------------------------

cells_x1 = 800
cells_x2 = 800

box_bound = 50.

min_x1 = -box_bound
min_x2 = -box_bound

max_x1 = box_bound
max_x2 = box_bound

g = gd.RegGeom([cells_x1,cells_x2], [min_x1,min_x2], x1=[max_x1,max_x2])

# ----------------------------------------------------------------------

all_it = {}
all_times = {}
now = None

# Temporary data files
if(rank==0):
    now = datetime.now() 
comm.Barrier()
now = comm.bcast(now,root=0)

folder_path = './tmp_files_'+str(now)

tmp_file_string_root_rho = 'tmp_pickle_rho_'
tmp_file_string_root_bv = 'tmp_pickle_bv_'
tmp_file_string_root_bv2 = 'tmp_pickle_bv2_'

tmp_file_string_rho = tmp_file_string_root_rho+str(rank)+'.pickle'
tmp_file_string_bv = tmp_file_string_root_bv+str(rank)+'.pickle'
tmp_file_string_bv2 = tmp_file_string_root_bv2+str(rank)+'.pickle'

if(rank==0):

    path_dict_rho = {"tnt":"/mnt/rafast/chabanov/viscous_merger_dUHR_tntyst_analysis/2d-data/"+
                       "rates_dU_10_15_tnt_HR/rho_b--rates_dU_10_15_tnt_HR.pickle",
                "low":"/mnt/rafast/chabanov/viscous_merger_dUHR_tntyst_analysis/2d-data/"+
                       "dU_10_15_linear_HR/rho_b--dU_10_15_linear_HR.pickle",
                "med":"/mnt/rafast/chabanov/viscous_merger_dUHR_tntyst_analysis/2d-data/"+
                       "dU_10_15_pwl_0.5_HR/rho_b--dU_10_15_pwl_0.5_HR.pickle",
                "high":"/mnt/rafast/chabanov/viscous_merger_dUHR_tntyst_analysis/2d-data/"+
                       "dU_10_15_pwl_0.75_HR/rho_b--dU_10_15_pwl_0.75_HR.pickle"}

    path_dict_bv = {"tnt":"/mnt/rafast/chabanov/viscous_merger_dUHR_tntyst_analysis/2d-data/"+
                       "rates_dU_10_15_tnt_HR/pi--rates_dU_10_15_tnt_HR.pickle",
                "low":"/mnt/rafast/chabanov/viscous_merger_dUHR_tntyst_analysis/2d-data/"+
                       "dU_10_15_linear_HR/pi--dU_10_15_linear_HR.pickle",
                "med":"/mnt/rafast/chabanov/viscous_merger_dUHR_tntyst_analysis/2d-data/"+
                       "dU_10_15_pwl_0.5_HR/pi--dU_10_15_pwl_0.5_HR.pickle",
                "high":"/mnt/rafast/chabanov/viscous_merger_dUHR_tntyst_analysis/2d-data/"+
                      "dU_10_15_pwl_0.75_HR/pi--dU_10_15_pwl_0.75_HR.pickle"}

    path_dict_bv2 = {"tnt":"/mnt/rafast/chabanov/viscous_merger_dUHR_tntyst_analysis/2d-data/"+
                       "rates_dU_10_15_tnt_HR/P--rates_dU_10_15_tnt_HR.pickle",
                "low":"/mnt/rafast/chabanov/viscous_merger_dUHR_tntyst_analysis/2d-data/"+
                       "dU_10_15_linear_HR/P--dU_10_15_linear_HR.pickle",
                "med":"/mnt/rafast/chabanov/viscous_merger_dUHR_tntyst_analysis/2d-data/"+
                       "dU_10_15_pwl_0.5_HR/P--dU_10_15_pwl_0.5_HR.pickle",
                "high":"/mnt/rafast/chabanov/viscous_merger_dUHR_tntyst_analysis/2d-data/"+
                      "dU_10_15_pwl_0.75_HR/P--dU_10_15_pwl_0.75_HR.pickle"}

    pickle_dict_root_rho = {}
    pickle_dict_root_bv = {}
    pickle_dict_root_bv2 = {}

    for key in path_dict_rho:
    
        with open(path_dict_rho[key], 'rb') as r:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data_dict_root_rho = pickle.load(r)
            pickle_dict_root_rho.update({key:data_dict_root_rho})

        with open(path_dict_bv[key], 'rb') as g:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data_dict_root_bv = pickle.load(g)
            pickle_dict_root_bv.update({key:data_dict_root_bv})

        with open(path_dict_bv2[key], 'rb') as gg:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data_dict_root_bv2 = pickle.load(gg)
            pickle_dict_root_bv2.update({key:data_dict_root_bv2})

    # Create temporary folder and temporary pickle files

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Nested folders '{folder_path}' created.")
    else:
        print(f"Nested folders '{folder_path}' already exist.")

    # Get iterations and times

    for key in pickle_dict_root_rho:
        all_it.update({key:[]})

    for key in pickle_dict_root_rho:
        for elem in pickle_dict_root_rho[key]:
            all_it[key].append(int(elem))

    for key in all_it:
        all_it[key].sort()

    for key in pickle_dict_root_rho:
        all_times.update({key:[]})
        for i in range(0,len(all_it[key])):
            all_times[key].append(pickle_dict_root_rho[key][str(all_it[key][i])].time)

    #  Divide data in chunks

    NN = 1000000000000000
    for key in all_times:
        NN = min(len(all_times[key]),NN)

    loop_start = 0
    loop_end   = NN
    loop_cycles = loop_end - loop_start
    chunks = int( loop_cycles/size )

    if(chunks==0):
	    print("Too many processes for given data!")
	    sys.exit()

    for pp in range(0,size):

        tmp_dict_rho = {}
        for key in pickle_dict_root_rho:
            tmp_dict_rho.update({key:{}})

        tmp_dict_bv = {}
        for key in pickle_dict_root_bv:
            tmp_dict_bv.update({key:{}})

        tmp_dict_bv2 = {}
        for key in pickle_dict_root_bv2:
            tmp_dict_bv2.update({key:{}})

        if(pp<size-1):
            for i in range(pp*chunks,(pp+1)*chunks,1):

                for key in pickle_dict_root_rho:
                    iteration = str(all_it[key][i])
                    data_rho = pickle_dict_root_rho[key][iteration]
                    tmp_dict_rho[key].update({str(i):data_rho})

        if(pp==size-1):
            for i in range(pp*chunks,loop_cycles,1):
                for key in pickle_dict_root_rho:
                    iteration = str(all_it[key][i])
                    data_rho = pickle_dict_root_rho[key][iteration]
                    tmp_dict_rho[key].update({str(i):data_rho})

        if(pp<size-1):
            for i in range(pp*chunks,(pp+1)*chunks,1):

                for key in pickle_dict_root_bv:
                    iteration = str(all_it[key][i])
                    data_bv = pickle_dict_root_bv[key][iteration]
                    tmp_dict_bv[key].update({str(i):data_bv})

        if(pp==size-1):
            for i in range(pp*chunks,loop_cycles,1):
                for key in pickle_dict_root_bv:
                    iteration = str(all_it[key][i])
                    data_bv = pickle_dict_root_bv[key][iteration]
                    tmp_dict_bv[key].update({str(i):data_bv})

        if(pp<size-1):
            for i in range(pp*chunks,(pp+1)*chunks,1):

                for key in pickle_dict_root_bv2:
                    iteration = str(all_it[key][i])
                    data_bv2 = pickle_dict_root_bv2[key][iteration]
                    tmp_dict_bv2[key].update({str(i):data_bv2})

        if(pp==size-1):
            for i in range(pp*chunks,loop_cycles,1):
                for key in pickle_dict_root_bv2:
                    iteration = str(all_it[key][i])
                    data_bv2 = pickle_dict_root_bv2[key][iteration]
                    tmp_dict_bv2[key].update({str(i):data_bv2})

        with open(folder_path+'/'+tmp_file_string_root_rho+str(pp)+'.pickle', 'wb') as f:
    		# Pickle the 'data' dictionary using the highest protocol available.
	        pickle.dump(tmp_dict_rho,f,pickle.HIGHEST_PROTOCOL)

        f.close()

        with open(folder_path+'/'+tmp_file_string_root_bv+str(pp)+'.pickle', 'wb') as f:
    		# Pickle the 'data' dictionary using the highest protocol available.
	        pickle.dump(tmp_dict_bv,f,pickle.HIGHEST_PROTOCOL)

        f.close()

        with open(folder_path+'/'+tmp_file_string_root_bv2+str(pp)+'.pickle', 'wb') as f:
    		# Pickle the 'data' dictionary using the highest protocol available.
	        pickle.dump(tmp_dict_bv2,f,pickle.HIGHEST_PROTOCOL)

        f.close()

    # Close original dictionairy

    pickle_dict_root_rho.clear()
    r.close()
    del pickle_dict_root_rho

    pickle_dict_root_bv.clear()
    g.close()
    del pickle_dict_root_bv

    pickle_dict_root_bv2.clear()
    gg.close()
    del pickle_dict_root_bv2

# ----------------------------------------------------------------------

# Broadcast

comm.Barrier()
all_it    = comm.bcast(all_it,root=0)
all_times = comm.bcast(all_times,root=0)

# ----------------------------------------------------------------------

# Finally, each process reads its dedicated pickle file

with open(folder_path+'/'+tmp_file_string_rho, 'rb') as rr:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pickle_dict_rho = pickle.load(rr)

with open(folder_path+'/'+tmp_file_string_bv, 'rb') as gg:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pickle_dict_bv = pickle.load(gg)

with open(folder_path+'/'+tmp_file_string_bv2, 'rb') as ggg:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pickle_dict_bv2 = pickle.load(ggg)

# ----------------------------------------------------------------------

# Plot

# Note that we always plot the same iteration for all
# simulations

plt.style.use('default')

for elem0 in pickle_dict_rho[key_list[0]]:

    this_it = {}

    iii = int(elem0)

    for key in all_it:
        this_it.update({key:(elem0,str(-1))})

    fig,ax = plt.subplots(2,2,figsize=(14,11))

    left  = 0.12     # the left side of the subplots of the figure
    right = 0.95    # the right side of the subplots of the figure
    bottom = 0.085   # the bottom of the subplots of the figure
    top = 0.95      # the top of the subplots of the figure
    wspace = 0.25    # the amount of width reserved for blank space between subplots
    hspace = 0.2    # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    for elem in ax:
        for elem2 in elem:
            elem2.set_ylim(-8.*msun_to_km,8.*msun_to_km)
            elem2.set_xlim(-8.*msun_to_km,8.*msun_to_km)

    key = 'tnt'

    plot0 = ax[0,0].pcolormesh(np.transpose(pickle_dict_bv[key][this_it[key][0]].coords2d()[0])*msun_to_km,
                      np.transpose(pickle_dict_bv[key][this_it[key][0]].coords2d()[1])*msun_to_km,
                      np.transpose(pickle_dict_bv[key][this_it[key][0]].data/
                                   pickle_dict_bv2[key][this_it[key][0]].data),
                   norm=colors.SymLogNorm(vmin=-1e-2,vmax=1e-2,linthresh=1e-6,linscale=1.),
                   cmap=cm1, shading='nearest',rasterized=True)

    ax[0,0].contour(np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[0])*msun_to_km,
                np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[1])*msun_to_km,
                    np.transpose(pickle_dict_rho[key][this_it[key][0]].data),
                       rho_levels, colors=clrs,
                       linewidths=lw_0,linestyles=styler);

    ax[0,0].contour(np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[0])*msun_to_km,
                np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[1])*msun_to_km,
                    np.transpose(pickle_dict_rho[key][this_it[key][0]].data),
                       [direct_dict[key]], colors="fuchsia",
                       linewidths=lw_0,linestyles="solid");

    ax[0,0].set_title(key+"$;~t-t_{mer} = $"+str(round(
        (all_times[key][iii]-t_merge_dict_true[key])/millis,3)),
                  fontsize=font_size)

    key = 'low'

    plot1 = ax[0,1].pcolormesh(np.transpose(pickle_dict_bv[key][this_it[key][0]].coords2d()[0])*msun_to_km,
                      np.transpose(pickle_dict_bv[key][this_it[key][0]].coords2d()[1])*msun_to_km,
                      np.transpose(pickle_dict_bv[key][this_it[key][0]].data/
                                   pickle_dict_bv2[key][this_it[key][0]].data),
                   norm=colors.SymLogNorm(vmin=-my_vmax,vmax=my_vmax,linthresh=my_vmin,linscale=1.),
                   cmap=cm1, shading='nearest',rasterized=True)

    ax[0,1].contour(np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[0])*msun_to_km,
                np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[1])*msun_to_km,
                    np.transpose(pickle_dict_rho[key][this_it[key][0]].data),
                       rho_levels, colors=clrs,
                       linewidths=lw_0,linestyles=styler);

    ax[0,1].contour(np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[0])*msun_to_km,
                np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[1])*msun_to_km,
                    np.transpose(pickle_dict_rho[key][this_it[key][0]].data),
                       [direct_dict[key]], colors="fuchsia",
                       linewidths=lw_0,linestyles="solid");

    ax[0,1].set_title(key+"$;~t-t_{mer} = $"+str(round(
        (all_times[key][iii]-t_merge_dict_true[key])/millis,3)),
                  fontsize=font_size)

    key = 'med'

    plot2 = ax[1,0].pcolormesh(np.transpose(pickle_dict_bv[key][this_it[key][0]].coords2d()[0])*msun_to_km,
                      np.transpose(pickle_dict_bv[key][this_it[key][0]].coords2d()[1])*msun_to_km,
                      np.transpose(pickle_dict_bv[key][this_it[key][0]].data/
                                   pickle_dict_bv2[key][this_it[key][0]].data),
                   norm=colors.SymLogNorm(vmin=-my_vmax,vmax=my_vmax,linthresh=my_vmin,linscale=1.),
                   cmap=cm1, shading='nearest',rasterized=True)

    ax[1,0].contour(np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[0])*msun_to_km,
                np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[1])*msun_to_km,
                    np.transpose(pickle_dict_rho[key][this_it[key][0]].data),
                       rho_levels, colors=clrs,
                       linewidths=lw_0,linestyles=styler);

    ax[1,0].contour(np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[0])*msun_to_km, 
                np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[1])*msun_to_km,
                    np.transpose(pickle_dict_rho[key][this_it[key][0]].data),
                       [direct_dict[key]], colors="fuchsia",
                       linewidths=lw_0,linestyles="solid");

    ax[1,0].set_title(key+"$;~t-t_{mer} = $"+str(round(
        (all_times[key][iii]-t_merge_dict_true[key])/millis,3)),
                  fontsize=font_size)

    key = 'high'

    plot3 = ax[1,1].pcolormesh(np.transpose(pickle_dict_bv[key][this_it[key][0]].coords2d()[0])*msun_to_km,
                      np.transpose(pickle_dict_bv[key][this_it[key][0]].coords2d()[1])*msun_to_km,
                      np.transpose(pickle_dict_bv[key][this_it[key][0]].data/
                                   pickle_dict_bv2[key][this_it[key][0]].data),
                   norm=colors.SymLogNorm(vmin=-my_vmax,vmax=my_vmax,linthresh=my_vmin,linscale=1.),
                   cmap=cm1, shading='nearest',rasterized=True)

    ax[1,1].contour(np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[0])*msun_to_km,
                np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[1])*msun_to_km,
                    np.transpose(pickle_dict_rho[key][this_it[key][0]].data),
                       rho_levels, colors=clrs,
                       linewidths=lw_0,linestyles=styler);

    ax[1,1].contour(np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[0])*msun_to_km, 
                np.transpose(pickle_dict_rho[key][this_it[key][0]].coords2d()[1])*msun_to_km,
                    np.transpose(pickle_dict_rho[key][this_it[key][0]].data),
                       [direct_dict[key]], colors="fuchsia",
                       linewidths=lw_0,linestyles="solid");

    ax[1,1].set_title(key+"$;~t-t_{mer} = $"+str(round(
        (all_times[key][iii]-t_merge_dict_true[key])/millis,3)),
                  fontsize=font_size)


    ax[1,0].set_xlabel("$x~[km]$",fontsize=font_size)
    ax[1,1].set_xlabel("$x~[km]$",fontsize=font_size)
    ax[0,0].set_ylabel("$y~[km]$",fontsize=font_size)
    ax[1,0].set_ylabel("$y~[km]$",fontsize=font_size)

    for elem in ax:
        for elem2 in elem:
            elem2.xaxis.set_tick_params(labelsize=font_size)
            elem2.yaxis.set_tick_params(labelsize=font_size)

    divider0 = make_axes_locatable(ax[0,0])
    divider1 = make_axes_locatable(ax[0,1])
    divider2 = make_axes_locatable(ax[1,0])
    divider3 = make_axes_locatable(ax[1,1])

    caxx0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = plt.colorbar(plot0,cax=caxx0)
    #cbar0.ax.set_ylabel('$\\rho$',rotation=0,labelpad=20,fontsize=font_size)
    cbar0.ax.tick_params(labelsize=font_size)

    caxx1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(plot1,cax=caxx1)
    #cbar1.ax.set_ylabel('$\\rho$',rotation=0,labelpad=20,fontsize=font_size)
    cbar1.ax.tick_params(labelsize=font_size)

    caxx2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(plot2,cax=caxx2)
    #cbar2.ax.set_ylabel('$\\rho$',rotation=0,labelpad=20,fontsize=font_size)
    cbar2.ax.tick_params(labelsize=font_size)

    caxx3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(plot3,cax=caxx3)
    #cbar3.ax.set_ylabel('$\\rho$',rotation=0,labelpad=20,fontsize=font_size)
    cbar3.ax.tick_params(labelsize=font_size)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    fig.suptitle("$\\Pi/p$",
             fontsize=font_size,y=0.99)

    plt.savefig("./dU_pi_over_pi_0.174M_"+str(iii).zfill(4)+".png",
                bbox_inches="tight",dpi=200)

    plt.close('all')
