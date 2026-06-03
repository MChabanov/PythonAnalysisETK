import sys
import numpy as np
import time

import postcactus as pc
from postcactus.simdir import SimDir
from postcactus import visualize as viz
from postcactus import grid_data as gd
from postcactus.fourier_util import *

from postcactus import cactus_grid_h5 as cgr
from postcactus import cactus_grid_ascii as cgra

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

# Set simulation directories and obtain iterations

key = "/mnt/raarchive/chabanov/direct_Urca_runs/dU_10_15_linear_HR"
key_print = "dU_10_15_linear_HR"

sd = SimDir(key)

# ----------------------------------------------------------------------

# List of variables to pickle

string_list = [ "rho_b", "pi", "bulkRelax", "bulkVisInvRelax",
                "bulkVis", "expansion_scalar", "kin_vorticity_spatial_xy",
                "temp", "vel[0]", "vel[1]", "betax", "betay", "alp",
                "w_lorentz", "P", "int_energy"]

# ----------------------------------------------------------------------

# Organize data to be loadable in parallel

# Each process gets a variable

NN = len(string_list)

if (size != NN):
    #print("Use exactly "+str(NN)+" processes!")
    comm.Barrier()
    MPI.COMM_WORLD.Abort(1)

# Helper function

def get_string(index,string_list):
    return string_list[index]

# ----------------------------------------------------------------------

this_string = string_list[rank]

# ----------------------------------------------------------------------

# Get iterations

# Note that the iterations for 2D and 3D data might differ

Hdf5  = cgr.GridH5Dir(sd)
Ascii = cgra.GridASCIIDir(sd)
rdr   = [Hdf5, Ascii]

itera_list  = pc.cactus_grid_omni.GridOmniReader((0,1),rdr).get_iters(this_string)
timing_list = pc.cactus_grid_omni.GridOmniReader((0,1),rdr).get_times(this_string)

# ----------------------------------------------------------------------

if(rank==0):
	print("Checkpoint Start")

# ----------------------------------------------------------------------

# Grid setup for resampling

cells_x1 = 800
cells_x2 = 800

box_bound = 50.

min_x1 = -box_bound
min_x2 = -box_bound

max_x1 = box_bound
max_x2 = box_bound

g = gd.RegGeom([cells_x1,cells_x2], [min_x1,min_x2], x1=[max_x1,max_x2])

# ----------------------------------------------------------------------

# Which iterations do we want to pickle?

iii_plot_list = [] #[0,1,2,3,4,5,6,7,8,9]

for i in range(0,len(itera_list)):
	iii_plot_list.append(i)

time_plot_list = []

nnn = len(iii_plot_list)
                 
it_list = []

for iii in iii_plot_list:
    it_list.append(itera_list[iii])
    
for elem in iii_plot_list:

	if(rank==0):
		print(str(timing_list[elem]))
	
	time_plot_list.append(timing_list[elem])

# ----------------------------------------------------------------------

def load(sd,string,g,it_list,index):

    this_it     = it_list[index]

    tmp = sd.grid.xy.read(string,this_it,geom=g,adjust_spacing=0,order=1)

    data_dict_return = {str(this_it):tmp}

    return data_dict_return

# ----------------------------------------------------------------------

data_dict = {}
	
for i in range(0,nnn):

	if(rank==0):
		print(i)

	tmp_data = load(sd,this_string,g,it_list,i)

	data_dict.update(tmp_data)

if(rank==0):
	print("Checkpoint Load")

# ----------------------------------------------------------------------

# Each variable gets its own file

with open(this_string+"--"+key_print+'.pickle', 'wb') as f:
		# Pickle the 'data' dictionary using the highest protocol available.
	pickle.dump(data_dict,f,pickle.HIGHEST_PROTOCOL)

f.close()

if(rank==0):
	print("Checkpoint End")

# ----------------------------------------------------------------------

MPI.Finalize()
