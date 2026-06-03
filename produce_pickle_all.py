import sys
import numpy as np

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

key = "/mnt/raarchive/chabanov/direct_Urca_runs/dU_10_15_pwl_0.75_HR"
key_print = "dU_10_15_pwl_0.75_HR"

sd = SimDir(key)

# ----------------------------------------------------------------------

# Get iterations by loading only on rank 0

# Note that this operation is blocking

if (rank == 0):

    # Note that the iterations for 2D and 3D data might differ

    Hdf5  = cgr.GridH5Dir(sd)
    Ascii = cgra.GridASCIIDir(sd)
    rdr   = [Hdf5, Ascii]

    itera_list  = pc.cactus_grid_omni.GridOmniReader((0,1),rdr).get_iters("rho_b")
    timing_list = pc.cactus_grid_omni.GridOmniReader((0,1),rdr).get_times("rho_b")

elif (rank != 0):

    itera_list = None
    timing_list = None

# Broadcast iterations to all processes

itera_list  = comm.bcast(itera_list,root=0)
timing_list = comm.bcast(timing_list,root=0) 

# ----------------------------------------------------------------------

if(rank==0):
	print("Checkpoint Start")

# ----------------------------------------------------------------------

# Grid setup for resampling

cells_x1 = 1000
cells_x2 = 1000

box_bound = 50.

min_x1 = -box_bound
min_x2 = -box_bound

max_x1 = box_bound
max_x2 = box_bound

g = gd.RegGeom([cells_x1,cells_x2], [min_x1,min_x2], x1=[max_x1,max_x2])

# ----------------------------------------------------------------------

# List of variables to pickle

string_list = [ "rho_b", "pi", "bulkRelax", "bulkVisInvRelax",
                "bulkVis", "expansion_scalar", "kin_vorticity_spatial_xy",
                "temp", "vel[0]", "vel[1]", "betax", "betay", "alp",
                "w_lorentz", "P", "int_energy"]

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

# Re-organize data to be loadable in parallel

# Fast changing index belongs to it_list
# That's important as we want processes to load a low number
# of variables

NN = len(string_list) * nnn

one_d_data_list = np.linspace(0,NN,NN,endpoint=False,dtype=int)

# Helper functions

def get_string(one_d_index,string_list,it_list):
    return string_list[int(one_d_index/len(it_list))]

def get_it(one_d_index,string_list,it_list):
    return it_list[one_d_index-len(it_list)*int(one_d_index/len(it_list))]

# ----------------------------------------------------------------------

def load(sd,string_list,g,it_list,index):

    this_string = get_string(index,string_list,it_list)
    this_it     = get_it(index,string_list,it_list)

    tmp = sd.grid.xy.read(this_string,this_it, geom=g,adjust_spacing=0,order=1)

    data_dict_return = {this_string:{str(this_it):tmp}}

    return data_dict_return

# ----------------------------------------------------------------------

loop_start = 0
loop_end   = NN
loop_cycles = loop_end - loop_start
chunks = int( loop_cycles/size )

if(chunks==0):
	print("Too many processes for given data!")
	sys.exit()

data_dict = {}
for elem in string_list:
	data_dict.update({elem:{}})

if (rank < size-1):
	
	for i in range(rank*chunks,(rank+1)*chunks,1):

		if(rank==0):
			print(i)

		tmp_data = load(sd,string_list,g,it_list,i)

		for elem in tmp_data:
			data_dict[elem].update(tmp_data[elem])

elif (rank == size-1):

	for i in range(rank*chunks,loop_cycles,1):

		tmp_data = load(sd,string_list,g,it_list,i)

		for elem in tmp_data:

			data_dict[elem].update(tmp_data[elem])

if(rank==0):
	print("Checkpoint Load")

# ----------------------------------------------------------------------

# Send all dictionairies to rank 0

# Note that this operation is blocking

if (rank != 0):
    comm.send(data_dict, dest=0, tag=rank)
elif (rank == 0):
    all_data_list = []
    for i in range(1,size):
    	all_data_list.append(comm.recv(source=i,tag=i))


if(rank==0):
	print("Checkpoint Send")

# ----------------------------------------------------------------------

# Construct final dictinairies and pickle on rank 0

if (rank==0):

	all_data_dict = {}
	for elem in string_list:
		all_data_dict.update({elem:{}})

	# Rank 0 dictionairy

	for elem in data_dict:
		all_data_dict[elem].update(data_dict[elem])

	# Other ranks

	for i in range(1,size):

		rank_dict = all_data_list[i-1]

		for elem in rank_dict:
			all_data_dict[elem].update(rank_dict[elem])
	
	# Each variable gets its own file

	for elem in all_data_dict:

		with open(elem+"--"+key_print+'.pickle', 'wb') as f:
    			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(all_data_dict[elem],f,pickle.HIGHEST_PROTOCOL)

		f.close()

if(rank==0):
	print("Checkpoint End")

# ----------------------------------------------------------------------

MPI.Finalize()
