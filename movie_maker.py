import sys
import numpy as np

import matplotlib as mpl

mpl.use('Agg')

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

# -----

class puncture_loc:
    def __init__(self,path,nout):
        self.path = path
        self.nout = nout

        for i in range(0,nout):

            out_num = str(i).zfill(4)

            # 0 iteration, 8 time, 12 data
            
            pt_loc_x_0 = np.loadtxt(self.path+'/output-'+out_num+'/Scalars/pt_loc_x[0]..asc')
            pt_loc_y_0 = np.loadtxt(self.path+'/output-'+out_num+'/Scalars/pt_loc_y[0]..asc')
            pt_loc_z_0 = np.loadtxt(self.path+'/output-'+out_num+'/Scalars/pt_loc_z[0]..asc')
            pt_loc_x_1 = np.loadtxt(self.path+'/output-'+out_num+'/Scalars/pt_loc_x[1]..asc')
            pt_loc_y_1 = np.loadtxt(self.path+'/output-'+out_num+'/Scalars/pt_loc_y[1]..asc')
            pt_loc_z_1 = np.loadtxt(self.path+'/output-'+out_num+'/Scalars/pt_loc_z[1]..asc')

            assert pt_loc_x_0.ndim == 2, "dimension should be two but is one, only one iteration available"

            its = len(pt_loc_x_0[:,0])

            punc_0 = np.zeros((its,5))
            punc_1 = np.zeros((its,5))
            
            punc_0[:,0] = pt_loc_x_0[:,0]
            punc_0[:,1] = pt_loc_x_0[:,8]
            punc_0[:,2] = pt_loc_x_0[:,12]
            punc_0[:,3] = pt_loc_y_0[:,12]
            punc_0[:,4] = pt_loc_z_0[:,12]

            punc_1[:,0] = pt_loc_x_1[:,0]
            punc_1[:,1] = pt_loc_x_1[:,8]
            punc_1[:,2] = pt_loc_x_1[:,12]
            punc_1[:,3] = pt_loc_y_1[:,12]
            punc_1[:,4] = pt_loc_z_1[:,12]

            if(i==0):
                self.punc_0 = punc_0
                self.punc_1 = punc_1
            else:
                self.punc_0 = np.append(self.punc_0,punc_0,axis=0)
                self.punc_1 = np.append(self.punc_1,punc_1,axis=0)

        self.punc_0 = np.unique(self.punc_0, axis=0)
        self.punc_1 = np.unique(self.punc_1, axis=0)

    def get_x_time(self,time):
        for i in range(0,len(self.punc_0[:,1])):
            if(self.punc_0[i,1]==time):
                return (self.punc_0[i,2],self.punc_1[i,2])

    def get_y_time(self,time):
        for i in range(0,len(self.punc_0[:,1])):
            if(self.punc_0[i,1]==time):
                return (self.punc_0[i,3],self.punc_1[i,3])

    def get_z_time(self,time):
        for i in range(0,len(self.punc_0[:,1])):
            if(self.punc_0[i,1]==time):
                return (self.punc_0[i,4],self.punc_1[i,4])

    def get_x_it(self,it):
        for i in range(0,len(self.punc_0[:,0])):
            if(self.punc_0[i,0]==it):
                return (self.punc_0[i,2],self.punc_1[i,2])

    def get_y_it(self,it):
        for i in range(0,len(self.punc_0[:,0])):
            if(self.punc_0[i,0]==it):
                return (self.punc_0[i,3],self.punc_1[i,3])

    def get_z_it(self,it):
        for i in range(0,len(self.punc_0[:,0])):
            if(self.punc_0[i,0]==it):
                return (self.punc_0[i,4],self.punc_1[i,4])

    def get_punc_0(self):
        return self.punc_0

    def get_punc_1(self):
        return self.punc_1

# -----

key_dir = "BBH_handoff_McLachlan_pm08_large_NoPsi6Threshold"
path = "/shared/rc/ccrg/le8016/"+key_dir+"/"

sd_dict = {"bbh" : SimDir(path)}

key_list = ["bbh"]

puncs = puncture_loc(path,1)

# -----

Hdf5_dict   = {}
Ascii_dict  = {}
rdr_dict    = {}
itera_dict  = {}
timing_dict = {}

for key in sd_dict:
    Hdf5_dict.update({key   : cgr.GridH5Dir(sd_dict[key])})
    Ascii_dict.update({key  : cgra.GridASCIIDir(sd_dict[key])})
    rdr_dict.update({key    : [Hdf5_dict[key], Ascii_dict[key]]})
    itera_dict.update({key  : pc.cactus_grid_omni.GridOmniReader((0,1), rdr_dict[key]).get_iters("rho_b")})
    timing_dict.update({key : pc.cactus_grid_omni.GridOmniReader((0,1), rdr_dict[key]).get_times("rho_b")})

# -----

factor = 1.61887093132742e-18
rhosat = 1.66e14 * factor

geom_to_gauss15 = 1.e+15/4.2447e-5

linthresh = 1.e-15
linscale = 1.

vmin = 5.
vmax = 5.e-1

#cont = [1.e13*factor,1.e14*factor,5.e14*factor,7.e14*factor]
cont = [0.2]

#cont_label = {1.e13*factor:"$\\mathbf{ 10^{13} \\ \\mathrm{\mathbf{g}} \\ \\mathrm{\\mathbf{cm}}^{-3} }$",
#              7.e14*factor:"$\\mathbf{ 7 \\times 10^{14} }$"}
#cont_label = {1.e13*factor:"$\\mathbf{ 10^{13} }$",
#              7.e14*factor:"$\\mathbf{ 7 \\times 10^{14} }$"}
#cont_label = {1.e13*factor:"$\\mathbf{ 10^{13} \\ \\mathrm{\mathbf{g}} \\ \\mathrm{\\mathbf{cm}}^{-3} }$",
#              7.e14*factor:"$\\mathbf{ 7 \\times 10^{14} \\ \\mathrm{\\mathbf{g}} \\ \\mathrm{\\mathbf{cm}}^{-3} }$"}

millis = 203.01930744592713
msun_to_km = 1.4766696910334391 

color_stream = "red"

plotting_styles = ["r-","g-","b-","k-","yo-","rx-","gx-","bx-","kx-","yx-","r--","g--","b--","k--","y--"]

#RdBu_r
#RdGy_r
#Greys
cm1     = viz.get_color_map("jet")

string_dens1 = "fuchsia"
clrs   = [string_dens1]

dens_stream0 = 2.0
dens_stream1 = 2.0
dens_stream2 = 2.0
dens_stream3 = 2.0

lw_0 = 2.0
lw_1 = 2.0
lw_2 = 2.0
lw_3 = 2.0
lw_4 = 2.0
lw_5 = 2.0
lw_6 = 2.0
lw_7 = 2.0

# -----

list_of_max_vars = ["rho_b","smallb2","w_lorentz"]
list_of_min_vars = ["alp"]

data_dict = {}

for key in sd_dict:
    
    tmp_dict = {}
    
    for i in range(0,len(list_of_max_vars)):
        
        time_s = sd_dict[key].ts.max[list_of_max_vars[i]]
        
        dummy_dict = {list_of_max_vars[i]+"_max" : time_s}
        
        tmp_dict.update(dummy_dict)
        
    for i in range(0,len(list_of_min_vars)):
        
        time_s = sd_dict[key].ts.min[list_of_min_vars[i]]
        
        dummy_dict = {list_of_min_vars[i]+"_min" : time_s}
        
        tmp_dict.update(dummy_dict)
    
    data_dict.update({key:tmp_dict})

def get_scalar_time(key,scalar,time):
    for i in range(0,len(data_dict[key][scalar].t)):
            if(data_dict[key][scalar].t[i]==time):
                return data_dict[key][scalar].y[i]

# -----

import pickle

pickle_path = './pickle_files/'

with open(pickle_path+'rho_b--'+key_dir+'.pickle','rb') as r:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    rho_data_dict = pickle.load(r)

with open(pickle_path+'alp--'+key_dir+'.pickle','rb') as r:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    alp_data_dict = pickle.load(r)

with open(pickle_path+'eps--'+key_dir+'.pickle','rb') as r:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    eps_data_dict = pickle.load(r)

with open(pickle_path+'radcool_gf--'+key_dir+'.pickle','rb') as r:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    cool_data_dict = pickle.load(r)

with open(pickle_path+'w_lorentz--'+key_dir+'.pickle','rb') as r:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    lorentz_data_dict = pickle.load(r)

with open(pickle_path+'smallb2--'+key_dir+'.pickle','rb') as r:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    smallb2_data_dict = pickle.load(r)

# -----

left  = 0.08     # the left side of the subplots of the figure
right = 0.92    # the right side of the subplots of the figure
bottom = 0.05   # the bottom of the subplots of the figure
top = 0.95      # the top of the subplots of the figure
wspace = 0.0    # the amount of width reserved for blank space between subplots
hspace = 0.1    # the amount of height reserved for white space between subplots

fontsize=10
plt.rcParams['font.size'] = str(fontsize)

iii = 0

for elem in rho_data_dict:

	this_time = rho_data_dict[elem].time
	this_it = str(elem)

	xmin = puncs.get_x_it(int(this_it))[0]-1.75
	xmax = puncs.get_x_it(int(this_it))[0]+1.75
	ymin = puncs.get_y_it(int(this_it))[0]-1.75
	ymax = puncs.get_y_it(int(this_it))[0]+1.75

	alpha_min = get_scalar_time('bbh','alp_min',this_time)

	fig,ax = plt.subplots(2,3,figsize=(15,10))
	fig.subplots_adjust(wspace=0.3)
	#cbar_ax = fig.add_axes([0.93, 0.02, 0.01, 0.96])


	for buba in ax:
    		for ggg in buba:
        		ggg.set_ylim(ymin,ymax)
        		ggg.set_xlim(xmin,xmax)
    
	plot0 = ax[0,0].pcolormesh(np.transpose(rho_data_dict[this_it].coords2d()[0]), 
                      np.transpose(rho_data_dict[this_it].coords2d()[1]),
                      np.transpose(rho_data_dict[this_it].data),
                   norm=colors.LogNorm(vmax=0.1,vmin=1e-7),
                   cmap="jet", shading='gouraud')
    
	ax[0,0].contour(np.transpose(alp_data_dict[this_it].coords2d()[0]), 
                      np.transpose(alp_data_dict[this_it].coords2d()[1]),
                      np.transpose(alp_data_dict[this_it].data),
                       [alpha_min], colors=clrs,
                       linewidths=lw_0);
    
	ax[0,0].set_title("$\\rho$")
    
	plot1 = ax[0,1].pcolormesh(np.transpose(eps_data_dict[this_it].coords2d()[0]), 
                      np.transpose(eps_data_dict[this_it].coords2d()[1]),
                      np.transpose(eps_data_dict[this_it].data),
                   norm=colors.LogNorm(vmax=1,vmin=1e-6),
                   cmap="magma", shading='gouraud')
    
	ax[0,1].contour(np.transpose(alp_data_dict[this_it].coords2d()[0]), 
                      np.transpose(alp_data_dict[this_it].coords2d()[1]),
                      np.transpose(alp_data_dict[this_it].data),
                       [alpha_min], colors=clrs,
                       linewidths=lw_0);
    
	ax[0,1].set_title("$\\epsilon$")
    
	plot2 = ax[1,0].pcolormesh(np.transpose(cool_data_dict[this_it].coords2d()[0]), 
                      np.transpose(cool_data_dict[this_it].coords2d()[1]),
                      np.transpose(cool_data_dict[this_it].data),
                   norm=colors.LogNorm(vmin=1e-4,vmax=1e2),
                   cmap="Blues", shading='gouraud')
    
	ax[1,0].contour(np.transpose(alp_data_dict[this_it].coords2d()[0]), 
                      np.transpose(alp_data_dict[this_it].coords2d()[1]),
                      np.transpose(alp_data_dict[this_it].data),
                       [alpha_min], colors=clrs,
                       linewidths=lw_0);
    
	ax[1,0].set_title("$L_c$")
    
	plot3 = ax[1,1].pcolormesh(np.transpose(lorentz_data_dict[this_it].coords2d()[0]), 
                      np.transpose(lorentz_data_dict[this_it].coords2d()[1]),
                      np.abs(np.transpose(lorentz_data_dict[this_it].data)-1.),
                   norm=colors.LogNorm(vmin=1e-2,vmax=9),
                   cmap="gray_r", shading='gouraud')
    
	ax[1,1].contour(np.transpose(alp_data_dict[this_it].coords2d()[0]), 
                      np.transpose(alp_data_dict[this_it].coords2d()[1]),
                      np.transpose(alp_data_dict[this_it].data),
                       [alpha_min], colors=clrs,
                       linewidths=lw_0);
    
	ax[1,1].set_title("$W$")
    
	plot4 = ax[0,2].pcolormesh(np.transpose(smallb2_data_dict[this_it].coords2d()[0]), 
                      np.transpose(smallb2_data_dict[this_it].coords2d()[1]),
                      np.transpose(smallb2_data_dict[this_it].data),
                   norm=colors.LogNorm(vmax=1e-3,vmin=1e-8),
                   cmap="summer", shading='gouraud')
    
	ax[0,2].contour(np.transpose(alp_data_dict[this_it].coords2d()[0]), 
                      np.transpose(alp_data_dict[this_it].coords2d()[1]),
                      np.transpose(alp_data_dict[this_it].data),
                       [alpha_min], colors=clrs,
                       linewidths=lw_0);
    
	ax[0,2].set_title("$b^2$")
    
	plot5 = ax[1,2].pcolormesh(np.transpose(rho_data_dict[this_it].coords2d()[0]), 
                      np.transpose(rho_data_dict[this_it].coords2d()[1]),
                      np.transpose(smallb2_data_dict[this_it].data/rho_data_dict[this_it].data),
                   norm=colors.LogNorm(vmax=100,vmin=1e-3),
                   cmap="summer", shading='gouraud')
    
	ax[1,2].contour(np.transpose(alp_data_dict[this_it].coords2d()[0]), 
                      np.transpose(alp_data_dict[this_it].coords2d()[1]),
                      np.transpose(alp_data_dict[this_it].data),
                       [alpha_min], colors=clrs,
                       linewidths=lw_0);
    
	ax[1,2].set_title("$b^2/\\rho$")
        
	divider_0 = make_axes_locatable(ax[0,0])
	divider_1 = make_axes_locatable(ax[0,1])
	divider_2 = make_axes_locatable(ax[1,0])
	divider_3 = make_axes_locatable(ax[1,1])
	divider_4 = make_axes_locatable(ax[0,2])
	divider_5 = make_axes_locatable(ax[1,2])
    
	caxx_0 = divider_0.append_axes("right", size="5%", pad=0.05)
	caxx_1 = divider_1.append_axes("right", size="5%", pad=0.05)
	caxx_2 = divider_2.append_axes("right", size="5%", pad=0.05)
	caxx_3 = divider_3.append_axes("right", size="5%", pad=0.05)
	caxx_4 = divider_4.append_axes("right", size="5%", pad=0.05)
	caxx_5 = divider_5.append_axes("right", size="5%", pad=0.05)

	cbar0 = plt.colorbar(plot0,cax=caxx_0)
	cbar1 = plt.colorbar(plot1,cax=caxx_1)
	cbar2 = plt.colorbar(plot2,cax=caxx_2)
	cbar3 = plt.colorbar(plot3,cax=caxx_3)
	cbar4 = plt.colorbar(plot4,cax=caxx_4)
	cbar5 = plt.colorbar(plot5,cax=caxx_5)
    
	fig.suptitle('t= '+str(round(this_time,3)))

	plt.savefig('./figures/'+key_dir+'--'+str(iii).zfill(4)+".png",bbox_inches="tight")

	fig.clf()
	plt.close()

	iii+=1
