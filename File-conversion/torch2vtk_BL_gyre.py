import torch
import numpy as np
#import foamFileOperation
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time
import math
import vtk
from vtk.util import numpy_support as VN

#plot the loss on CPU (first load the net)



def create_vtk(x,y,y_inner):


	x = torch.Tensor(x).to(device)
	y = torch.Tensor(y).to(device)
	y_inner = torch.Tensor(y_inner).to(device)
	h_n = 128 #for u,v,p
	input_n = 2 # this is what our answer is a function of. In the original example 3 : x,y,scale 



	class Swish(nn.Module):
		def __init__(self, inplace=True):
			super(Swish, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			if self.inplace:
				x.mul_(torch.sigmoid(x))
				return x
			else:
				return x * torch.sigmoid(x)

	class MySquared(nn.Module):
		def __init__(self, inplace=True):
			super(MySquared, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			return torch.square(x)

	

	


	class Net2_inner(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_inner, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				Swish(),
				nn.Linear(h_n,h_n),


				Swish(),
				nn.Linear(h_n,h_n),


				Swish(),
				nn.Linear(h_n,h_n),

				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output

	class Net2_outer(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_outer, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				Swish(),
				nn.Linear(h_n,h_n),


				Swish(),
				nn.Linear(h_n,h_n),


				Swish(),
				nn.Linear(h_n,h_n),

				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output
	
	################################################################
	############################################################################
	#### Initiate the network used to represent concentration ##############
	#net1 = Net1().to(device)

	net2_inner = Net2_inner().to(device)
	net2_outer = Net2_outer().to(device)


	

	print('load the network')
	net2_outer.load_state_dict(torch.load(pytorch_file_outer,map_location=torch.device('cpu')))
	net2_inner.load_state_dict(torch.load(pytorch_file_inner,map_location=torch.device('cpu')))

	net2_outer.eval()
	net2_inner.eval()

			
	############### Convert network to VTK #################################
	print ('Loading', mesh_file)
	reader = vtk.vtkXMLUnstructuredGridReader()
	reader.SetFileName(mesh_file)
	reader.Update()
	data_vtk = reader.GetOutput()


	net_in = torch.cat((x.requires_grad_(),y.requires_grad_()),1)
	output_u = net2_outer(net_in)  #evaluate model
	output_u = output_u.data.numpy() 

	conc_outer = np.zeros((n_points,1)) #soln
	conc_outer[:,0] = output_u[:,0] * U_scale


	net_in = torch.cat((x.requires_grad_(),y_inner.requires_grad_()),1)
	output_u = net2_inner(net_in)  #evaluate model
	output_u = output_u.data.numpy() 

	conc_inner = np.zeros((n_points,1)) #soln
	conc_inner[:,0] = output_u[:,0] * U_scale
	


	#Save VTK
	theta_vtk = VN.numpy_to_vtk(conc_inner)
	theta_vtk.SetName('conc_inner_temp')   #concentration
	data_vtk.GetPointData().AddArray(theta_vtk)
	theta_vtk = VN.numpy_to_vtk(conc_outer)
	theta_vtk.SetName('conc_outer')   #concentration
	data_vtk.GetPointData().AddArray(theta_vtk)

	####### apply the BL asymptotic scaling ###########
	c = VN.vtk_to_numpy(data_vtk.GetPointData().GetArray('conc_inner_temp'))
	c_final= np.zeros((n_points))
	for i in range(n_points):
		pt  =  data_vtk.GetPoint(i)
		if ( pt[1] > inf_scale *math.sqrt(Diff) ): #outside of BL
			c_final[i] = 0. 
		else:
			c_final[i] = c[i]

	data_vtk.GetPointData().RemoveArray('conc_inner_temp')
	theta_vtk = VN.numpy_to_vtk(c_final)
	theta_vtk.SetName('conc_inner')   #concentration
	data_vtk.GetPointData().AddArray(theta_vtk)



	#### save ######################
	myoutput = vtk.vtkDataSetWriter()
	myoutput.SetInputData(data_vtk)
	myoutput.SetFileName(output_filename)
	myoutput.Write()


	print ('Done!' )


############## Set parameters here (make sure you are calling the appropriate network in the code. Network code needs to be compied here)

device = torch.device("cpu")


mesh_file = "/home/aa3878/Data/ML/Amir/near-wall/Results/concBL_fenics.vtu"
output_filename = "/home/aa3878/Data/ML/Amir/near-wall/Results/concGyre_PINNbl.vtk"
pytorch_file_outer = "/home/aa3878/Data/ML/Amir/near-wall/Results/pert_doublegre_outer.pt"
pytorch_file_inner = "/home/aa3878/Data/ML/Amir/near-wall/Results/pert_doublegre_inner.pt"

X_scale = 1. #2
Y_scale = 1. #1.
U_scale = 1. #soln scale

Diff = 0.0001 #0.0005
inf_scale =  8  #100. #used to set BC at infinity 






print ('Loading', mesh_file)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(mesh_file)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('n_points of the mesh:' ,n_points)
x_vtk_mesh = np.zeros((n_points,1))
y_vtk_mesh = np.zeros((n_points,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)

print ('Net input max x:', np.max(x_vtk_mesh))
print ('Net input max y:', np.max(y_vtk_mesh))

x  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) / X_scale
y  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1)) / Y_scale

y_inner = y / (inf_scale *math.sqrt(Diff) )

create_vtk(x,y,y_inner)




