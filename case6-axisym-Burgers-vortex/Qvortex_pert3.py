import torch
import numpy as np
#import foamFileOperation
#from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import math
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time

#use Eqn 6.34 from Vorticity and Vortex Dynamics book (Q-vortex, Burgers vortex)

def geo_train(device,x_in,y_in,xb,yb,xb2,yb2,cb,xb_N_up,yb_N_up, xb_N_down,yb_N_down, batchsize,learning_rate,epochs,path,Flag_batch,Diff,Flag_BC_exact,Lambda_BC  ):
	if (Flag_batch):
	 x = torch.Tensor(x_in).to(device)
	 y = torch.Tensor(y_in).to(device) 
	 xb = torch.Tensor(xb).to(device) 
	 yb = torch.Tensor(yb).to(device) 
	 xb2 = torch.Tensor(xb2).to(device) 
	 yb2 = torch.Tensor(yb2).to(device) 
	 cb = torch.Tensor(cb).to(device) 
	 xb_N_up = torch.Tensor(xb_N_up).to(device) 
	 yb_N_up = torch.Tensor(yb_N_up).to(device) 
	 xb_N_down = torch.Tensor(xb_N_down).to(device) 
	 yb_N_down = torch.Tensor(yb_N_down).to(device) 
	 #xb = torch.Tensor(xb).to(device) #These are different size as x and cannot go into the same TensorDataset
	 #yb = torch.Tensor(yb).to(device)
	 #cb = torch.Tensor(cb).to(device)  	
	 dataset = TensorDataset(x,y)
	 dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = True )
	else:
	 x = torch.Tensor(x_in).to(device)
	 y = torch.Tensor(y_in).to(device)   

	if(1):
		 x = x.type(torch.cuda.FloatTensor)
		 y = y.type(torch.cuda.FloatTensor)
		 xb = xb.type(torch.cuda.FloatTensor)
		 yb = yb.type(torch.cuda.FloatTensor)
		 xb2 = xb2.type(torch.cuda.FloatTensor)
		 yb2 = yb2.type(torch.cuda.FloatTensor)
		 cb = cb.type(torch.cuda.FloatTensor)
		 xb_N_up = xb_N_up.type(torch.cuda.FloatTensor)
		 yb_N_up = yb_N_up.type(torch.cuda.FloatTensor)
		 xb_N_down = xb_N_down.type(torch.cuda.FloatTensor)
		 yb_N_down = yb_N_down.type(torch.cuda.FloatTensor)

	h_nD = 30
	h_n = 128
	h_n2 = 128 #140 #180 #160 #outer
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
	def Velocity(r_input,z_input):
		gamma = 0.2
		Vel_theta = 1. / r_input * ( 1. - torch.exp(-r_input*r_input) )
		Vel_z =  gamma * z_input
		Vel_r =  -gamma * r_input / 2.
		return Vel_r,Vel_z, Vel_theta


	class Net1(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net1, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(2,h_nD),
				nn.Tanh(),
				nn.Linear(h_nD,h_nD),
				nn.Tanh(),
				nn.Linear(h_nD,h_nD),
				nn.Tanh(),

				nn.Linear(h_nD,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output
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
				nn.Linear(input_n,h_n2),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n2,h_n2),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n2,h_n2),

				Swish(),
				nn.Linear(h_n2,h_n2),

				Swish(),
				nn.Linear(h_n2,h_n2),


				Swish(),
				nn.Linear(h_n2,h_n2),


				Swish(),
				nn.Linear(h_n2,h_n2),


				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),

				nn.Linear(h_n2,1),
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
	
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	# use the modules apply function to recursively apply the initialization
	net2_inner.apply(init_normal)
	net2_outer.apply(init_normal)



	############################################################################

	optimizer2_inner = optim.Adam(net2_inner.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer2_outer = optim.Adam(net2_outer.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	#optimizer2 = optim.LBFGS(net2.parameters(), lr=learning_rate)


	############################################################################
	#### Define the governing equation ##############
	def criterion_inner(x,y):
 		#u: theta vel,  v: z vel  ,  x: r  , y:z 

		x.requires_grad = True
		y.requires_grad = True
		
		#net_in = torch.cat((x),1)
		net_in = torch.cat((x,y),1)
		C = net2_inner(net_in)
		C = C.view(len(C),-1)

		u,v, w = Velocity(x,y*inf_scale * math.sqrt(Diff) )  #u: r vel,  v: z vel  ,  x: r  , y:z  w: theta vel
		
		c_x = torch.autograd.grad(C,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		#c_xx = torch.autograd.grad(c_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		c_y = torch.autograd.grad(C,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		c_yy = torch.autograd.grad(c_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		
		#loss_1 = u *c_x - Diff * (c_xx + c_yy)
		loss_1 = u *c_x  + v/(inf_scale*math.sqrt(Diff))*c_y - c_yy / (inf_scale**2) 




		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1)) 

		return loss

	def criterion_outer(x,y):
		#u: theta vel,  v: z vel  ,  x: r  , y:z 

		x.requires_grad = True
		y.requires_grad = True
		
		#net_in = torch.cat((x),1)
		net_in = torch.cat((x,y*physical_height ),1)
		C = net2_outer(net_in)
		C = C.view(len(C),-1)

		u,v,w = Velocity(x,y*physical_height )
		
		c_x = torch.autograd.grad(C,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		c_xx = torch.autograd.grad(c_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		c_y = torch.autograd.grad(C,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		c_yy = torch.autograd.grad(c_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		
		loss_1 = u *c_x + v*c_y - Diff * (c_xx + c_x/x + c_yy)
		#loss_1 = u *c_x + v*c_y - 10.*Diff * (c_xx + c_yy)



		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1)) 

		return loss


	###################################################################
	############################################################################
	#### Define the boundary conditions ##############
	def Loss_BC(xb,yb,xb2,yb2,xb_N_up,yb_N_up ,xb_N_down,yb_N_down,cb):
		#xb = torch.FloatTensor(xb).to(device)
		#yb = torch.FloatTensor(yb).to(device)
		#cb = torch.FloatTensor(cb).to(device)

		#xb_Neumann_up = torch.FloatTensor(xb_Neumann_up).to(device)
		#yb_Neumann_up = torch.FloatTensor(yb_Neumann_up).to(device)
		#xb_Neumann_down = torch.FloatTensor(xb_Neumann_down).to(device)
		#yb_Neumann_down = torch.FloatTensor(yb_Neumann_down).to(device)

		xb_N_up.requires_grad = True
		yb_N_up.requires_grad = True
		xb_N_down.requires_grad = True
		yb_N_down.requires_grad = True

		xb2.requires_grad = True
		yb2.requires_grad = True

		net_in = torch.cat((xb, yb), 1)
		out_outer = net2_outer(net_in )
		out_inner = net2_inner(net_in )


		c_bc_outer = out_outer.view(len(out_outer), -1)
		c_bc_inner = out_inner.view(len(out_inner), -1)

		loss_f = nn.MSELoss()
		loss_bc_Dirichlet_left_right = loss_f(c_bc_outer, cb) + loss_f(c_bc_inner, cb)    #The Dirichlet BC (left and right)


		net_in = torch.cat((xb2, yb2), 1)
		out_n_r = net2_inner(net_in )
		out_inner_r= out_n_r.view(len(out_n_r), -1)
		c_r_down = torch.autograd.grad(out_inner_r,xb2,grad_outputs=torch.ones_like(xb2),create_graph = True,only_inputs=True)[0]
		loss_r = loss_f(c_r_down, torch.zeros_like(c_r_down))
		



		net_in_up = torch.cat((xb_N_up, yb_N_up), 1)
		net_in_down = torch.cat((xb_N_down, yb_N_down), 1)
		out_n = net2_inner(net_in_up )
		c_n_up = out_n.view(len(out_n), -1)
		out_n = net2_outer(net_in_down )
		c_n_down= out_n.view(len(out_n), -1)

		loss_bc_match = loss_f(c_n_up , c_n_down )  #Matching BC
		
		out_n = net2_inner(net_in_down )
		c_n_down2 = out_n.view(len(out_n), -1)
		#out_n = net2_outer(net_in_up )
		#c_n_up2 = out_n.view(len(out_n), -1)

		#c_y_up = torch.autograd.grad(c_n_up2,yb_N_up,grad_outputs=torch.ones_like(yb_N_up),create_graph = True,only_inputs=True)[0]
		c_y_down = torch.autograd.grad(c_n_down2,yb_N_down,grad_outputs=torch.ones_like(yb_N_down),create_graph = True,only_inputs=True)[0]

		#loss_bc_Neumann_up = loss_f(c_y_up, torch.zeros_like(c_y_up))  #The zero Neumann BC top
		loss_bc_Neumann_down = loss_f(c_y_down, flux_BC* math.sqrt(Diff)*inf_scale* torch.ones_like(c_y_down))  #The flux Neumann BC bottom

		loss_bc = loss_bc_Dirichlet_left_right + 60.*loss_bc_match + loss_bc_Neumann_down  + loss_r  #+ loss_bc_Neumann_up

		return loss_bc


	
	LOSS = []
	tic = time.time()


	if(Flag_pretrain):
		print('Reading (pretrain) functions first...')
		net2_inner.load_state_dict(torch.load(path+"test_new_smaller_inner" + ".pt"))
		net2_outer.load_state_dict(torch.load(path+"test_new_smaller_outer" + ".pt"))

	if (Flag_schedule):
		scheduler_inner = torch.optim.lr_scheduler.StepLR(optimizer2_inner, step_size=step_epoch, gamma=decay_rate)
		scheduler_outer = torch.optim.lr_scheduler.StepLR(optimizer2_outer, step_size=step_epoch, gamma=decay_rate)


	############################################################################
	####  Main loop##############
	if(Flag_batch):# This one uses dataloader
		for epoch in range(epochs):
			loss_eqn_tot = 0.
			loss_bc_tot = 0.
			n = 0
			for batch_idx, (x_in,y_in) in enumerate(dataloader):
				#zero gradient
				#net1.zero_grad()
				##Closure function for LBFGS loop:
				#def closure():
				net2_inner.zero_grad()
				net2_outer.zero_grad()
				loss_eqn = criterion_inner(x_in,y_in) + 10.*criterion_outer(x_in,y_in)
				loss_bc = Loss_BC(xb,yb,xb2,yb2,xb_N_up,yb_N_up ,xb_N_down,yb_N_down ,cb)
				loss = loss_eqn + Lambda_BC* loss_bc
				loss.backward()
				#return loss
				#loss = closure()
				#optimizer2.step(closure)
				#optimizer3.step(closure)
				#optimizer4.step(closure)
				optimizer2_outer.step() 
				optimizer2_inner.step()
				loss_eqn_tot += loss_eqn
				loss_bc_tot += loss_bc
				n += 1 
				if batch_idx % 20 ==0:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}\t Loss BC {:.6f}'.format(
						epoch, batch_idx * len(x_in), len(dataloader.dataset),
						100. * batch_idx / len(dataloader), loss.item(),loss_bc.item()))
			if (Flag_schedule):
						scheduler_outer.step()
						scheduler_inner.step()
						print('learning rate is ', optimizer2_inner.param_groups[0]['lr'], optimizer2_outer.param_groups[0]['lr'])
			loss_eqn_tot = loss_eqn_tot / n
			loss_bc_tot = loss_bc_tot / n
			print('*****Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f} ****'.format(loss_eqn_tot, loss_bc_tot) )
			if (epoch % 1000 == 0):
				torch.save(net2_outer.state_dict(),path+"qvortex3_outer" + ".pt")
				torch.save(net2_inner.state_dict(),path+"qvortex3_inner" + ".pt")

				#if epoch %100 == 0:
				#	torch.save(net2.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epoch)+"hard_u.pt")
		#Test with all data
		loss_eqn = criterion_inner(x,y)	 + criterion_outer(x,y)	
		loss_bc = Loss_BC(xb,yb,xb_N_up,yb_N_up ,xb_N_down,yb_N_down,cb)
		loss = loss_eqn + Lambda_BC* loss_bc
		print('**** Final (all batches) \tLoss: {:.10f} \t Loss BC {:.6f}'.format(
					loss.item(),loss_bc.item()))
		

	else:
		for epoch in range(epochs):
			#zero gradient
			#net1.zero_grad()
			##Closure function for LBFGS loop:
			#def closure():
			net2.zero_grad()
			loss_eqn = criterion(x,y)
			loss_bc = Loss_BC(xb,yb,xb_Neumann_up,yb_Neumann_up ,xb_Neumann_down,yb_Neumann_down, cb)
			if (Flag_BC_exact):
				loss = loss_eqn #+ loss_bc
			else:
				loss = loss_eqn + Lambda_BC * loss_bc
			loss.backward()
			#return loss
			#loss = closure()
			#optimizer2.step(closure)
			#optimizer3.step(closure)
			#optimizer4.step(closure)
			optimizer2.step() 
			if epoch % 10 ==0:
				print('Train Epoch: {} \tLoss: {:.10f} \t Loss BC {:.6f}'.format(
					epoch, loss.item(),loss_bc.item()))
				LOSS.append(loss.item())

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)


	torch.save(net2_outer.state_dict(),path+"qvortex3_outer" + ".pt")
	torch.save(net2_inner.state_dict(),path+"qvortex3_inner" + ".pt")



	return 

	############################################################
	##save loss
	##myFile = open('Loss track'+'stenosis_para'+'.csv','w')#
	##with myFile:
		#writer = csv.writer(myFile)
		#writer.writerows(LOSS)
	#LOSS = np.array(LOSS)
	#np.savetxt('Loss_track_pipe_para.csv',LOSS)

	############################################################

	#save network
	##torch.save(net1.state_dict(),"stenosis_para_axisy_sigma"+str(sigma)+"scale"+str(scale)+"_epoch"+str(epochs)+"boundary.pt")
	#torch.save(net2.state_dict(),path+"geo_para_axisy"+"_epoch"+str(epochs)+"c.pt")
	##torch.save(net3.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epochs)+"hard_v.pt")
	##torch.save(net4.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epochs)+"hard_P.pt")
	#####################################################################


#######################################################



############################################################################
####  Define parameters ##############

#device = torch.device("cpu")
device = torch.device("cuda")


Flag_batch = True #Use batch
Flag_Chebyshev = False #Use Chebyshev pts for more accurcy in BL region; Not implemented in 2D
Flag_BC_exact = False #Not implemented yet in 2D
Lambda_BC  = 10. #1. #10. # the weight used in enforcing the BC
Flag_pretrain = False #IF true read previous files


batchsize = 256 * 2 #128  #Total number of batches 
learning_rate = 2e-5 #3e-5  # 5e-5 #7e-6 #1e-5 / 2. # 1e-4  

Flag_schedule = True
if (Flag_schedule):
	learning_rate = 4e-4 #starting learning rate
	step_epoch = 6000 
	decay_rate = 0.5



inf_scale =  8  #100. #used to set BC at infinity 


epochs = 32000 #18000 


u_inf = 1. #free stream vel
nu = 0.1 #kinematic visc
Diff = 0.0001 #0.0005 #0.001 / 10. 
Pr = nu/Diff

nPt = 100 
xStart = 0.002 #0.01 # 0.0001   #x: r  , it is singular at r=0 
xEnd = 0.5
yStart = 0.  #y: z
yEnd = 1.

physical_height = 0.3 

if(Flag_Chebyshev): #!!!Not a very good idea (makes even the simpler case worse)
 x = np.polynomial.chebyshev.chebpts1(2*nPt)
 x = x[nPt:]
 if(0):#Mannually place more pts at the BL 
    x = np.linspace(0.95, xEnd, nPt)
    x[1] = 0.2
    x[2] = 0.5
 x[0] = 0.
 x[-1] = xEnd
 x = np.reshape(x, (nPt,1))
else:
 x = np.linspace(xStart, xEnd, nPt)
 y = np.linspace(yStart, yEnd, nPt)
 x, y = np.meshgrid(x, y)
 x = np.reshape(x, (np.size(x[:]),1))
 y = np.reshape(y, (np.size(y[:]),1))

print('shape of x',x.shape)
print('shape of y',y.shape)



############################################################################
####  Defnine the Dirichlet BCs	##############
C_BC1 = 0.  #Left Dirichlet BC
C_BC2 = 1. #Right Dirichlet BC (not used here)
flux_BC = -5. #-0.1

xleft = np.linspace(xStart, xStart, nPt)
xright = np.linspace(xEnd, xEnd, nPt)
xup = np.linspace(xStart, xEnd, nPt)
xdown = np.linspace(xStart, xEnd, nPt)
yleft = np.linspace(yStart, yEnd, nPt)
yright = np.linspace(yStart, yEnd, nPt)
yup = np.linspace(yEnd, yEnd, nPt)
ydown = np.linspace(yStart, yStart, nPt)
cleft = np.linspace(C_BC1, C_BC1, nPt)
cright = np.linspace(C_BC1, C_BC1, nPt)
cup = np.linspace(C_BC2, C_BC2, nPt)
cdown = np.linspace(C_BC1, C_BC1, nPt)

if(0): #Dirichlet BC left and right
 #xb = np.concatenate((xleft, xright), 0) 
 #yb = np.concatenate((yleft, yright), 0)
 #cb = np.concatenate((cleft, cright), 0)
 xb = np.concatenate((xright), 0) 
 yb = np.concatenate((yright), 0)
 cb = np.concatenate((cright), 0)
else: #Only Dirichlet BC right
 xb = xright #xb = np.concatenate((xleft, xright), 0)
 yb = yright #yb = np.concatenate((yleft, yright), 0)
 cb = cright #cb = np.concatenate((cleft, cright), 0)


xleft_r0 = np.linspace(0., 0., nPt)
yleft_r0 = np.linspace(yStart, yEnd, nPt)
xb2 = xleft_r0
yb2 = yleft_r0

##Define zero Neumann BC location
#xb_Neumann = np.concatenate((xup, xdown), 0) 	
#yb_Neumann = np.concatenate((yup, ydown), 0) 
xb_N_up = xup	
yb_N_up =  yup
xb_N_down = xdown	
yb_N_down = ydown

xb= xb.reshape(-1, 1) #need to reshape to get 2D array
xb2= xb2.reshape(-1, 1)
yb= yb.reshape(-1, 1) #need to reshape to get 2D array
yb2 = yb2.reshape(-1, 1) #need to reshape to get 2D array
cb= cb.reshape(-1, 1) #need to reshape to get 2D array
xb_N_up = xb_N_up.reshape(-1, 1) #need to reshape to get 2D array
yb_N_up = yb_N_up.reshape(-1, 1) #need to reshape to get 2D array
xb_N_down = xb_N_down.reshape(-1, 1) #need to reshape to get 2D array
yb_N_down = yb_N_down.reshape(-1, 1) #need to reshape to get 2D array


path = "Results/"

#Analytical soln
#A = (C_BC2 - C_BC1) / (exp(Vel/Diff) - 1)
#B = C_BC1 - A
#C_analytical = A*np.exp(Vel/Diff*x[:] ) + B



#path = pre+"aneurysmsigma01scalepara_100pt-tmp_"+str(ii)
geo_train(device,x,y,xb,yb,xb2,yb2,cb,xb_N_up,yb_N_up, xb_N_down,yb_N_down, batchsize,learning_rate,epochs,path,Flag_batch,Diff,Flag_BC_exact,Lambda_BC  )
#tic = time.time()



