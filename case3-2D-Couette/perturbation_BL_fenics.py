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
import math
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time

#Solve steady 2D advection-Diffusion eqn with given velocity profile using perturbation method:
#---> This version solves same problem is the one did in fenics

def geo_train(device,x_in,y_in,xb,yb,cb,xb_Neumann_up,yb_Neumann_up, xb_Neumann_down,yb_Neumann_down, batchsize,learning_rate,epochs,path,Flag_batch,Diff,Flag_BC_exact,Lambda_BC  ):
	if (Flag_batch):
	 x = torch.Tensor(x_in).to(device)
	 y = torch.Tensor(y_in).to(device) 
	 xb = torch.Tensor(xb).to(device) 
	 yb = torch.Tensor(yb).to(device) 
	 cb = torch.Tensor(cb).to(device) 
	 xb_Neumann_up = torch.Tensor(xb_Neumann_up).to(device) 
	 yb_Neumann_up = torch.Tensor(yb_Neumann_up).to(device) 
	 xb_Neumann_down = torch.Tensor(xb_Neumann_down).to(device) 
	 yb_Neumann_down = torch.Tensor(yb_Neumann_down).to(device) 
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
		 cb = cb.type(torch.cuda.FloatTensor)
		 xb_Neumann_up = xb_Neumann_up.type(torch.cuda.FloatTensor)
		 yb_Neumann_up = yb_Neumann_up.type(torch.cuda.FloatTensor)
		 xb_Neumann_down = xb_Neumann_down.type(torch.cuda.FloatTensor)
		 yb_Neumann_down = yb_Neumann_down.type(torch.cuda.FloatTensor)

	h_nD = 30
	h_n = 128
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
	def Velocity(x_input,y_input):
		Vel = 10.*y_input
		return Vel
	def Velocity_blas(x_input,y_input):
		#print('shape:', x_input.shape, y_input.shape )  #gives shape: torch.Size([128, 1])
		Re_x = u_inf * x_input / nu
		delta = 4.64 * x_input / torch.sqrt(Re_x)  #eqn 6.31b
		#torch.where(condition =  torch.abs(x_input) < 1e-8 and torch.abs(x_input) < 1e-8  , x, y)
		s1= u_inf * ( 1.5*y_input/delta - 0.5 * (y_input/delta)**3  )
		s2= u_inf * torch.ones_like(x_input)
		Vel1 = torch.where(y_input<delta, s1, s2)
		Vel1 = torch.where( (torch.abs(x_input) < 1e-8) & (torch.abs(y_input) < 1e-8)  ,torch.zeros_like(x_input) , Vel1)
		Vel = torch.where((torch.abs(x_input) < 1e-8) & (torch.abs(y_input) >= 1e-8)  ,s2 , Vel1)
		#if (abs(x_input) < 1e-8):  #x=0
		#	if (abs(y_input) < 1e-8):  #y=0
		#		Vel = 0. 
		#	else:
		#		Vel = u_inf
		#else:
		#	if (y_input<delta):
		#		Vel = u_inf * ( 1.5*y_input/delta - 0.5 * (y_input/delta)**3  )
		#	else:
		#		Vel = u_inf
		return Vel
	def WSS(mu_input):
		return mu_input*10.

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
	class Net2(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2, self).__init__()
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
			if (Flag_BC_exact):
				output = output*(x- xStart) *(x- xEnd )*(y- yStart) *(y- yEnd ) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
			return  output

	
	################################################################
	############################################################################
	#### Initiate the network used to represent concentration ##############
	#net1 = Net1().to(device)
	net2 = Net2().to(device)
	
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	# use the modules apply function to recursively apply the initialization
	net2.apply(init_normal)



	############################################################################

	optimizer2 = optim.Adam(net2.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	#optimizer2 = optim.LBFGS(net2.parameters(), lr=learning_rate)


	############################################################################
	#### Define the governing equation ##############
	def criterion(x,y):

		#print (x)
		#x = torch.Tensor(x).to(device)
		#y = torch.Tensor(y).to(device)
		#x = torch.FloatTensor(x).to(device)
		#x= torch.from_numpy(x).to(device)

		x.requires_grad = True
		y.requires_grad = True
		
		#net_in = torch.cat((x),1)
		net_in = torch.cat((x,y),1)
		C = net2(net_in)
		C = C.view(len(C),-1)

		u = Velocity(x,y*inf_scale * math.sqrt(Diff) )
		
		c_x = torch.autograd.grad(C,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		c_xx = torch.autograd.grad(c_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		c_y = torch.autograd.grad(C,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		c_yy = torch.autograd.grad(c_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		
		#loss_1 = u *c_x - Diff * (c_xx + c_yy)
		loss_1 = u *c_x - c_yy / (inf_scale**2) - Diff * c_xx



		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1)) 

		return loss

	###################################################################
	############################################################################
	#### Define the boundary conditions ##############
	def Loss_BC(xb,yb,xb_Neumann_up,yb_Neumann_up ,xb_Neumann_down,yb_Neumann_down,cb):
		#xb = torch.FloatTensor(xb).to(device)
		#yb = torch.FloatTensor(yb).to(device)
		#cb = torch.FloatTensor(cb).to(device)

		#xb_Neumann_up = torch.FloatTensor(xb_Neumann_up).to(device)
		#yb_Neumann_up = torch.FloatTensor(yb_Neumann_up).to(device)
		#xb_Neumann_down = torch.FloatTensor(xb_Neumann_down).to(device)
		#yb_Neumann_down = torch.FloatTensor(yb_Neumann_down).to(device)

		xb_Neumann_up.requires_grad = True
		yb_Neumann_up.requires_grad = True
		xb_Neumann_down.requires_grad = True
		yb_Neumann_down.requires_grad = True

		net_in = torch.cat((xb, yb), 1)
		out = net2(net_in )


		c_bc = out.view(len(out), -1)

		loss_f = nn.MSELoss()
		loss_bc_Dirichlet_left = loss_f(c_bc, cb)  #The Dirichlet BC (left)


		net_in_up = torch.cat((xb_Neumann_up, yb_Neumann_up), 1)
		out_n = net2(net_in_up )
		c_n_up = out_n.view(len(out_n), -1)
		loss_bc_match = loss_f(c_n_up , torch.zeros_like(c_n_up) )  #The Dirichlet BC (up)

		net_in_down = torch.cat((xb_Neumann_down, yb_Neumann_down), 1)
		out_n = net2(net_in_down )
		c_n_down = out_n.view(len(out_n), -1)

		#c_y_up = torch.autograd.grad(c_n_up,yb_Neumann_up,grad_outputs=torch.ones_like(yb_Neumann_up),create_graph = True,only_inputs=True)[0]
		c_y_down = torch.autograd.grad(c_n_down,yb_Neumann_down,grad_outputs=torch.ones_like(yb_Neumann_down),create_graph = True,only_inputs=True)[0]

		#loss_bc_Neumann_up = loss_f(c_y_up, torch.zeros_like(c_y_up))  #The zero Neumann BC top
		loss_bc_Neumann_down = loss_f(c_y_down, flux_BC* math.sqrt(Diff)*inf_scale* torch.ones_like(c_y_down))  #The flux Neumann BC bottom

		loss_bc = loss_bc_Dirichlet_left + loss_bc_match + loss_bc_Neumann_down

		return loss_bc


	
	LOSS = []
	tic = time.time()


	if(Flag_pretrain):
		print('Reading (pretrain) functions first...')
		net2.load_state_dict(torch.load(path+"pertfe_2Dbl" + ".pt"))


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
				net2.zero_grad()
				loss_eqn = criterion(x_in,y_in)
				loss_bc = Loss_BC(xb,yb,xb_Neumann_up,yb_Neumann_up ,xb_Neumann_down,yb_Neumann_down ,cb)
				loss = loss_eqn + Lambda_BC* loss_bc
				loss.backward()
				#return loss
				#loss = closure()
				#optimizer2.step(closure)
				#optimizer3.step(closure)
				#optimizer4.step(closure)
				optimizer2.step() 
				loss_eqn_tot += loss_eqn
				loss_bc_tot += loss_bc
				n += 1 
				if batch_idx % 20 ==0:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}\t Loss BC {:.6f}'.format(
						epoch, batch_idx * len(x_in), len(dataloader.dataset),
						100. * batch_idx / len(dataloader), loss.item(),loss_bc.item()))
			loss_eqn_tot = loss_eqn_tot / n
			loss_bc_tot = loss_bc_tot / n
			print('*****Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f} ****'.format(loss_eqn_tot, loss_bc_tot) )

				#if epoch %100 == 0:
				#	torch.save(net2.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epoch)+"hard_u.pt")
		#Test with all data
		loss_eqn = criterion(x,y)	
		loss_bc = Loss_BC(xb,yb,xb_Neumann_up,yb_Neumann_up ,xb_Neumann_down,yb_Neumann_down,cb)
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


	torch.save(net2.state_dict(),path+"pertfe_2Dbl" + ".pt")


	#####
	#Analytical soln; integral after Eqn. 6.71 (does not apply to this problem)
	x_wall = np.linspace(xStart, xEnd, 100)
	#T_wall = -flux_BC * np.sqrt(x_wall[:]) / ( 0.4587 * math.sqrt(u_inf/nu)* (Pr**(1./3.)) ) 
	y_wall = np.linspace(yStart, yStart, 100)
	x_wall= x_wall.reshape(-1, 1)
	y_wall= y_wall.reshape(-1, 1)
	x_wall = torch.Tensor(x_wall).to(device)
	y_wall = torch.Tensor(y_wall).to(device)
	net_in = torch.cat((x_wall,y_wall),1)
	output_wall = net2(net_in)  #evaluate model
	C_Result_wall = output_wall.cpu().data.numpy()

	plt.figure()
	#plt.plot(x_wall.cpu().data.numpy()[:], T_wall[:], '--', label='AnalyticalT_wall', alpha=0.5) 
	plt.plot(x_wall.cpu().data.numpy()[:], C_Result_wall[:], '--', label='PINN T_wall', alpha=0.5) 
	plt.legend(loc='best')
	plt.show()

	###################
	#plot
	net_in = torch.cat((x,y),1)
	output = net2(net_in)  #evaluate model
	C_Result = output.cpu().data.numpy()

	x = x.cpu()
	y = y.cpu()

	plt.figure()
	plt.subplot(2, 1, 1)
	plt.scatter(x.detach().numpy(), inf_scale *math.sqrt(Diff) * y.detach().numpy(), c = C_Result, cmap = 'coolwarm')
	plt.title('PINN results')
	plt.colorbar()
	plt.show()

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
Flag_pretrain = False# True #IF true read previous files


batchsize = 128  #Total number of batches 
learning_rate = 1e-5 / 2. # 1e-4  


inf_scale =  8  #100. #used to set BC at infinity 


epochs = 2000 #6000 #1000


u_inf = 1. #free stream vel
nu = 0.1 #kinematic visc
Diff = 0.001 / 10. 
Pr = nu/Diff

nPt = 100 
xStart = 0.
xEnd = 1.
yStart = 0.
yEnd = 1.

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
flux_BC = -10. #-0.1

xleft = np.linspace(xStart, xStart, nPt)
xright = np.linspace(xEnd, xEnd, nPt)
xup = np.linspace(xStart, xEnd, nPt)
xdown = np.linspace(xStart, xEnd, nPt)
yleft = np.linspace(yStart, yEnd, nPt)
yright = np.linspace(yStart, yEnd, nPt)
yup = np.linspace(yEnd, yEnd, nPt)
ydown = np.linspace(yStart, yStart, nPt)
cleft = np.linspace(C_BC1, C_BC1, nPt)
cright = np.linspace(C_BC2, C_BC2, nPt)
cup = np.linspace(C_BC2, C_BC2, nPt)
cdown = np.linspace(C_BC1, C_BC1, nPt)

if(0): #Dirichlet BC everywhere
 xb = np.concatenate((xleft, xright, xup, xdown), 0)
 yb = np.concatenate((yleft, yright, yup, ydown), 0)
 cb = np.concatenate((cleft, cright, cup, cdown), 0)
else: #Only Dirichlet BC left 
 xb = xleft #xb = np.concatenate((xleft, xright), 0)
 yb = yleft #yb = np.concatenate((yleft, yright), 0)
 cb = cleft #cb = np.concatenate((cleft, cright), 0)

##Define zero Neumann BC location
#xb_Neumann = np.concatenate((xup, xdown), 0) 	
#yb_Neumann = np.concatenate((yup, ydown), 0) 
xb_Neumann_up = xup	
yb_Neumann_up =  yup
xb_Neumann_down = xdown	
yb_Neumann_down = ydown

xb= xb.reshape(-1, 1) #need to reshape to get 2D array
yb= yb.reshape(-1, 1) #need to reshape to get 2D array
cb= cb.reshape(-1, 1) #need to reshape to get 2D array
xb_Neumann_up = xb_Neumann_up.reshape(-1, 1) #need to reshape to get 2D array
yb_Neumann_up = yb_Neumann_up.reshape(-1, 1) #need to reshape to get 2D array
xb_Neumann_down = xb_Neumann_down.reshape(-1, 1) #need to reshape to get 2D array
yb_Neumann_down = yb_Neumann_down.reshape(-1, 1) #need to reshape to get 2D array


path = "Results/"

#Analytical soln
#A = (C_BC2 - C_BC1) / (exp(Vel/Diff) - 1)
#B = C_BC1 - A
#C_analytical = A*np.exp(Vel/Diff*x[:] ) + B



#path = pre+"aneurysmsigma01scalepara_100pt-tmp_"+str(ii)
geo_train(device,x,y,xb,yb,cb,xb_Neumann_up,yb_Neumann_up, xb_Neumann_down,yb_Neumann_down, batchsize,learning_rate,epochs,path,Flag_batch,Diff,Flag_BC_exact,Lambda_BC  )
#tic = time.time()


 



