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
import scipy.io

#Solve 1D nonlinear eqn in "advanced mathematical methods for scientists and engineers" book page 463 Sec 9.7

def geo_train(device,x_in,y_in,xb,cb,batchsize,learning_rate,epochs,path,Flag_batch,C_analytical,Diff,Flag_BC_exact ):
	if (Flag_batch):
	 dataset = TensorDataset(torch.Tensor(x_in))
	 dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = True )
	else:
	 x = torch.Tensor(x_in)  
	 y = torch.Tensor(y_in)  
	h_n = 60 # 40
	input_n = 1 # this is what our answer is a function of. In the original example 3 : x,y,scale 
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
				output = output*x*(x-1) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
				#output = output*x*(1-x) + torch.exp(math.log(0.1)*x) #Do it exponentially? Not as good
			return  output

	class Net1(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net1, self).__init__()
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
				output = output*x*(x-1) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
				#output = output*x*(1-x) + torch.exp(math.log(0.1)*x) #Do it exponentially? Not as good
			return  output

	class Net3(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net3, self).__init__()
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
				output = output*x*(x-1) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
				#output = output*x*(1-x) + torch.exp(math.log(0.1)*x) #Do it exponentially? Not as good
			return  output

	
	################################################################

	net_inner = Net1().to(device) 
	net_outer = Net2().to(device) 
	net_original = Net3().to(device)  #original method


	###### Initialize the neural network using a standard method ##############
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	# use the modules apply function to recursively apply the initialization
	net_inner.apply(init_normal)
	net_outer.apply(init_normal)
	net_original.apply(init_normal)

	############################################################
	optimizer1 = optim.Adam(net_inner.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer2 = optim.Adam(net_outer.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer3 = optim.Adam(net_original.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)


	###### Definte the PDE and physics loss here ##############

	def criterion_original(x):

		#print (x)
		x = torch.Tensor(x).to(device)
		#x = torch.FloatTensor(x).to(device)
		#x= torch.from_numpy(x).to(device)

		x.requires_grad = True
		
		#net_in = torch.cat((x),1)
		net_in = x
		C = net_original(net_in)
		C = C.view(len(C),-1)



		
		c_x = torch.autograd.grad(C,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		c_xx = torch.autograd.grad(c_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		
		loss_1 =   Diff * c_xx + 2 * c_x + torch.exp(C)




		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1)) 

		return loss

	def criterion_outer(x):

		#print (x)
		x = torch.Tensor(x).to(device)
		#x = torch.FloatTensor(x).to(device)
		#x= torch.from_numpy(x).to(device)

		x.requires_grad = True
		
		#net_in = torch.cat((x),1)
		net_in = x
		C = net_outer(net_in)
		C = C.view(len(C),-1)


		
		c_x = torch.autograd.grad(C,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		#c_xx = torch.autograd.grad(c_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		
		loss_1 =  2*c_x + torch.exp(C) 




		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1)) 

		return loss

	def criterion_inner(x):

		#print (x)
		x = torch.Tensor(x).to(device)
		#x = torch.FloatTensor(x).to(device)
		#x= torch.from_numpy(x).to(device)

		x.requires_grad = True
		
		#net_in = torch.cat((x),1)
		net_in = x
		C = net_inner(net_in)
		C = C.view(len(C),-1)


		
		c_x = torch.autograd.grad(C,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		c_xx = torch.autograd.grad(c_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		
		loss_1 =  c_xx/inf_scale + 2 * c_x 




		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1)) 

		return loss

	###### Define boundary conditions ##############

	###################################################################
	def Loss_BC_original(xb,cb):
		xb = torch.FloatTensor(xb).to(device)
		cb = torch.FloatTensor(cb).to(device)

		#xb.requires_grad = True
		
		#net_in = torch.cat((xb),1)
		out = net_original(xb)
		cNN = out.view(len(out), -1)
		#cNN = cNN*(1.-xb) + cb    #cNN*xb*(1-xb) + cb
		loss_f = nn.MSELoss()
		loss_bc = loss_f(cNN, cb/U_scale)
		return loss_bc

	def Loss_BC_outer(xb):
		xb = torch.FloatTensor(xb).to(device)
		#cb = torch.FloatTensor(cb).to(device)

		#xb.requires_grad = True
		
		#net_in = torch.cat((xb),1)
		out = net_outer(xb)
		cNN = out.view(len(out), -1)
		#cNN = cNN*(1.-xb) + cb    #cNN*xb*(1-xb) + cb
		loss_f = nn.MSELoss()
		loss_bc = loss_f(cNN, torch.zeros_like(cNN))
		return loss_bc

	def Loss_BC_inner(xb):
		xb = torch.FloatTensor(xb).to(device)
		#cb = torch.FloatTensor(cb).to(device)

		#xb.requires_grad = True
		
		#net_in = torch.cat((xb),1)
		out = net_inner(xb)
		cNN = out.view(len(out), -1)
		#cNN = cNN*(1.-xb) + cb    #cNN*xb*(1-xb) + cb
		loss_f = nn.MSELoss()
		loss_bc = loss_f(cNN, torch.zeros_like(cNN))
		return loss_bc


	def Loss_BC_match(xb,yb):
		xb = torch.FloatTensor(xb).to(device)
		yb = torch.FloatTensor(yb).to(device)

		#xb.requires_grad = True
		
		#net_in = torch.cat((xb),1)
		out = net_inner(yb)
		output_inner = out.view(len(out), -1)

		out = net_outer(xb)
		output_outer = out.view(len(out), -1)

		#cNN = cNN*(1.-xb) + cb    #cNN*xb*(1-xb) + cb
		loss_f = nn.MSELoss()
		loss_bc = loss_f(output_inner, output_outer)
		return loss_bc


	######## Main loop ###########

	tic = time.time()

	if(Flag_pretrain):
		print('Reading (pretrain) functions first...')
		net_outer.load_state_dict(torch.load(path+"outer_1dnonlinear" + ".pt"))
		net_inner.load_state_dict(torch.load(path+"inner_1dnonlinear" + ".pt"))
		net_original.load_state_dict(torch.load(path+"original_1dnonlinear" + ".pt"))

	if(Flag_batch):# This one uses dataloader
		for epoch in range(epochs):
			for batch_idx, (x_in) in enumerate(dataloader):
		
				net2.zero_grad()
				loss_eqn = criterion(x_in)
				loss_bc = Loss_BC(xb,cb)
				loss = loss_eqn + loss_bc
				loss.backward()
		
				optimizer2.step() 
				if batch_idx % 100 ==0:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
						epoch, batch_idx * len(x), len(dataloader.dataset),
						100. * batch_idx / len(dataloader), loss.item()))
					
				#if epoch %100 == 0:
				#	torch.save(net2.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epoch)+"hard_u.pt")
	else:
		for epoch in range(epochs):
			#zero gradient
			#net1.zero_grad()
			##Closure function for LBFGS loop:
			#def closure():
			net_inner.zero_grad()
			net_outer.zero_grad()
			net_original.zero_grad()
			loss_eqn_outer = criterion_outer(x)
			loss_eqn_inner = criterion_inner(x)
			loss_eqn_original = criterion_original(x) #the original method (traditional adv-dif solver)
			loss_bc_inner = Loss_BC_inner(xb_inner)
			loss_bc_outer = Loss_BC_outer(xb_outer)
			loss_bc_match = Loss_BC_match(xb_inner ,xb_outer)
			loss_bc_original = Loss_BC_original(xb,cb)

			loss = loss_eqn_outer + loss_eqn_inner + Lambda_bc * (loss_bc_inner + loss_bc_outer + 10.* loss_bc_match)
			loss.backward()

			loss_original = loss_eqn_original + Lambda_bc*loss_bc_original
			loss_original.backward()
	
			optimizer1.step() 
			optimizer2.step() 
			optimizer3.step() 
			if epoch % 5 ==0:
				print('Train Epoch: {} \tLoss: {:.10f} Loss_eqn-inner: {:.10f} Loss_eqn-outer: {:.10f}   '.format(epoch, loss.item(),loss_eqn_inner.item(),loss_eqn_outer.item()  ))
				print('Loss_bc_inner: {:.8f} Loss_bc_outer: {:.8f}  Loss_bc_match: {:.8f}  '.format( loss_bc_inner.item(),loss_bc_outer.item(),loss_bc_match.item()  ))
				print('Loss original {:.10f} Loss_eqn_original {:.10f} Loss_bc_original: {:.8f}'.format( loss_original.item() ,loss_eqn_original.item(),loss_bc_original.item()  ))

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time = ", elapseTime)
	###################
	#plot

	results = net_original(x)  #evaluate model
	C_Result_original = U_scale * results.data.numpy()

	outer_results = net_outer(x)  #evaluate model
	C_Result_outer =  U_scale * outer_results.data.numpy()

	x_inner = np.linspace(0., inf_scale *Diff , nPt) #This physical x value gives--> 0<y<1
	inner_results = net_inner(x)  #evaluate inner model (input is y = zeta/inf_scale where zeta = x/Diff )==> input = y = x/(inf_scale *Diff)
	C_Result_inner = U_scale * inner_results.data.numpy()


	#compute gradient at the BL to evaluate BL accuracy quantitatively 
	if(0):
		xbinner = torch.Tensor(xb_inner).to(device)
		xbinner.requires_grad = True
		net_in = xbinner
		out = net_inner(net_in)
		out = out.view(len(out),-1)
		c_x0 = torch.autograd.grad(out,xbinner,grad_outputs=torch.ones_like(xbinner),create_graph = True,only_inputs=True)[0]
		grad_output = U_scale * c_x0.data.numpy()
		out_original = net_original(net_in)
		out_original = out_original.view(len(out_original),-1)
		c_x0_original = torch.autograd.grad(out_original,xbinner,grad_outputs=torch.ones_like(xbinner),create_graph = True,only_inputs=True)[0]
		grad_output_original = U_scale * c_x0_original.data.numpy()

	#print('PINN gradient at BL wall: ', grad_output )
	#print('PINN gradient at BL wall (original): ', grad_output_original )
	#print('Analytical soln gradient at BL wall: ', B * ( -1.0 + 1./Diff )  )

	torch.save(net_outer.state_dict(),path+"outer_1dnonlinear" + ".pt")
	torch.save(net_inner.state_dict(),path+"inner_1dnonlinear" + ".pt")
	torch.save(net_original.state_dict(),path+"original_1dnonlinear" + ".pt")


	################
	#compute error for perturbation method###
	x1 = np.linspace(x_inner[-1], 1. , nPt) 
	x_all = np.concatenate((x_inner, x1), 0)
	x_all2 = x_all.reshape(-1, 1)
	x_all2 = torch.Tensor(x_all2).to(device)
	results_all_outer = net_outer(x_all2)  #evaluate model
	results_all_outer =  U_scale * results_all_outer.data.numpy()
	results_all_inner = net_inner(x_all2/(Diff*inf_scale))  #evaluate model
	results_all_inner =  U_scale * results_all_inner.data.numpy()
	
	#A2 = ( np.exp(-x_all[:]) -  np.exp(-x_all[:]/Diff)  ) #analytical soln
	#B2 = 1. / ( exp(-1.) -  exp(-1./Diff)  )  #analytical soln
	#C_analytical2 = A2 * B2  #analytical soln
	#e_outer = abs(results_all_outer[:,0] - C_analytical2[:]) #outer error
	#e_inner = abs(results_all_inner[:,0] - C_analytical2[:]) #inner error
	#error_final = np.minimum(e_outer,e_inner)

	#print('shapes ', np.shape(e_outer), np.shape(results_all_outer), np.shape(x_all) )


	#load numerical soln from Matlab
	data_mat = scipy.io.loadmat(analytical_file)
	X_mat = data_mat['x']
	Y_mat = data_mat['y']
	X_mat = np.array(X_mat[0,:])
	Y_mat = np.array(Y_mat[0,:])




	#### Plot ########
	plt.figure()
	plt.plot(X_mat[:],Y_mat,'-', label='True solution', alpha=1.0,zorder=0)
	plt.plot(x.detach().numpy() , C_Result_original, 'k-', label='Original PINN', alpha=1.,zorder=0) #PINN
	plt.plot(x.detach().numpy() , C_Result_outer, '--', label='BL-PINN outer solution', alpha=1.,markersize=6,zorder=10,color='violet') #PINN
	plt.plot(x_inner[:] , C_Result_inner, 'r--', label='BL-PINN inner solution', alpha=1.,markersize=6,zorder=10) #PINN
	plt.legend(loc='best')
	plt.show()



	#plt.figure()
	#plt.plot(x_all[:], error_final[:], '--', label='Error', alpha=0.5) 
	#plt.legend(loc='best')
	#plt.show()

	return


#######################################################
#Main code:
device = torch.device("cpu")
epochs  = 2000 #2500

Flag_batch = False #Use batch or not 
Flag_BC_exact = False #If True enforces BC exactly HELPS ALOT here!!!
Flag_pretrain =  False #IF true read previous files

Lambda_bc = 1.

## Parameters###
Diff = 1e-3 #0.1 

nPt = 100 
xStart = 0.
xEnd = 1.


x = np.linspace(xStart, xEnd, nPt)
x = np.reshape(x, (nPt,1))

#zeta = x/Diff
#y = zeta / inf_scale   ( 0<y<1)

y = np.linspace(0, 1, nPt)
y = np.reshape(y, (nPt,1))

print('shape of x',x.shape)

#boundary pt and boundary condition
#X_BC_loc = 1.
#C_BC = 1.
#xb = np.array([X_BC_loc],dtype=np.float32)
#cb = np.array([C_BC ], dtype=np.float32)
C_BC1 = 0.
C_BC2 = 0.
xb = np.array([0.,1.],dtype=np.float32)
cb = np.array([C_BC1,C_BC2 ], dtype=np.float32)
xb= xb.reshape(-1, 1) #need to reshape to get 2D array
cb= cb.reshape(-1, 1) #need to reshape to get 2D array
#xb = np.transpose(xb)  #transpose because of the order that NN expects instances of training data
#cb = np.transpose(cb)


xb_inner = np.array([0.],dtype=np.float32)
xb_inner= xb_inner.reshape(-1, 1)
xb_outer = np.array([1.],dtype=np.float32)
xb_outer= xb_outer.reshape(-1, 1)


batchsize = 64
learning_rate = 1e-4

inf_scale = 8.  #100. #used to set BC at infinity 



U_scale = 1.0  # This is the max value in analytical soln; need to scale by this

path = "Results/"
analytical_file = 'pert_nonlinearODE'

#Analytical soln
A = ( np.exp(-x[:]) -  np.exp(-x[:]/Diff)  )
B = 1. / ( exp(-1.) -  exp(-1./Diff)  )
C_analytical = A * B



#path = pre+"aneurysmsigma01scalepara_100pt-tmp_"+str(ii)
net2_final = geo_train(device,x,y,xb,cb,batchsize,learning_rate,epochs,path,Flag_batch,C_analytical,Diff,Flag_BC_exact )
#tic = time.time()



