# BL-PINN
Theory-guided physics-informed neural networks for boundary layer problems with singular perturbation

The codes for the paper  A. Arzani,  K. W. Cassel, R. M. D'Souza, Theory-guided physics-informed neural networks for boundary layer problems with singular perturbation, Journal of Computational Physics. \


Pytorch codes are included for the different examples presented in the paper. \
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \

Convering the results to VTK:  A sample torch2vtk code is provided that shows how the outputs could be converted to VTK format that could be visualized in ParaView. You should edit this code for your own application and neural network. \


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
Installation: \
Install Pytorch: \
https://pytorch.org/

Install VTK after Pytorch is installed.  \
An example with pip:

conda activate pytorch \
pip install vtk 
