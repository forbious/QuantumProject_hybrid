from ctypes import resize
from typing_extensions import Self
import pennylane as qml
import numpy as np
import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim



import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram, plot_bloch_vector
import os




os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



#pytorch uses cuda acceleration if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_qubits = 3

#dev = qml.device('qiskit.aer', wires=n_qubits) #modified for qiskit

dev = qml.device('qiskit.aer', wires=n_qubits) #modified for qiskit

#in classical gans the starting points is to draw samples from either a real distribution or from the generator

#this simple example, our real data will be a qbit that has been rotated from the startign state |0> to an arbitrary fixed state

phi = np.pi / 6
theta = np.pi / 2
omega = np.pi / 7


def real(angles, **kwargs):
    qml.Hadamard(wires=0)
    qml.Rot(*angles, wires=0)




#real data circuit and "fake" data will output on the same wire 0
#while wire 1 is provided as a workspace for the generator

def generator(w, **kwargs):
    qml.Hadamard(wires=0)
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=1)
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=1)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(w[6], wires=0)
    qml.RY(w[7], wires=0)
    qml.RZ(w[8], wires=0)

#the discriminator's output will be on wire 2

def discriminator(w):
    qml.Hadamard(wires=0)
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=2)
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=2)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=2)
    qml.CNOT(wires=[0, 2])
    qml.RX(w[6], wires=2)
    qml.RY(w[7], wires=2)
    qml.RZ(w[8], wires=2)


#creating our QNodes 


#real data to discriminator
@qml.qnode(dev, interface="torch", diff_method="parameter-shift") #modified for torch
def realTo_disc_circuit(phi, theta, omega, disc_weights):
    inputs = [phi, theta, omega]
    real(inputs)
    discriminator(disc_weights)
    #measures expectation value for 
    
    return qml.expval(qml.PauliZ(2))
        




#gen to disc
@qml.qnode(dev, interface="torch", diff_method="parameter-shift") #modified for torch
def genTo_disc_circuit(inputs, gen_weights):
    generator(gen_weights)
    discriminator(inputs)
    return qml.expval(qml.PauliZ(2))

# we need to define 2 cost functions for out discriminator and generator


def prob_real_true(disc_weights):
    p_realTrue = (realTo_disc_circuit(phi, theta, omega, disc_weights) + 1)/2
    
    return p_realTrue



def prob_fake_true(gen_weights, disc_weights):
    p_fakeTrue = (genTo_disc_circuit(gen_weights, disc_weights) + 1)/2
    
    return p_fakeTrue


#intilized to be close to |1>
phi = np.pi / 6
theta = np.pi / 2
omega = np.pi / 7
np.random.seed(0)
eps = 1e-2
init_gen_weights = torch.from_numpy(np.array([np.pi] + [0] * 8) + \
                   np.random.normal(scale=eps, size=(9,)))
init_disc_weights = torch.from_numpy(np.random.normal(size=(9,)))

#Parameter allow back propagation
print(prob_real_true(init_disc_weights))
print(prob_fake_true(init_gen_weights,init_disc_weights))

gen_weights = nn.ParameterList(
    
        nn.Parameter(data=init_gen_weights[x], requires_grad = True)
            for x in range(9)
    ).to(device)        

disc_weights = nn.ParameterList(
    
        nn.Parameter(data=init_disc_weights[x], requires_grad = True)
        for x in range(9)
    ).to(device)



angles = np.array([phi,theta,omega])
inputs_real = torch.from_numpy(angles)


epochs = 30
lrG = 0.1
lrD = 0.4

#binary cross entropy loss function
loss = nn.BCELoss()


#trick for using same function by swapping 0s and 1s
real_labels = torch.full((1,), 0.0, dtype=torch.double, device=device)

fake_labels = torch.full((1,), 1.0, dtype=torch.double, device=device)

optD = optim.SGD(disc_weights, lr=lrD,momentum=0.2)
optG = optim.SGD(gen_weights, lr=lrG,momentum=0.2)

epochList = []
T_lossD_sim = []
T_lossG_sim = []
T_errorD_sim = []

generated_bloch_vectors_sim = []
list_Error_sim = []

obs = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]

bloch_vector_generator = qml.map(generator, obs, dev, interface="torch")

bloch_vector_real = qml.map(real, obs, dev, interface="torch")

for step in range(epochs):
    
    optD.zero_grad()
    outD_real = prob_real_true(disc_weights).reshape(1)
    outD_fake = prob_fake_true(gen_weights, disc_weights).reshape(1)
    
    real_errorD = loss(outD_real, real_labels)
    fake_errorD = loss(outD_fake, fake_labels)
    #backpropogate gradients
    real_errorD.backward()
    fake_errorD.backward()
    
    errorD = real_errorD + fake_errorD
    toNumD = errorD.cpu().detach().numpy()
    T_lossD_sim.append(toNumD.tolist())
    optD.step()

    optG.zero_grad()
    outD_fake = prob_fake_true(gen_weights, disc_weights).reshape(1)
    errorG = loss(outD_fake, real_labels)
    errorG.backward()
    optG.step()

    
    toNumG = errorG.cpu().detach().numpy()
    
    T_lossG_sim.append(toNumG.tolist())

    #calc distance
    list_Error_sim.append(np.linalg.norm((bloch_vector_generator(gen_weights).detach().cpu().numpy() - bloch_vector_real([phi, theta, omega]).cpu().numpy())))

    epochList.append(step)

    if step % 2 == 0:
        print(f'Iteration: {step}, Discriminator Loss: {errorD:0.3f}, Generator Loss: {errorG:0.3f}')






    
figure_1, ax1 = plt.subplots(1)
ax1.plot(epochList, T_lossD_sim, label= "Discriminator")
ax1.plot(epochList, T_lossG_sim, label= "Generator")
ax1.set_xlabel('epochs')
ax1.set_ylabel('Loss Value')
ax1.set_title('Loss of Generator and Discriminator vs Epochs')
ax1.legend()


figure_2, ax2 = plt.subplots(1)
ax2.plot(epochList, list_Error_sim, label= "Distance")
ax2.set_xlabel('epochs')
ax2.set_ylabel('Distance')
ax2.set_title('Total Distance from Real vector vs Epochs')










bloch_vector_real = qml.map(real, obs, dev, interface="torch")
    


bloch_vector_generator = qml.map(generator, obs, dev, interface="torch")

#real bloch vector is figure 3
plot_bloch_vector(bloch_vector_real([phi, theta, omega]).tolist())
#gen_bloch vector is figure 4
plot_bloch_vector(bloch_vector_generator(gen_weights).tolist())

plt.show()