import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from train_model import BC


model = BC(4, 2, 32)
model.load_state_dict(torch.load("bc_weights"))


N = 10          # number of rollouts
T = 20          # each one has T timesteps
for iter in range(N):
    xi = np.zeros((T, 2))
    s = np.random.rand(2)
    g = np.random.rand(2)
    for timestep in range(T):
        context = np.concatenate((s, g))
        a = model.encoder(torch.Tensor(context)).detach().numpy()
        xi[timestep, :] = np.copy(s)
        s += a
    plt.plot(g[0], g[1], 'ko')
    plt.plot(xi[:,0], xi[:,1], 'bo-')
    plt.axis([0, 1, 0, 1])
    plt.show()
