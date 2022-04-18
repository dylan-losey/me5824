import numpy as np
import pickle
import matplotlib.pyplot as plt



N = 3             # number of demonstrations
sigma_h = 0.1      # amount of noise in the demonstration
T = 10              # each demonstration has T timesteps
D = []
for iter in range(N):
    xi = np.zeros((T, 2))
    s = np.random.rand(2)
    g = np.random.rand(2)
    for timestep in range(T):
        a = np.random.normal((g - s) / 5.0, sigma_h)
        xi[timestep, :] = np.copy(s)
        D.append(s.tolist() + g.tolist() + a.tolist())
        s += a
    plt.plot(g[0], g[1], 'ko')
    plt.plot(xi[:,0], xi[:,1], 'bo-')
plt.axis([0, 1, 0, 1])
plt.show()

# save the data to a pickle file
pickle.dump(D, open("demos.pkl", "wb"))
