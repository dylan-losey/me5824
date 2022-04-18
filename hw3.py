import numpy as np

def boltzmann(a, theta, N=10000, beta=1.0):
    P_num = np.exp(beta * -np.linalg.norm(a - theta))
    P_den = 0
    for idx in range(N):
        action = np.random.random(2) * 2 - 1.0
        P_den += np.exp(beta * -np.linalg.norm(action - theta))
    return P_num / P_den

theta_cup = np.array([-1, 0])
theta_plate = np.array([0, 1])
theta_bowl = np.array([-0.5, -0.5])
a = np.array([-0.2, 0.6])

P_a_cup = boltzmann(a, theta_cup)
P_a_plate = boltzmann(a, theta_plate)
P_a_bowl = boltzmann(a, theta_bowl)

belief = np.array([P_a_cup * 0.4, P_a_plate * 0.4, P_a_bowl * 0.2])
print(belief / np.sum(belief))
