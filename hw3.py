import numpy as np

def boltz(Q, q, theta, beta):
    num = np.exp(-beta*abs(theta - q))
    den = np.exp(-beta*abs(theta - Q[0])) + np.exp(-beta*abs(theta - Q[1]))
    return num / den

Q1 = [55, 55]
Q2 = [40, 60]
Q3 = [10, 90]
Q4 = [0, 100]

Qset = [Q1, Q2, Q3, Q4]
beta = 0.01

for Q in Qset:
    Qval = 0
    for q in Q:
        normalizer = 0
        for theta1 in range(101):
            normalizer += boltz(Q, q, theta1, beta)
        for theta in range(101):
            P = boltz(Q, q, theta, beta)
            Qval += P * np.log2( 101 * P / normalizer )

    print(Qval)
