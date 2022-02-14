import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


class TrajOpt(object):

    def __init__(self):

        # initialize trajectory
        self.n_waypoints = 10
        self.n_dof = 2
        self.home = np.array([0., 0.])
        self.xi0 = np.zeros((self.n_waypoints, self.n_dof))
        self.xi0 = self.xi0.reshape(-1)

        # create start constraint and action constraint
        self.B = np.zeros((self.n_dof, self.n_dof * self.n_waypoints))
        for idx in range(self.n_dof):
            self.B[idx,idx] = 1
        self.lincon = LinearConstraint(self.B, self.home, self.home)
        self.nonlincon = NonlinearConstraint(self.nl_function, -1.0, 1.0)

    # each action cannot move more than 1 unit
    def nl_function(self, xi):
        xi = xi.reshape(self.n_waypoints, self.n_dof)
        actions = xi[1:, :] - xi[:-1, :]
        return np.linalg.norm(actions, axis=1)

    # trajectory cost function
    def trajcost(self, xi):
        xi = xi.reshape(self.n_waypoints, self.n_dof)
        cost = 0
        ### define your cost function here ###
        ### here is an example encouraging the robot to reach [5, 2] ###
        for idx in range(self.n_waypoints):
            cost += np.linalg.norm(np.array([5., 2.]) - xi[idx, :])
        return cost

    # run the optimizer
    def optimize(self):
        res = minimize(self.trajcost, self.xi0, method='SLSQP', constraints={self.lincon, self.nonlincon}, options={'eps': 1e-3, 'maxiter': 1000})
        xi = res.x.reshape(self.n_waypoints, self.n_dof)
        return xi, res


# here is the code to run our trajopt
trajopt = TrajOpt()
xi, res = trajopt.optimize()
print(xi)
plt.plot(xi[:,0], xi[:,1], 'bo-')
plt.axis("equal")
plt.show()
