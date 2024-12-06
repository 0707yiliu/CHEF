import time

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from gymnasium_envs.DMPs.cs import CanonicalSystem
from gymnasium_envs.utils import _inv_normalization


# %% define discrete dmp
class dmp_discrete():
    def __init__(self, n_dmps=1, n_bfs=100, dt=0, alpha_y=None, beta_y=None, **kwargs):
        self.n_dmps = n_dmps  # number of data dimensions, one dmp for one degree
        self.n_bfs = n_bfs  # number of basis functions
        self.dt = dt

        self.y0 = np.zeros(n_dmps)  # for multiple dimensions
        self.goal = np.ones(n_dmps)  # for multiple dimensions

        alpha_y_tmp = 60 if alpha_y is None else alpha_y
        beta_y_tmp = alpha_y_tmp / 4.0 if beta_y is None else beta_y
        self.alpha_y = np.ones(n_dmps) * alpha_y_tmp
        self.beta_y = np.ones(n_dmps) * beta_y_tmp
        self.tau = 1.0
        self.delta = np.ones((n_dmps, 1))
        self.delta_2 = np.ones((n_dmps, 1))

        self.w = np.zeros((n_dmps, n_bfs))  # weights for forcing term
        self.psi_centers = np.zeros(self.n_bfs)  # centers over canonical system for Gaussian basis functions
        self.psi_h = np.zeros(self.n_bfs)  # variance over canonical system for Gaussian basis functions

        # canonical system
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        self.timesteps = round(self.cs.run_time / self.dt)

        # generate centers for Gaussian basis functions
        self.generate_centers()

        # self.h = np.ones(self.n_bfs) * self.n_bfs / self.psi_centers # original
        self.h = np.ones(
            self.n_bfs) * self.n_bfs ** 1.5 / self.psi_centers / self.cs.alpha_x  # chose from trail and error

        # reset state
        self.reset_state()

    # Reset the system state
    def reset_state(self):
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def generate_centers(self):
        t_centers = np.linspace(0, self.cs.run_time, self.n_bfs)  # centers over time

        cs = self.cs
        x_track = cs.run()  # get all x over run time
        t_track = np.linspace(0, cs.run_time, cs.timesteps)  # get all time ticks over run time

        for n in range(len(t_centers)):
            for i, t in enumerate(t_track):
                if abs(t_centers[n] - t) <= cs.dt:  # find the x center corresponding to the time center
                    self.psi_centers[n] = x_track[i]

        return self.psi_centers

    def generate_psi(self, x):
        if isinstance(x, np.ndarray):
            x = x[:, None]

        self.psi = np.exp(-self.h * (x - self.psi_centers) ** 2)

        return self.psi

    def generate_weights(self, f_target):
        x_track = self.cs.run()
        psi_track = self.generate_psi(x_track)

        for d in range(self.n_dmps):
            # ------------ Original DMP in Schaal 2002
            # delta = self.goal[d] - self.y0[d]

            # ------------ Modified DMP in Schaal 2008
            delta = 1.0
            self.delta[d] = self.goal[d] - self.y0[d]

            for b in range(self.n_bfs):
                # as both number and denom has x(g-y_0) term, thus we can simplify the calculation process
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                denom = np.sum(x_track ** 2 * psi_track[:, b])
                # numer = np.sum(psi_track[:,b] * f_target[:,d]) # the simpler calculation
                # denom = np.sum(x_track * psi_track[:,b])
                # self.w[d, b] = numer / (denom*delta)

                self.w[d, b] = numer / denom
                if abs(delta) > 1e-6:
                    self.w[d, b] = self.w[d, b] / delta

        self.w = np.nan_to_num(self.w)
        # print(self.w.shape)

        return self.w

    def learning(self, y_demo, plot=False):
        if y_demo.ndim == 1:  # data is with only one dimension
            y_demo = y_demo.reshape(1, len(y_demo))

        self.y0 = y_demo[:, 0].copy()
        self.goal = y_demo[:, -1].copy()
        self.y_demo = y_demo.copy()

        # interpolate the demonstrated trajectory to be the same length with timesteps
        x = np.linspace(0, self.cs.run_time, y_demo.shape[1])
        y = np.zeros((self.n_dmps, self.timesteps))
        for d in range(self.n_dmps):
            y_tmp = interp1d(x, y_demo[d])
            for t in range(self.timesteps):
                y[d, t] = y_tmp(t * self.dt)

        # calculate velocity and acceleration of y_demo

        # method 1: using gradient
        dy_demo = np.gradient(y, axis=1) / self.dt
        ddy_demo = np.gradient(dy_demo, axis=1) / self.dt

        # method 2: using diff
        # dy_demo = np.diff(y) / self.dt
        # # let the first gradient same as the second gradient
        # dy_demo = np.hstack((np.zeros((self.n_dmps, 1)), dy_demo)) # Not sure if is it a bug?
        # # dy_demo = np.hstack((dy_demo[:,0].reshape(self.n_dmps, 1), dy_demo))

        # ddy_demo = np.diff(dy_demo) / self.dt
        # # let the first gradient same as the second gradient
        # ddy_demo = np.hstack((np.zeros((self.n_dmps, 1)), ddy_demo))
        # # ddy_demo = np.hstack((ddy_demo[:,0].reshape(self.n_dmps, 1), ddy_demo))

        x_track = self.cs.run()
        f_target = np.zeros((y_demo.shape[1], self.n_dmps))
        for d in range(self.n_dmps):
            # ---------- Original DMP in Schaal 2002
            # f_target[:,d] = ddy_demo[d] - self.alpha_y[d]*(self.beta_y[d]*(self.goal[d] - y_demo[d]) - dy_demo[d])

            # ---------- Modified DMP in Schaal 2008, fixed the problem of g-y_0 -> 0
            k = self.alpha_y[d]
            f_target[:, d] = (ddy_demo[d] - self.alpha_y[d] * (
                        self.beta_y[d] * (self.goal[d] - y_demo[d]) - dy_demo[d])) / k + x_track * (
                                         self.goal[d] - self.y0[d])

        self.generate_weights(f_target)

        if plot is True:
            # plot the basis function activations
            plt.figure()
            plt.subplot(211)
            psi_track = self.generate_psi(self.cs.run())
            plt.plot(psi_track)
            plt.title('basis functions')

            # plot the desired forcing function vs approx
            plt.subplot(212)
            plt.plot(f_target[:, 0])
            plt.plot(np.sum(psi_track * self.w[0], axis=1) * self.dt)
            plt.legend(['f_target', 'w*psi'])
            plt.title('DMP forcing function')
            plt.tight_layout()
            plt.show()

        # reset state
        self.reset_state()

    def reproduce(self, tau=None, initial=None, goal=None):
        # set temporal scaling
        if tau == None:
            timesteps = self.timesteps
        else:
            timesteps = round(self.timesteps / tau)

        # set initial state
        if initial != None:
            self.y0 = initial

        # set goal state
        if goal != None:
            self.goal = goal

        # reset state
        self.reset_state()

        y_reproduce = np.zeros((timesteps, self.n_dmps))
        dy_reproduce = np.zeros((timesteps, self.n_dmps))
        ddy_reproduce = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):
            y_reproduce[t], dy_reproduce[t], ddy_reproduce[t] = self.step(tau=tau)

        return y_reproduce.T, dy_reproduce.T, ddy_reproduce.T

    def step(self, tau=None):
        # run canonical system
        if tau == None:
            tau = self.tau
        x = self.cs.step_discrete(tau)

        # generate basis function activation
        psi = self.generate_psi(x)

        for d in range(self.n_dmps):
            # generate forcing term
            # ------------ Original DMP in Schaal 2002
            # f = np.dot(psi, self.w[d])*x*(self.goal[d] - self.y0[d]) / np.sum(psi)

            # ---------- Modified DMP in Schaal 2008, fixed the problem of g-y_0 -> 0
            # k = self.alpha_y[d]
            # f = k * (np.dot(psi, self.w[d]) * x / np.sum(psi)) - k * (self.goal[d] - self.y0[d]) * x # Modified DMP

            # ---------- Modified DMP with a simple solution to overcome the drawbacks of trajectory reproduction
            k = self.alpha_y[d]

            self.delta_2[d] = self.goal[d] - self.y0[d]  # Modified DMP extended
            if abs(self.delta[d]) > 1e-5:
                k2 = self.delta_2[d] / self.delta[d]
            else:
                k2 = 1.0

            f = k * (np.dot(psi, self.w[d]) * x * k2 / np.sum(psi)) - k * (self.goal[d] - self.y0[d]) * x

            # generate reproduced trajectory
            self.ddy[d] = self.alpha_y[d] * (self.beta_y[d] * (self.goal[d] - self.y[d]) - self.dy[d]) + f
            self.dy[d] += tau * self.ddy[d] * self.dt
            self.y[d] += tau * self.dy[d] * self.dt

        return self.y, self.dy, self.ddy


class dmp_discrete_dyn_weight():
    def __init__(self, n_dmps=1, n_bfs=100, dt=0, alpha_y=None, beta_y=None, **kwargs):
        self.n_dmps = n_dmps  # number of data dimensions, one dmp for one degree
        self.n_bfs = n_bfs  # number of basis functions
        self.dt = dt

        self.y0 = np.zeros(n_dmps)  # for multiple dimensions
        self.goal = np.ones(n_dmps)  # for multiple dimensions

        alpha_y_tmp = 60 if alpha_y is None else alpha_y
        beta_y_tmp = alpha_y_tmp / 4.0 if beta_y is None else beta_y
        self.alpha_y = np.ones(n_dmps) * alpha_y_tmp
        self.beta_y = np.ones(n_dmps) * beta_y_tmp
        self.tau = 1.0
        self.delta = np.ones((n_dmps, 1))
        self.delta_2 = np.ones((n_dmps, 1))

        self.w = np.zeros((n_dmps, n_bfs))  # weights for forcing term
        self.psi_centers = np.zeros(self.n_bfs)  # centers over canonical system for Gaussian basis functions
        self.psi_h = np.zeros(self.n_bfs)  # variance over canonical system for Gaussian basis functions

        # canonical system
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        self.timesteps = round(self.cs.run_time / self.dt)

        # generate centers for Gaussian basis functions
        self.generate_centers()

        # self.h = np.ones(self.n_bfs) * self.n_bfs / self.psi_centers # original
        self.h = np.ones(
            self.n_bfs) * self.n_bfs ** 1.5 / self.psi_centers / self.cs.alpha_x  # chose from trail and error

        # reset state
        self.reset_state()

    # Reset the system state
    def reset_state(self):
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def generate_centers(self):
        t_centers = np.linspace(0, self.cs.run_time, self.n_bfs)  # centers over time

        cs = self.cs
        x_track = cs.run()  # get all x over run time
        t_track = np.linspace(0, cs.run_time, cs.timesteps)  # get all time ticks over run time

        for n in range(len(t_centers)):
            for i, t in enumerate(t_track):
                if abs(t_centers[n] - t) <= cs.dt:  # find the x center corresponding to the time center
                    self.psi_centers[n] = x_track[i]

        return self.psi_centers

    def generate_psi(self, x):
        if isinstance(x, np.ndarray):
            x = x[:, None]

        self.psi = np.exp(-self.h * (x - self.psi_centers) ** 2)

        return self.psi

    def generate_weights(self, f_target):
        x_track = self.cs.run()
        psi_track = self.generate_psi(x_track)

        for d in range(self.n_dmps):
            # ------------ Original DMP in Schaal 2002
            # delta = self.goal[d] - self.y0[d]

            # ------------ Modified DMP in Schaal 2008
            delta = 1.0
            self.delta[d] = self.goal[d] - self.y0[d]

            for b in range(self.n_bfs):
                # as both number and denom has x(g-y_0) term, thus we can simplify the calculation process
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                denom = np.sum(x_track ** 2 * psi_track[:, b])
                # numer = np.sum(psi_track[:,b] * f_target[:,d]) # the simpler calculation
                # denom = np.sum(x_track * psi_track[:,b])
                # self.w[d, b] = numer / (denom*delta)

                self.w[d, b] = numer / denom
                if abs(delta) > 1e-6:
                    self.w[d, b] = self.w[d, b] / delta

        self.w = np.nan_to_num(self.w)
        # print(self.w.shape)

        return self.w

    def learning(self, y_demo, plot=False):
        if y_demo.ndim == 1:  # data is with only one dimension
            y_demo = y_demo.reshape(1, len(y_demo))

        self.y0 = y_demo[:, 0].copy()
        self.goal = y_demo[:, -1].copy()
        self.y_demo = y_demo.copy()

        # interpolate the demonstrated trajectory to be the same length with timesteps
        x = np.linspace(0, self.cs.run_time, y_demo.shape[1])
        y = np.zeros((self.n_dmps, self.timesteps))
        for d in range(self.n_dmps):
            y_tmp = interp1d(x, y_demo[d])
            for t in range(self.timesteps):
                y[d, t] = y_tmp(t * self.dt)

        # calculate velocity and acceleration of y_demo

        # method 1: using gradient
        dy_demo = np.gradient(y, axis=1) / self.dt
        ddy_demo = np.gradient(dy_demo, axis=1) / self.dt

        # method 2: using diff
        # dy_demo = np.diff(y) / self.dt
        # # let the first gradient same as the second gradient
        # dy_demo = np.hstack((np.zeros((self.n_dmps, 1)), dy_demo)) # Not sure if is it a bug?
        # # dy_demo = np.hstack((dy_demo[:,0].reshape(self.n_dmps, 1), dy_demo))

        # ddy_demo = np.diff(dy_demo) / self.dt
        # # let the first gradient same as the second gradient
        # ddy_demo = np.hstack((np.zeros((self.n_dmps, 1)), ddy_demo))
        # # ddy_demo = np.hstack((ddy_demo[:,0].reshape(self.n_dmps, 1), ddy_demo))

        x_track = self.cs.run()
        f_target = np.zeros((y_demo.shape[1], self.n_dmps))
        for d in range(self.n_dmps):
            # ---------- Original DMP in Schaal 2002
            # f_target[:,d] = ddy_demo[d] - self.alpha_y[d]*(self.beta_y[d]*(self.goal[d] - y_demo[d]) - dy_demo[d])

            # ---------- Modified DMP in Schaal 2008, fixed the problem of g-y_0 -> 0
            k = self.alpha_y[d]
            f_target[:, d] = (ddy_demo[d] - self.alpha_y[d] * (
                        self.beta_y[d] * (self.goal[d] - y_demo[d]) - dy_demo[d])) / k + x_track * (
                                         self.goal[d] - self.y0[d])

        self.generate_weights(f_target)

        if plot is True:
            # plot the basis function activations
            plt.figure()
            plt.subplot(211)
            psi_track = self.generate_psi(self.cs.run())
            plt.plot(psi_track)
            plt.title('basis functions')

            # plot the desired forcing function vs approx
            plt.subplot(212)
            plt.plot(f_target[:, 0])
            plt.plot(np.sum(psi_track * self.w[0], axis=1) * self.dt)
            plt.legend(['f_target', 'w*psi'])
            plt.title('DMP forcing function')
            plt.tight_layout()
            plt.show()

        # reset state
        self.reset_state()

        # get min and range for normalization and inverse normalization
        self.norm_min = self.w.min(axis=1)
        self.norm_range = self.w.max(axis=1) - self.norm_min

    def reproduce(self, dyn_w_gate=False, dyn_w=None, norm_range_max=1, norm_range_min=0, tau=None, initial=None, goal=None):
        # set temporal scaling
        if tau == None:
            timesteps = self.timesteps
        else:
            timesteps = round(self.timesteps / tau)

        # set initial state

        if initial is not None:
            self.y0 = initial

        # set goal state
        if goal is not None:
            self.goal = goal

        # reset state
        self.reset_state()

        y_reproduce = np.zeros((timesteps, self.n_dmps))
        dy_reproduce = np.zeros((timesteps, self.n_dmps))
        ddy_reproduce = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):
            y_reproduce[t], dy_reproduce[t], ddy_reproduce[t] = self.step(tau=tau,
                                                                          dyn_w_gate=dyn_w_gate,
                                                                          dyn_w=dyn_w,
                                                                          norm_range_max=norm_range_max,
                                                                          norm_range_min=norm_range_min)

        return y_reproduce.T, dy_reproduce.T, ddy_reproduce.T

    def step(self, dyn_w_gate=False, dyn_w=None, norm_range_max=1, norm_range_min=0, tau=None):
        # run canonical system
        if tau == None:
            tau = self.tau
        x = self.cs.step_discrete(tau)

        # generate basis function activation
        psi = self.generate_psi(x)

        for d in range(self.n_dmps):
            # generate forcing term
            # ------------ Original DMP in Schaal 2002
            # f = np.dot(psi, self.w[d])*x*(self.goal[d] - self.y0[d]) / np.sum(psi)

            # ---------- Modified DMP in Schaal 2008, fixed the problem of g-y_0 -> 0
            # k = self.alpha_y[d]
            # f = k * (np.dot(psi, self.w[d]) * x / np.sum(psi)) - k * (self.goal[d] - self.y0[d]) * x # Modified DMP

            # ---------- Modified DMP with a simple solution to overcome the drawbacks of trajectory reproduction
            k = self.alpha_y[d]

            self.delta_2[d] = self.goal[d] - self.y0[d]  # Modified DMP extended
            if abs(self.delta[d]) > 1e-5:
                k2 = self.delta_2[d] / self.delta[d]
            else:
                k2 = 1.0
            if dyn_w_gate is True and dyn_w is not None:
                dyn_w = _inv_normalization(dyn_w,
                                           self.norm_range,
                                           self.norm_min,
                                           range_max=norm_range_max,
                                           range_min=norm_range_min)
                assert self.w.shape == dyn_w.shape
                f = k * (np.dot(psi, dyn_w[d]) * x * k2 / np.sum(psi)) - k * (self.goal[d] - self.y0[d]) * x
            else:
                f = k * (np.dot(psi, self.w[d]) * x * k2 / np.sum(psi)) - k * (self.goal[d] - self.y0[d]) * x

            # generate reproduced trajectory
            self.ddy[d] = self.alpha_y[d] * (self.beta_y[d] * (self.goal[d] - self.y[d]) - self.dy[d]) + f
            self.dy[d] += tau * self.ddy[d] * self.dt
            self.y[d] += tau * self.dy[d] * self.dt

        return self.y, self.dy, self.ddy

# %% test code
if __name__ == "__main__":
    from gymnasium_envs.utils import interp_preprocessed_data_with_vel
    import os
    local_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
    data_path = local_path + '../datasets/reach/20241204105455.npz'
    ee_pos, ee_rot, ee_posvel, ee_rotvel, ee_quat, eeft = interp_preprocessed_data_with_vel(data_path)
    y_demo = ee_pos
    # # DMPs learning
    dmp = dmp_discrete_dyn_weight(n_dmps=y_demo.shape[0], n_bfs=20, dt=1.0 / y_demo.shape[1])
    dmp.learning(y_demo, plot=True)
    y_reproduce, dy_reproduce, ddy_reproduce = dmp.reproduce(initial=[0,0,0], goal=[-0.1, 0, 0.3])
    plt.plot(y_demo[2, :], 'g', label='demo')
    plt.plot(y_reproduce[2, :], 'r--', label='reproduce')
    plt.show()

    # data_len = 100
    #
    # # ----------------- For different initial and goal positions
    # t = np.linspace(0, 1.5 * np.pi, data_len)
    # y_demo = np.zeros((2, data_len))
    # y_demo[0, :] = np.sin(t)
    # y_demo[1, :] = 1
    #
    # # DMP learning
    # dmp = dmp_discrete(n_dmps=y_demo.shape[0], n_bfs=50, dt=1.0 / data_len)
    # dmp.learning(y_demo, plot=True)
    # current_tiem = time.time()
    # # reproduce learned trajectory
    # y_reproduce, dy_reproduce, ddy_reproduce = dmp.reproduce()
    # # print(time.time() - current_tiem)
    # # set new initial and goal poisitions
    # y_reproduce_2, dy_reproduce_2, ddy_reproduce_2 = dmp.reproduce(tau=0.5, initial=[0.2, -2], goal=[0, 0])
    #
    # plt.figure(figsize=(10, 5))
    # plt.plot(y_demo[0, :], 'g', label='demo sine')
    # plt.plot(y_reproduce[:, 0], 'r--', label='reproduce sine')
    # plt.plot(y_reproduce_2[:, 0], 'r-.', label='reproduce 2 sine')
    # plt.plot(y_demo[1, :], 'b', label='demo cosine')
    # plt.plot(y_reproduce[:, 1], 'm--', label='reproduce cosine')
    # plt.plot(y_reproduce_2[:, 1], 'm-.', label='reproduce 2 cosine')
    # plt.legend()
    # plt.grid()
    # plt.xlabel('time')
    # plt.ylabel('y')
    #
    # #
    # # # ----------------- For same initial and goal positions
    # # t = np.linspace(0, 2*np.pi, data_len)
    # #
    # # y_demo = np.zeros((2, data_len))
    # # y_demo[0,:] = np.sin(t)
    # # y_demo[1,:] = np.cos(t)
    # #
    # # # DMP learning
    # # dmp = dmp_discrete(n_dmps=y_demo.shape[0], n_bfs=400, dt=1.0/data_len)
    # # dmp.learning(y_demo, plot=False)
    # #
    # # # reproduce learned trajectory
    # # y_reproduce, dy_reproduce, ddy_reproduce = dmp.reproduce()
    # #
    # # # set new initial and goal poisitions
    # # y_reproduce_2, dy_reproduce_2, ddy_reproduce_2 = dmp.reproduce(tau=0.8, initial=[0.2, 0.8], goal=[0.5, 1.0])
    # #
    # # plt.figure(figsize=(10, 5))
    # # plt.plot(y_demo[0,:], 'g', label='demo sine')
    # # plt.plot(y_reproduce[:,0], 'r--', label='reproduce sine')
    # # plt.plot(y_reproduce_2[:,0], 'r-.', label='reproduce 2 sine')
    # # plt.plot(y_demo[1,:], 'b', label='demo cosine')
    # # plt.plot(y_reproduce[:,1], 'm--', label='reproduce cosine')
    # # plt.plot(y_reproduce_2[:,1], 'm-.', label='reproduce 2 cosine')
    # # plt.legend(loc="upper right")
    # # plt.ylim(-1.5, 3)
    # # plt.grid()
    # # plt.xlabel('time')
    # # plt.ylabel('y')
    #
    # plt.show()
