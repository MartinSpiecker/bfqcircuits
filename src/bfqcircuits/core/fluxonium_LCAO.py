#  Copyright (c) 2024. Martin Spiecker. All rights reserved.

import numpy as np
import scipy as sc

from bfqcircuits.core.fluxonium import Fluxonium
from bfqcircuits.utils import special as sp


class FluxoniumLCAO(Fluxonium):

    def __init__(self):

        super().__init__()

        self.wj = 0.0
        self.n_max = -1
        self.max_dist = -1

        self.subset = None
        self.S = None
        self.S_sweep = None

    #############################
    #####  sweep parameters #####
    #############################

    def convergence_sweep(self, N_max):
        """
        Convergences of the wave functions and corresponding energies with the number of basis states.

        :param N_max: maximum number of basis states to be used.
        :return: None.
        """

        self.S_sweep = np.zeros((N_max, N_max, N_max))
        super().convergence_sweep(N_max)

    def inspect_sweep(self, step):
        """
        Reset fluxonium to a specific sweep index. The routine assumes that the sweep has already been performed.

        :param: step: step of the sweep.
        :return: None
        """

        super().inspect_sweep(step)

        self.wj = np.sqrt(self.Ec * self.Ej)
        self.S = self.S_sweep[:, :, step]

    def _initialize_sweep(self, par_sweep):
        """
        Private routine: creates the data storage for the parameter sweep

        :param par_sweep: 1D-numpy array
        :return: None
        """

        super()._initialize_sweep(par_sweep)
        self.S_sweep = np.zeros((self.N, self.N, self.steps))

    def _calc_sweep(self, step):
        """
        Private routine that diagonalizes the Hamiltonian and stores the data of the parameter sweep.

        :param step: step of the parameter sweep
        :return: None
        """

        super()._calc_sweep(step)
        self.S_sweep[:self.N, :self.N, step] = self.S

    def _init_look_up_table(self, N):
        self.n_max = np.max(self.states[:N, 1])
        self.max_dist = np.max(self.states[:N, 0]) - np.min(self.states[:N, 0])
        self.special_integrals.init_look_up_table_fluxoniumLCAO(self.n_max, self.max_dist)

    ############################
    #####  sweep analysis  #####
    ############################

    def calc_dipole_moments(self, state1, state2):
        """
        Calculation of the flux and charge dipole moments between two states.

        :param state1: first state
        :param state2: second state
        :return: absolute flux dipole moment, absolute charge dipole moment
        """

        flux_dm = 0.0
        charge_dm = 0.0

        beta = np.sqrt(np.sqrt(self.Ej / self.Ec))

        self._init_look_up_table(self.N)

        for m in range(self.N):

            g = self.states[m, 0]
            i = self.states[m, 1]

            for n in range(self.N):

                h = self.states[n, 0]
                j = self.states[n, 1]

                xg = 2 * g * np.pi - self.p_ext
                xh = 2 * h * np.pi - self.p_ext

                d = beta * (xh - xg)
                s = beta * (xg + xh) / 2

                dist = h - g + self.max_dist

                flux_dm += self.v[m, state1] * self.v[n, state2] * (
                       self.special_integrals.x_shift_integral(i, j, dist, d) +
                       s * self.special_integrals.overlap_shift_integral(i, j, dist, d))

                if j > 0:
                    a = np.sqrt(2 * j) * self.special_integrals.overlap_shift_integral(i, j - 1, dist, d)
                else:
                    a = 0.0
                if i > 0:
                    b = np.sqrt(2 * i) * self.special_integrals.overlap_shift_integral(i - 1, j, dist, d)
                else:
                    b = 0.0

                charge_dm += 0.5 * self.v[m, state1] * self.v[n, state2] * (
                        (a - b) + d * self.special_integrals.overlap_shift_integral(i, j, dist, d))

        self._destroy_look_up_table()

        return np.abs(flux_dm) / (2 * np.pi * beta), beta * np.abs(charge_dm)

    def calc_sin_phi_over_two(self, state1, state2):
        """
        Calculation of the sin(phi/2) matrix element between two states.

        :param state1: first state
        :param state2: second state
        :return: sin(phi/2) matrix element
        """

        mel = 0.0

        beta = np.sqrt(np.sqrt(self.Ej / self.Ec))
        b = 1 / (2 * beta)

        self._init_look_up_table(self.N)

        for m in range(self.N):

            g = self.states[m, 0]
            i = self.states[m, 1]

            for n in range(self.N):

                h = self.states[n, 0]
                j = self.states[n, 1]

                xg = 2 * g * np.pi - self.p_ext
                xh = 2 * h * np.pi - self.p_ext

                d = beta * (xh - xg)
                s = beta * (xg + xh) / 2

                dist = h - g + self.max_dist

                mel += self.v[m, state1] * self.v[n, state2] * (
                    np.cos(b * s + self.p_ext / 2) * self.special_integrals.sin_shift_integral(i, j, dist, d, b) +
                    np.sin(b * s + self.p_ext / 2) * self.special_integrals.cos_shift_integral(i, j, dist, d, b))

        self._destroy_look_up_table()

        return np.abs(mel)

    ###################
    #####  plots  #####
    ###################

    def plot_states(self, ax, unit_mass=False):
        """
        Plot of potential and basis states.

        :param ax: matplotlib axes instance
        :param unit_mass: if False the flux potential is plotted in units of Phi_0.
                          if True the potential is transformed to correspond to a particle of unit mass and is plotted
                          in units of 1 / sqrt(GHz).
        :return: None
        """

        if not unit_mass:

            u = 2 * np.pi
            beta = u * np.sqrt(np.sqrt(self.Ej / self.Ec))

        else:

            u = np.sqrt(self.Ec)
            beta = np.sqrt(self.wj)

        sample = 1001
        x_range = (max(np.abs(2 * np.pi * self.states[:, 0] - self.p_ext)) + np.pi) / u
        x = np.linspace(-x_range, x_range, sample)

        ax_potential = ax.twinx()
        if not unit_mass:

            y = 4 * np.pi**2 * self.El * x**2 / 2 - self.Ej * np.cos(u * x + self.p_ext)
            ax_potential.plot(x, y, color='k', linewidth=1.0)

        else:
            y = (self.w * x)**2 / 2 - self.Ej * np.cos(u * x + self.p_ext)
            ax_potential.plot(x, y, color='k', linewidth=1.0)

        ax_potential.set_ylabel(r"$E$ (GHz)")

        # plot lowest n functions
        for i in range(self.states.shape[0]):

            m = self.states[i, 0]  # position
            k = self.states[i, 1]  # excitation

            phi_m = 2 * np.pi * m - self.p_ext
            xc = np.linspace((phi_m - 0.8 * np.pi) / u, (phi_m + 0.8 * np.pi) / u, 101)

            y = 0.5 * sp.norm_hermite(k, beta * (xc - (2 * np.pi * m - self.p_ext) / u)) + k

            ax.plot(xc, y, color=self.colors[i % 10], linewidth=1.0)

        if not unit_mass:
            ax.set_xlabel(r"$\phi$ ($\Phi_0$)")
        else:
            ax.set_xlabel(r"$x_\mathrm{r}$ $(1 / \sqrt{GHz})$")
        ax.set_ylabel(r"excitation")

    def plot_fluxonium(self, ax, n, x_range=5.0, unit_mass=False, fill_between=True, scale=1.0):
        """
        Plot of the potential and wave functions.

        :param ax: matplotlib axes instance
        :param n: number of lower states that will be plotted
        :param x_range: x-axis range that will be plotted in units depending on unit_mass.
        :param unit_mass: if False the flux potential is plotted in units of Phi_0.
                          if True the potential is transformed to correspond to a particle of unit mass and is plotted
                          in units of 1 / sqrt(GHz).
        :param fill_between: fill wave functions
        :param scale: scale the wave functions.
        :return: None
        """

        sample = 1001
        x = np.linspace(-x_range, x_range, sample)

        beta = 0.0
        if not unit_mass:

            u = 2 * np.pi
            beta = u * np.sqrt(np.sqrt(self.Ej / self.Ec))

            y = 4 * np.pi ** 2 * self.El * x ** 2 / 2 - self.Ej * np.cos(u * x + self.p_ext)
            ax.plot(x, y, color='k', linewidth=1.0, zorder=0)

        else:

            u = np.sqrt(self.Ec)
            beta = np.sqrt(self.wj)

            y = (self.w * x) ** 2 / 2 - self.Ej * np.cos(u * x + self.p_ext)
            ax.plot(x, y, color='k', linewidth=1.0, zorder=0)

        # plot lowest n functions
        for i in range(0, n):

            y = np.zeros_like(x)

            for j in range(0, self.N):

                m = self.states[j, 0]  # position
                k = self.states[j, 1]  # excitation

                # integral of wave function corresponds to hbar * w. For fine adjustments amp  can be used
                y += scale * self.v[j, i] * self.w * np.sqrt(beta) * sp.norm_hermite(k, beta * (x - (2 * m * np.pi -
                                                                                                     self.p_ext) / u))
            y += self.E[i]

            if not fill_between:
                ax.plot(x, y, color=self.colors[i % 10], linewidth=1.0, zorder=1)
            else:
                ax.fill_between(x, y, self.E[i], np.ones_like(y),
                                color=self.colors[i % 10], alpha=0.5, linewidth=1.0, zorder=1)

        if not unit_mass:
            ax.set_xlabel(r"$\phi$ ($\Phi_0$)")
        else:
            ax.set_xlabel(r"$x_\mathrm{r}$ $(1 / \sqrt{GHz})$")
        ax.set_ylabel(r"$E$ (GHz)")

    ##################
    #####  core  #####
    ##################

    def set_states(self, states):
        """
        Set list of local basis states.

        :param states: 2D numpy array - [[well number, excitation], ...]
        :return: None
        """

        self.states = states
        self.N = self.states.shape[0]
        self.n_max = np.max(self.states[:, 1])
        self.max_dist = np.max(self.states[:, 0]) - np.min(self.states[:, 0])

    def create_states(self, state_list):
        """
        Helper routine to create the local basis states.

        :param state_list: list of tuples e.g. [(-5, 5), (-2, 2), (-1, 1)], which creates local ground states
                           in the wells from -5 to 5, local excited states in the wells from -2 to 2, etc.
        :return: None
        """

        k = 0
        states = []
        for lst in state_list:
            for i in range(lst[0], lst[1] + 1):
                states.append([i, k])
            k += 1

        self.set_states(np.asarray(states))

    def diagonalize_hamiltonian(self):
        """
        Numerical diagonalization of the fluxonium Hamiltonian.
        For the solver "eigh" the tupel self.subset = (state_min, state_max) can be given to solve only
        the states within the subset.

        :return: None
        """

        self.H = np.zeros((self.N, self.N))
        self.S = np.zeros((self.N, self.N))

        beta = np.sqrt(np.sqrt(self.Ej / self.Ec))
        b = 1 / beta

        for m in range(self.N):

            g = self.states[m, 0]
            i = self.states[m, 1]

            for n in range(m + 1):

                h = self.states[n, 0]
                j = self.states[n, 1]

                xg = 2 * g * np.pi - self.p_ext
                xh = 2 * h * np.pi - self.p_ext

                d = beta * (xh - xg)
                s = beta * (xg + xh) / 2

                dist = h - g + self.max_dist  # for integral look up

                self.H[m, n] = (0.5 * (self.wj * (i + j + 1 - d**2 / 4) + self.w**2 / self.wj * s**2)
                                * self.special_integrals.overlap_shift_integral(i, j, dist, d)
                                + self.w**2 / self.wj * s * self.special_integrals.x_shift_integral(i, j, dist, d)
                                + 0.5 * (self.w**2 / self.wj - self.wj)
                                * self.special_integrals.x2_shift_integral(i, j, dist, d)
                                - (-1.0)**(g + h) * self.Ej
                                * self.special_integrals.cos_shift_integral(i, j, dist, d, b))

                self.S[m, n] = self.special_integrals.overlap_shift_integral(i, j, dist, d)

                if n < m:
                    self.H[n, m] = self.H[m, n]
                    self.S[n, m] = self.S[m, n]

        if self.subset is None:
            subset = [0, self.N - 1]
        else:
            subset = [self.subset[0], self.subset[1]]
        subset[1] = np.min((subset[1], self.N - 1))

        self.E = np.zeros(self.N)
        self.v = np.zeros((self.N, self.N))
        self.E[subset[0]:subset[1] + 1], self.v[:, subset[0]:subset[1] + 1] = sc.linalg.eigh(self.H, self.S,
                                                                                             type=1, driver="gvx",
                                                                                             subset_by_index=subset)

    def calc_hamiltonian_parameters(self):
        """
        Calculation of all relevant fluxonium parameters

        :return: None
        """

        super().calc_hamiltonian_parameters()

        # calculate plasma frequency
        self.wj = np.sqrt(self.Ec * self.Ej)

    def __repr__(self):
        """
        Returns a representation of the fluxonium parameters.

        :return: string
        """

        return (f"L = {self.L:.4e}\n"
                f"C = {self.C:.4e}\n"
                f"Ec = {self.Ec:.4e}\n"
                f"El = {self.El:.4e}\n"
                f"Ej = {self.Ej:.4e}\n"
                f"Ejs = {self.Ejs:.4e}\n"
                f"Ejd = {self.Ejd:.4e}\n"
                f"ℏω = {self.w:.4e}\n"
                f"ℏωᴊ = {self.wj:.4e}\n"
                f"u = {self.u:.4e}\n"
                f"Z = {self.Z:.4e}\n"
                f"flux_zpf = {self.flux_zpf:.4e}\n"
                f"charge_zpf = {self.charge_zpf:.4e}\n"
                )
