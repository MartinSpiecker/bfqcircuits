#  Copyright (c) 2024. Martin Spiecker. All rights reserved.

import numpy as np
import scipy as sc
import scipy.special as sp
import scipy.constants as pyc
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import schemdraw
import schemdraw.elements as elm

class Transmon:

    def __init__(self):

        # transmon parameters
        self.C = 0.0
        self.Ej = None
        self.Ec = 0.0
        self.w = 0.0
        self.ng = 0.0

        self.N = 0
        self.H = None
        self.E = None
        self.Eg = None
        self.v = None

        # sweeps
        self.steps = 0
        self.par_sweep = None
        self.transmon_pars_sweep = None
        self.H_sweep = None
        self.E_sweep = None
        self.Eg_sweep = None
        self.v_sweep = None

        # plots
        self.colors = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])

    def set_parameters(self, C=None, Ec=None, Ej=None, ng=None, N=None):
        """
        Helper routine to set relevant transmon parameters.

        :param C: capacitance in [F].
        :param Ec: charging energy in [GHz]. Note the definition Ec = 4e^2 / C.
        :param Ej: Josephson energy or list/array with higher harmonic Josephson energies in [GHz].
        :param ng: offset charge in [2e].
        :param N: number of basis states.
        :return: None.
        """

        if C is not None:
            self.C = C
            self.Ec = 4e-9 * pyc.e ** 2 / pyc.h / self.C
        elif Ec is not None:
            self.Ec = Ec
            self.C = 4e-9 * pyc.e ** 2 / pyc.h / self.Ec
        if Ej is not None:
            if hasattr(Ej, '__iter__'):
                self.Ej = np.asarray(Ej)
            else:
                self.Ej = np.array([Ej])
        if ng is not None:
            self.ng = ng
        if N is not None:
            self.N = N

        self.w = np.sqrt(self.Ej[0] * self.Ec)

    #############################
    #####  sweep parameters #####
    #############################

    def sweep_offset_charge(self, ng_sweep):
        """
        Sweep of offset charge.

        :param ng_sweep: 1D-numpy array with the offset charge values in [2e].
        :return: None.
        """

        self._initialize_sweep(ng_sweep)

        for i in range(self.steps):
            self.ng = self.par_sweep[i]
            self._calc_sweep(i)

    def sweep_parameter(self, par_sweep):
        """
        Sweep of a transmon parameters.

        :param par_sweep: dict with keys "C", "Ec", "Ej", "ng" and corresponding 1D numpy arrays.
        :return: None.
        """

        keys = list(par_sweep.keys())

        self.steps = par_sweep[keys[0]].shape[0]
        self._initialize_sweep(np.arange(self.steps))

        for i in range(self.steps):
            for key in keys:
                setattr(self, key, par_sweep[key][i])
                if key == "C":
                    self.Ec = 4e-9 * pyc.e ** 2 / pyc.h / self.C
                if key == "Ec":
                    self.C = 4e-9 * pyc.e ** 2 / pyc.h / self.Ec
                if key == "Ej":
                    if hasattr(self.Ej, '__iter__'):
                        self.Ej = np.asarray(self.Ej)
                    else:
                        self.Ej = np.array([self.Ej])

                self.w = np.sqrt(self.Ej[0] * self.Ec)

            self._calc_sweep(i)

    def convergence_sweep(self, N_max):
        """
        Convergences of the wave functions and corresponding energies with the number of basis states.

        :param N_max: maximum number of basis states to be used.
        :return: None
        """

        self.steps = N_max
        self.par_sweep = np.arange(1, N_max + 1)
        self.transmon_pars_sweep = np.zeros(self.steps, dtype=object)
        self.H_sweep = np.full((N_max, N_max, self.steps), np.nan)
        self.E_sweep = np.full((N_max, self.steps), np.nan)
        self.Eg_sweep = np.zeros(self.steps)
        self.v_sweep = np.full((N_max, N_max, self.steps), np.nan)

        for i in range(self.steps):
            self.N = self.par_sweep[i]
            self._calc_sweep(i)

    def inspect_sweep(self, step):
        """
        Reset transmon to a specific sweep index. The routine assumes that the sweep has already been performed.

        :param: step: step of the sweep.
        :return: None
        """

        self.C = self.transmon_pars_sweep[step].C
        self.Ec = self.transmon_pars_sweep[step].Ec
        self.Ej = self.transmon_pars_sweep[step].Ej
        self.ng = self.transmon_pars_sweep[step].ng
        self.w = self.transmon_pars_sweep[step].w
        self.N = self.transmon_pars_sweep[step].N
        self._operators()

        self.H = self.H_sweep[:, :, step]
        self.E = np.empty(self.N)
        self.E[:] = self.E_sweep[:, step]
        self.Eg = self.Eg_sweep[step]
        self.v = self.v_sweep[:, :, step]

    def _initialize_sweep(self, par_sweep):
        """
        Private routine that creates the data storage for the parameter sweep

        :param par_sweep: 1D-numpy array
        :return: None
        """

        self.steps = par_sweep.size
        self.par_sweep = par_sweep
        self.transmon_pars_sweep = np.zeros(self.steps, dtype=object)
        self.H_sweep = np.zeros((self.N, self.N, self.steps))
        self.E_sweep = np.zeros((self.N, self.steps))
        self.Eg_sweep = np.zeros(self.steps)
        self.v_sweep = np.zeros((self.N, self.N, self.steps))

    def _calc_sweep(self, step):
        """
        Private routine that diagonalizes the Hamiltonian and stores the data of the given sweep step.

        :param step: step of the parameter sweep
        :return: None
        """

        self.transmon_pars_sweep[step] = self.Parameters(self)
        self.diagonalize_hamiltonian()
        self.H_sweep[:self.N, :self.N, step] = self.H
        self.E_sweep[:self.N, step] = self.E
        self.Eg_sweep[step] = self.Eg
        self.v_sweep[:self.N, :self.N, step] = self.v

    ############################
    #####  analysis  #####
    ############################

    def substract_groundstate_energy(self):
        """
        Subtracts ground state energy from self.E.

        :return: None
        """

        self.E -= self.Eg

    def add_groundstate_energy(self):
        """
        Adds ground state energy to self.E.

        :return: None
        """

        self.E += self.Eg

    def substract_groundstate_energy_sweep(self):
        """
        Subtracts ground state energy from self.E_sweep.

        :return: None
        """

        self.E_sweep -= self.Eg_sweep

    def add_groundstate_energy_sweep(self):
        """
        Subtracts ground state energy from self.E_sweep.

        :return: None
        """

        self.E_sweep += self.Eg_sweep

    def calc_dipole_moments_sweep(self, state1, state2):
        """
        Calculation of the flux and charge dipole moments between two states as a function of the sweep parameter.

        :param state1: first state
        :param state2: second state
        :return: 1D numpy array with absolute flux dipole moments,
                 1D numpy array with absolute charge dipole moments
        """

        flux_dm_sweep = np.zeros(self.steps)
        charge_dm_sweep = np.zeros(self.steps)

        for i in range(self.steps):

            self.inspect_sweep(i)
            flux_dm_sweep[i], charge_dm_sweep[i] = self.calc_dipole_moments(state1, state2)

        return flux_dm_sweep, charge_dm_sweep

    def calc_dipole_moments(self, state1, state2):
        """
        Calculation of the flux and charge dipole moments between two states.

        :param state1: first state
        :param state2: second state
        :return: absolute flux dipole moment, absolute charge dipole moment
        """

        fdm = np.dot(self.v[:, state1], np.dot(self.flux_operator, self.v[:, state2]))
        cdm = np.sum(self.v[:, state1] * self.charge_operator * self.v[:, state2])

        return np.abs(fdm), np.abs(cdm)

    def calc_sin_phi_over_two_sweep(self, state1, state2):
        """
        Calculation of the sin(phi/2) matrix element between two states as a function of the sweep parameter.

        :param state1: first state
        :param state2: second state
        :return: 1D numpy array with sin(phi/2) matrix elements
        """

        mel_sweep = np.zeros(self.steps)

        for i in range(self.steps):

            self.inspect_sweep(i)
            mel_sweep[i] = self.calc_sin_phi_over_two(state1, state2)

        return mel_sweep

    def calc_sin_phi_over_two(self, state1, state2):
        """
        Calculation of the sin(phi/2) matrix element between two states.

        :param state1: first state
        :param state2: second state
        :return: sin(phi/2) matrix element
        """

        mel = np.dot(self.v[:, state1], np.dot(self.calc_sin_phi_over_operator, self.v[:, state2]))

        return np.abs(mel)

    ###################
    #####  plots  #####
    ###################

    def plot_transmon(self, ax, n, x_range=1.0, remove_ng=False, fill_between=True, scale=None):
        """
        Plot of the potential and wave functions.

        :param ax: matplotlib axes instance.
        :param n: number of lower states that will be plotted.
        :param x_range: x-axis range of the resonator that will be plotted in units of Phi_0.
        :param remove_ng: if True, complex valued plane wave from the offset charge is removed from the wave function.
                          If False, complex valued wave functions are plotted
        :param fill_between: fill wave functions
        :param scale: scale the amplitude of the wave functions
        :return: None
        """

        sample = 1001
        x = np.linspace(- x_range / 2, x_range / 2, sample)
        if self.Ej.size == 1:
            y = - self.Ej * np.cos(2 * np.pi * x)
        else:
            y = np.zeros_like(x)
            for i in range(self.Ej.size):
                y -= self.Ej[i] * np.cos((i + 1) * 2 * np.pi * x)
        ax.plot(x, y, 'k')

        if scale is None:
            scale = np.sqrt(self.w)  # integral of squared wavefunction defaults to hbar * w.

        for i in range(n - 1, -1, -1):  # plot lowest functions last looks better in case they overlap

            y = np.zeros_like(x, dtype="complex")

            for j in range(self.N):

                if not remove_ng:   # Wave functions as they actually are. They fullfill the periodic boundary condition.
                                    # However, they entail a complex valued plane wave with wave vector ng
                    y += (scale * self.v[j, i] / np.sqrt(2 * np.pi) *
                          np.exp(2j * np.pi * (j - (self.N - 1) // 2) * x))

                else:
                    y += (scale * self.v[j, i] / np.sqrt(2 * np.pi) *
                          np.exp(2j * np.pi * (j - (self.N - 1) // 2 - self.ng) * x))

            if not remove_ng:
                if fill_between:
                    ax.fill_between(x, self.E[i] + np.real(y), self.E[i],
                                    edgecolor=None, facecolor=self.colors[i], alpha=0.5)
                    ax.fill_between(x, self.E[i] + np.imag(y), self.E[i],
                                    edgecolor=None, facecolor=self.colors[i], alpha=0.5)
                ax.plot(x, self.E[i] + np.real(y), self.colors[i], ls="-")
                ax.plot(x, self.E[i] + np.imag(y), self.colors[i], ls="--")
            else:
                if np.sum(np.abs(np.real(y))) > np.sum(np.abs(np.imag(y))):
                    if fill_between:
                        ax.fill_between(x, self.E[i] + np.real(y), self.E[i],
                                        edgecolor=None, facecolor=self.colors[i], alpha=0.5)
                    ax.plot(x, self.E[i] + np.real(y), self.colors[i])
                else:
                    if fill_between:
                        ax.fill_between(x, self.E[i] + np.imag(y), self.E[i],
                                        edgecolor=None, facecolor=self.colors[i], alpha=0.5)
                    ax.plot(x, self.E[i] + np.imag(y), self.colors[i])

        ax.set_xlabel(r"$\phi$ ($\Phi_0$)")
        ax.set_ylabel("$E$ (GHz)")

    def plot_energy_sweep(self, ax, n):
        """
        Plot of the transmon energies as a function of the sweep parameter.

        :param ax: matplotlib axes instance.
        :param n: list/array of states that will be plotted.
        :return: None.
        """

        if not hasattr(n, '__iter__'):
            n = [n]

        for i in n:
            ax.plot(self.par_sweep, self.E_sweep[i, :], self.colors[i % 10])

        ax.set_xlabel("sweep parameter")
        ax.set_ylabel(r"$E$ (GHz)")

    def plot_convergence_sweep(self, ax, n):
        """
        Plot of energy convergence. This routine assumes that a convergence_test has been performed.

        :param ax: matplotlib axes instance.
        :param n: number of lower states that will be plotted.
        :return: None.
        """

        converged_energy = np.abs(self.E_sweep - self.E_sweep[:, -1, np.newaxis])

        for i in range(n):
            ax.plot(self.par_sweep, converged_energy[i, :], self.colors[i % 10])

        ax.set_xlabel("$N$")
        ax.set_ylabel(r"$\Delta E$ (GHz)")

    def plot_dipole_to_various_states_sweep(self, ax, ref_state, state_list, dipole="flux"):
        """
        3D plot of dipole moments as a function of the sweep parameter.

        :param ax: matplotlib axes instance with projection="3d".
        :param ref_state: state of interest.
        :param state_list: list or 1D numpy array of other states.
        :param dipole: either "flux" or "charge" for flux and charge dipole moments, respectively.
        :return: None.
        """

        dmax = 0.0

        for i in state_list:

            flux_dm, charge_dm = self.calc_dipole_moments_sweep(ref_state, i)

            if dipole == "flux":
                dm_sweep = flux_dm
            elif dipole == "charge":
                dm_sweep = charge_dm
            dmax = max((dmax, np.max(dm_sweep)))

            v = []
            for k in range(self.steps):
                v.append([self.par_sweep[k], self.E_sweep[i, k], np.abs(dm_sweep[k])])
            for k in range(self.steps - 1, -1, -1):
                v.append([self.par_sweep[k], self.E_sweep[i, k], 0.0])

            r, g, b = cl.to_rgb(self.colors[i % 10])
            poly3dCollection = Poly3DCollection([v], linewidths=0.0)
            poly3dCollection.set_facecolor((r, g, b, 0.5))

            ax.add_collection3d(poly3dCollection)

        ax.set_xlim(np.min(self.par_sweep), np.max(self.par_sweep))
        ax.set_ylim(np.min(self.E_sweep[state_list, :]), np.max(self.E_sweep[state_list, :]))
        ax.set_zlim(0, dmax)

        ax.set_xlabel("sweep parameter")
        ax.set_ylabel(r"$E$ (GHz)")
        if dipole == "flux":
            ax.set_zlabel(r"$|\langle \psi_m|\phi| \psi_n\rangle |$ ($\Phi_0$)", labelpad=8)
        elif dipole == "charge":
            ax.set_zlabel(r"$|\langle \psi_m|q| \psi_n\rangle |$ ($2e$)", labelpad=8)

    ##################
    #####  core  #####
    ##################

    def diagonalize_hamiltonian(self):
        """
        Numerical diagonalization of the transmon Hamiltonian.

        :return: None.
        """

        cs = np.arange(self.N) - (self.N - 1) // 2  # charge states

        diagonal = 0.5 * self.Ec * (cs - self.ng)**2

        if self.Ej.size == 1:
            off_diagonal = - 0.5 * self.Ej * np.ones(self.N - 1)
            self.H = np.diag(diagonal) - np.diag(off_diagonal, 1) - np.diag(off_diagonal, -1)
            self.E, self.v = sc.linalg.eigh_tridiagonal(diagonal, off_diagonal)
            self.Eg = self.E[0]
        else:
            self.H = np.diag(diagonal)
            for i in range(self.Ej.size):
                self.H -= 0.5 * (np.diag(np.full(self.N - 1 - i, self.Ej[i]), 1 + i) +
                                 np.diag(np.full(self.N - 1 - i, self.Ej[i]), - 1 - i))
            self.E, self.v = sc.linalg.eigh(self.H)
            self.Eg = self.E[0]

        self._operators()

    def energies_first_order_approx(self):
        """
        First order harmonic oscillator approximation of the transmon energies.

        :return: 1D numpy array with state energies in [GHz]
        """

        n = np.arange(self.N)
        E_approx = self.w * (n + 0.5) - self.Ej[0] - self.Ec / 16 * (n**2 + n + 0.5)

        return E_approx

    def charge_dispersion_approx(self, Ej_over_Ec, n):
        """
        Approximate formula for the energy charge dispersion.

        :param Ej_over_Ec: 1D numpy array with Ej / Ec values
        :param n: number of lower states.
        :return: 1D array with the energy charge dispersion in [GHz]
        """

        e = np.zeros((n, Ej_over_Ec.size))

        for i in range(n):
            x = np.sqrt(Ej_over_Ec)
            e[i, :] = 2**(5 * i + 4) / sp.factorial(i, exact=True) / np.sqrt(x / np.pi) * x**i * np.exp(- 8 * x)

        return e

    def _operators(self):
        """
        Private routine that creates the flux and charge dipole operators and the sin(phi / 2) operator.

        :return: None
        """

        cs = np.arange(self.N) - (self.N - 1) // 2  # charge states
        self.charge_operator = cs

        k = np.add.outer(-cs, cs)
        b = k != 0
        self.flux_operator = np.zeros((self.N, self.N))
        # We neglect in the following line the prefactor - 1j.
        # The factor 2 pi is needed to convert from the phase operator to the flux operator
        self.flux_operator[b] = (1 - 2 * (k[b] % 2)) / (2 * np.pi * k[b])

        self.calc_sin_phi_over_operator = 8 * k * (1 - 2 * (k % 2)) / (1 - 4 * k**2)

    def draw_circuit(self):
        """
        Creates a figure with the circuit.

        :return: matplotlib figure instance
        """

        figwidth = 4.0
        fig, ax = plt.subplots(1, 1, figsize=(figwidth, 2.2 / 4.0 * figwidth))
        ax.axis("off")

        d = schemdraw.Drawing(canvas=ax)
        d.config(unit=2.0, fontsize=15)
        d += elm.Capacitor().up().label("$C$")
        d += elm.Line().right()
        d += elm.Josephson().down().label(r"$E_\mathrm{J}$")
        d += elm.Line().left()

        d.draw()

        #ax.axvline(-1.0)
        #ax.axvline(3.0)
        #ax.axhline(-0.1)
        #ax.axhline(2.1)

        ax.set_xlim(-1.0, 3.0)
        ax.set_ylim(-0.1, 2.1)

        return fig

    def show_formulas(self):
        """
        Creates a figure showing the most relevant formulas.

        :return: matplotlib figure instance
        """

        figwidth = 9.0
        fig, ax = plt.subplots(1, 1, figsize=(figwidth, 4.0 / 9.0 * figwidth))
        ax.axis("off")

        ax.text(0.5, 2.0, "Transmon Hamiltonian:", va="top", fontsize=12)
        ax.text(0.75, 1.42, "$H$", va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(1.0, 1.5, r"$=\frac{1}{2C}(q - q_\mathrm{g})^2"
                           r" - E_\mathrm{J}\cos\left(\frac{2 \pi}{\Phi_0}\phi\right)$" + "\n" +
                           r"$ = \frac{1}{2} E_C (n - n_\mathrm{g})^2"
                           r" - E_\mathrm{J}\cos(\varphi)$" + "\n" +
                           r"$ = \frac{1}{2} (p - p_\mathrm{g})^2"
                           r" - E_\mathrm{J} + \frac{\hbar^2\omega^2}{2}x^2 - \dots, \quad \hbar\omega = \sqrt{E_\mathrm{J} E_C}$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        ax.text(0.5, -0.5, "The wave functions are subject to the periodic boundary condition:", va="top", fontsize=12)
        ax.text(0.75, -1.0, r"$\psi(\varphi + 2 \pi) = \psi(\varphi)$", va="top", fontsize=15, math_fontfamily='dejavuserif')

        #ax.axvline(0.0)
        #ax.axvline(9.0)
        #ax.axhline(-1.9)
        #ax.axhline(2.1)

        ax.set_xlim(0.0, 9.0)
        ax.set_ylim(- 1.9, 2.1)

        return fig

    def __repr__(self):
        """
        Returns a representation of the transmon parameters.

        :return: string
        """

        if self.Ej.size > 1:
            s = "["
            for i in range(self.Ej.size - 1):
                s += f"{self.Ej[i]:.4e},"
            s += f"{self.Ej[-1]:.4e}]"
        else:
            s = f"{self.Ej[0]:.4e},"

        return (f"C = {self.C:.4e}\n"
                f"Ec = {self.Ec:.4e}\n"
                f"Ej = {s}\n"
                f"ℏω = {self.w:.4e}\n"
                f"Ej / Ec = {np.sqrt(self.Ej[0] / self.Ec):.4e}\n"
                )

    ########################################
    ##### parameter storage for sweeps #####
    ########################################

    class Parameters:

        def __init__(self, transmon):

            self.C = transmon.C
            self.Ec = transmon.Ec
            self.Ej = transmon.Ej
            self.ng = transmon.ng
            self.w = transmon.w
            self.N = transmon.N








