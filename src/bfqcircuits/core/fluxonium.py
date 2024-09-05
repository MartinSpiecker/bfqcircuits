#  Copyright (c) 2024. Martin Spiecker. All rights reserved.

import numpy as np
import scipy as sc
import scipy.constants as pyc
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import schemdraw
import schemdraw.elements as elm

from bfqcircuits.utils import special as sp


class Fluxonium:

    def __init__(self):

        # fluxonium parameters
        self.L = 0.0
        self.C = 0.0

        self.Ec = 0.0
        self.El = 0.0
        self.Ej = 0.0
        self.Ejs = 0.0
        self.Ejd = 0.0
        self.ratio = 0.0
        self.p_ext = 0.0

        self.w = 0
        self.u = 0
        self.Z = 0.0
        self.flux_zpf = 0.0
        self.charge_zpf = 0.0

        self.N = 0
        self.sqrts = None
        self.H = None
        self.E = None
        self.Eg = 0.0
        self.v = None
        self.special_integrals = sp.SpecialIntegrals()

        # sweeps
        self.steps = 0
        self.par_sweep = None
        self.fluxonium_pars_sweep = None
        self.H_sweep = None
        self.E_sweep = None
        self.Eg_sweep = None
        self.v_sweep = None

        # plots
        self.colors = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])

    def set_parameters(self, L=None, C=None, Ej=None, Ejs=None, Ejd=None, ratio=None, p_ext=None, N=None):
        """
        Helper routine to set relevant fluxonium parameters.

        :param L: inductance in [H].
        :param C: capacitance in [F].
        :param Ej: Josephson energy in [GHz].
        :param Ejs: sum of the Josephson energies of the SQUID [GHz].
        :param Ejd: difference of the Josephson energies (outer junction  - inner junction) of the SQUID [GHz].
        :param ratio: ratio of the loop areas - fluxonium loop area (enclosed by the inner junction) / SQUID loop area.
        :param p_ext: external flux bias in [rad].
        :param N: number of basis states.
        :return: None.
        """

        if L is not None:
            self.L = L
        if C is not None:
            self.C = C
        if Ej is not None:
            self.Ej = Ej
        if Ejs is not None:
            self.Ejs = Ejs
        if Ejd is not None:
            self.Ejd = Ejd
        if ratio is not None:
            self.ratio = ratio
        if p_ext is not None:
            self.p_ext = p_ext
        if N is not None:
            self.N = N

    def set_flux_squid(self, p_ext):
        """
        Calculation of the effective Josephson energy and effective flux bias.

        :param p_ext: external flux bias in [rad].
        :return: None.
        """

        p_s = 2 * p_ext / (2 * self.ratio + 1.0)

        Ejs = self.Ejs * np.cos(p_s / 2)
        Ejd = self.Ejd * np.sin(p_s / 2)  # outer junction  - inner junction

        self.Ej = np.sign(Ejs) * np.sqrt(Ejs ** 2 + Ejd ** 2)
        self.p_ext = p_ext + np.arctan(Ejd / Ejs)

    #############################
    #####  sweep parameters #####
    #############################

    def sweep_external_flux(self, pext_sweep):
        """
        Sweep of externally applied magentic flux.

        :param pext_sweep: 1D-numpy array with the external flux bias in [rad].
        :return: None.
        """

        self._initialize_sweep(pext_sweep)

        self._init_look_up_table(self.N)
        for i in range(self.steps):
            self.p_ext = self.par_sweep[i]
            self._calc_sweep(i)
        self._destroy_look_up_table()

    def sweep_external_flux_squid(self, pext_sweep):
        """
        Sweep of externally applied global magentic flux to a SQUID-junction fluxonium.

        The routine assumes that sum of the SQUID-junction Josephson energies self.Ejs and asymmetry self.Ejd as well as
        the ratio of fluxonium loop area to SQUID-area (self.ratio) have been defined already.
        For further details see explanation of self.set_flux_squid().

        :param pext_sweep: 1D-numpy array with the external flux bias in [rad].
        :return: None.
        """

        self._initialize_sweep(pext_sweep)

        self._init_look_up_table(self.N)
        for i in range(self.steps):
            self.set_flux_squid(self.par_sweep[i])
            self._calc_sweep(i)
        self._destroy_look_up_table()

    def sweep_parameter(self, par_sweep, par_name=None):
        """
        Sweep of a fluxonium parameter specified by par_name.

        :param par_sweep:  either dict with keys "L", "C", "Ej", "Ejs", "Ejd", "ratio", "p_ext" and corresponding
                           1D numpy parameter arrays,
                           or 1D numpy array with parameter sweep for the parameter specified by par_name.
        :param par_name: "L", "C", "Ej", "Ejs", "Ejd", "ratio", "p_ext".
        :return: None.
        """

        if type(par_sweep) == dict:

            keys = list(par_sweep.keys())

            self.steps = par_sweep[keys[0]].size
            self._initialize_sweep(np.arange(self.steps))
            for i in range(self.steps):
                for key in keys:
                    setattr(self, key, par_sweep[key][i])
                self.calc_hamiltonian_parameters()
                if "Ejs" in keys or "Ejd" in keys or "ratio" in keys:
                    self.set_flux_squid(self.p_ext)
                self._calc_sweep(i)
        else:
            if par_name in ["L", "C"]:

                self._initialize_sweep(par_sweep)
                for i in range(self.steps):
                    setattr(self, par_name, self.par_sweep[i])
                    self.calc_hamiltonian_parameters()
                    self._calc_sweep(i)

            elif par_name in ["Ej", "p_ext"]:

                self._initialize_sweep(par_sweep)
                self._init_look_up_table(self.N)
                for i in range(self.steps):
                    setattr(self, par_name, self.par_sweep[i])
                    self._calc_sweep(i)
                self._destroy_look_up_table()

            elif par_name in ["Ejs", "Ejd", "ratio"]:

                self._initialize_sweep(par_sweep)
                self._init_look_up_table(self.N)
                for i in range(self.steps):
                    setattr(self, par_name, self.par_sweep[i])
                    self.set_flux_squid(self.p_ext)
                    self._calc_sweep(i)
                self._destroy_look_up_table()

            else:
                print("Parameter does not exist or should not be swept in this routine. \n" +
                      "For instance, for sweeping the number of basis functions use the convergence_sweep routine.")

    def convergence_sweep(self, N_max):
        """
        Convergences of the wave functions and corresponding energies with the number of basis states.

        :param N_max: maximum number of basis states to be used.
        :return: None.
        """

        self.par_sweep = np.arange(1, N_max + 1)

        self.steps = N_max
        self.fluxonium_pars_sweep = np.zeros(self.steps, dtype=object)
        self.H_sweep = np.full((N_max, N_max, self.steps), np.nan)
        self.E_sweep = np.full((N_max, self.steps), np.nan)
        self.Eg_sweep = np.empty(self.steps)
        self.v_sweep = np.full((N_max, N_max, self.steps), np.nan)

        self._init_look_up_table(N_max)
        for i in range(self.steps):
            self.N = self.par_sweep[i]
            self._calc_sweep(i)
        self._destroy_look_up_table()

    def inspect_sweep(self, step):
        """
        Reset fluxonium to a specific sweep index. The routine assumes that the sweep has already been performed.

        :param: step: step of the sweep.
        :return: None.
        """

        self.L = self.fluxonium_pars_sweep[step].L
        self.C = self.fluxonium_pars_sweep[step].C

        self.Ec = self.fluxonium_pars_sweep[step].Ec
        self.El = self.fluxonium_pars_sweep[step].El
        self.Ej = self.fluxonium_pars_sweep[step].Ej
        self.p_ext = self.fluxonium_pars_sweep[step].p_ext

        self.w = self.fluxonium_pars_sweep[step].w
        self.u = self.fluxonium_pars_sweep[step].u
        self.Z = self.fluxonium_pars_sweep[step].Z
        self.flux_zpf = self.fluxonium_pars_sweep[step].flux_zpf
        self.charge_zpf = self.fluxonium_pars_sweep[step].charge_zpf

        self.N = self.fluxonium_pars_sweep[step].N
        self.sqrts = np.sqrt(np.arange(1, self.N))

        self.H = self.H_sweep[:, :, step]
        self.E = np.empty(self.N)
        self.E[:] = self.E_sweep[:, step]
        self.Eg = self.Eg_sweep[step]
        self.v = self.v_sweep[:, :, step]

    def _initialize_sweep(self, par_sweep):
        """
        Private routine that creates the data storage for the parameter sweep.

        :param par_sweep: 1D-numpy array.
        :return: None.
        """

        self.steps = par_sweep.size
        self.par_sweep = par_sweep
        self.fluxonium_pars_sweep = np.zeros(self.steps, dtype=object)
        self.H_sweep = np.zeros((self.N, self.N, self.steps))
        self.E_sweep = np.zeros((self.N, self.steps))
        self.Eg_sweep = np.zeros(self.steps)
        self.v_sweep = np.zeros((self.N, self.N, self.steps))

    def _calc_sweep(self, step):
        """
        Private routine that diagonalizes the Hamiltonian and stores the data of the given sweep step.

        :param step: step of the parameter sweep.
        :return: None.
        """

        self.fluxonium_pars_sweep[step] = self.Parameters(self)
        self.diagonalize_hamiltonian()
        self.H_sweep[:self.N, :self.N, step] = self.H
        self.E_sweep[:self.N, step] = self.E
        self.Eg_sweep[step] = self.Eg
        self.v_sweep[:self.N, :self.N, step] = self.v

    def _init_look_up_table(self, N):
        self.special_integrals.init_look_up_table_fluxonium(N)

    def _destroy_look_up_table(self):
        self.special_integrals.destroy_look_up_table()

    ############################
    ##### analysis  #####
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

        :return: None.
        """

        self.E_sweep -= self.Eg_sweep

    def add_groundstate_energy_sweep(self):
        """
        Subtracts ground state energy from self.E_sweep.

        :return: None.
        """

        self.E_sweep += self.Eg_sweep

    def calc_inductive_loss_sweep(self, state_i, state_f, Q_ind, environment="TLSs", T=0.0):
        """
        Calculation of the inductive qubit loss between two states as a function of the sweep parameter.

        :param state_i: initial state.
        :param state_f: final state.
        :param Q_ind: quality factor of the inductor.
        :param environment: either "TLSs" or "Bosonic".
        :param T: Temperature of the environment.
        :return: 1D numpy array with G1 rates in [MHz].
        """

        G1 = np.empty(self.steps)

        for i in range(self.steps):

            self.inspect_sweep(i)
            G1[i] = self.calc_inductive_loss(state_i, state_f, Q_ind, environment, T)

        return G1

    def calc_capacitive_loss_sweep(self, state_i, state_f, Q_cap, environment="TLSs", T=0.0):
        """
        Calculation of the capacitive qubit loss between two states as a function of the sweep parameter.

        :param state_i: initial state.
        :param state_f: final state.
        :param Q_cap: quality factor of the capacitor.
        :param environment: either "TLSs" or "Bosonic".
        :param T: Temperature of the environment.
        :return: 1D numpy array with G1 rates in [MHz].
        """

        G1 = np.empty(self.steps)

        for i in range(self.steps):
            self.inspect_sweep(i)
            G1[i] = self.calc_capacitive_loss(state_i, state_f, Q_cap, environment, T)

        return G1

    def calc_inductive_loss(self, state_i, state_f, Q_ind, environment="TLSs", T=0.0):
        """
        Calculation of the inductive qubit loss between two states.

        :param state_i: initial state.
        :param state_f: final state.
        :param Q_ind: quality factor of the inductor.
        :param environment: either "TLSs" or "Bosonic".
        :param T: Temperature of the environment.
        :return: 1D numpy array with G1 rates in [MHz].
        """

        df = self.E[state_i] - self.E[state_f]
        flux_dm, _ = self.calc_dipole_moments(state_i, state_f)

        G1 = 16e3 * np.pi**3 * self.El / Q_ind * flux_dm**2

        if T > 0.0:
            if environment == "TLSs":
                pass
            elif environment == "Bosonic":
                G1 /= np.tanh(1e12 * pyc.h * df / (2 * pyc.k * T))
            elif environment == "Brownian":
                G1 *= 0.5 * (1 + 1 / np.tanh(1e12 * pyc.h * df / (2 * pyc.k * T)))

        return G1

    def calc_capacitive_loss(self, state_i, state_f, Q_cap, environment="TLSs", T=0.0):
        """
        Calculation of the capacitive qubit loss between two states.

        :param state_i: initial state.
        :param state_f: final state.
        :param Q_cap: quality factor of the inductor.
        :param environment: either "TLSs" or "Bosonic".
        :param T: Temperature of the environment.
        :return: 1D numpy array with G1 rates in [MHz].
        """

        df = self.E[state_i] - self.E[state_f]
        _, charge_dm = self.calc_dipole_moments(state_i, state_f)

        G1 = 4e3 * np.pi * self.Ec / Q_cap * charge_dm**2
        #G1 = 16e9 * np.pi**3 * df**2 / self.Ec / Q_cap * flux_dm ** 2  # Identical formula

        if T > 0.0:
            if environment == "TLSs":
                pass
            elif environment == "Bosonic":
                G1 /= np.tanh(1e12 * pyc.h * df / (2 * pyc.k * T))
            elif environment == "Brownian":
                G1 *= 0.5 * (1 + 1 / np.tanh(1e12 * pyc.h * df / (2 * pyc.k * T)))

        return G1

    def calc_dipole_moments_sweep(self, state1, state2):
        """
        Calculation of the flux and charge dipole moments between two states as a function of the sweep parameter.

        :param state1: first state.
        :param state2: second state.
        :return: None.
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

        :param state1: first state.
        :param state2: second state.
        :return: absolute flux dipole moment, absolute charge dipole moment.
        """

        a_daggar = np.sum(self.sqrts * self.v[1:, state1] * self.v[:-1, state2])
        a = np.sum(self.sqrts * self.v[:-1, state1] * self.v[1:, state2])

        return np.abs(self.flux_zpf * (a_daggar + a)), np.abs(self.charge_zpf * (a_daggar - a))

    def calc_sin_phi_over_two_sweep(self, state1, state2):
        """
        Calculation of the sin(phi/2) matrix element between two states as a function of the sweep parameter.

        :param state1: first state.
        :param state2: second state.
        :return: 1D numpy array with sin(phi/2) matrix elements.
        """

        mel_sweep = np.zeros(self.steps)

        for i in range(self.steps):

            self.inspect_sweep(i)
            mel_sweep[i] = self.calc_sin_phi_over_two(state1, state2)

        return mel_sweep

    def calc_sin_phi_over_two(self, state1, state2):
        """
        Calculation of the sin(phi/2) matrix element between two states.

        :param state1: first state.
        :param state2: second state.
        :return: sin(phi/2) matrix element.
        """

        mel = 0
        for m in range(self.N):
            for n in range(self.N):
                mel += self.v[m, state1] * self.v[n, state2] * (
                       np.cos(self.p_ext / 2) * self.special_integrals.sin_integral(m, n, np.sqrt(2) * np.pi *
                                                                                    self.flux_zpf) +
                       np.sin(self.p_ext / 2) * self.special_integrals.cos_integral(m, n, np.sqrt(2) * np.pi *
                                                                                    self.flux_zpf))

        return np.abs(mel)

    ###################
    #####  plots  #####
    ###################

    def plot_fluxonium(self, ax, n, x_range=2.0, unit_mass=False, fill_between=True, scale=1.0):
        """
        Plot of the potential and wave functions.

        :param ax: matplotlib axes instance.
        :param n: number of lower states that will be plotted.
        :param x_range: x-axis range that will be plotted in units depending on unit_mass.
        :param unit_mass: if False the flux potential is plotted in units of Phi_0.
                          if True the potential is transformed to correspond to a particle of unit mass and is plotted
                          in units of 1 / sqrt(GHz).
        :param fill_between: fill wave functions.
        :param scale: scale the amplitude of wave functions.
        :return: None.
        """

        sample = 1001
        x = np.linspace(-x_range, x_range, sample)

        if not unit_mass:

            u = 2 * np.pi
            beta = u * np.sqrt(np.sqrt(self.El / self.Ec))

            y = 4 * np.pi ** 2 * self.El * x ** 2 / 2 - self.Ej * np.cos(u * x + self.p_ext)
            ax.plot(x, y, color='k', linewidth=1.0, zorder=0)

        else:

            u = np.sqrt(self.Ec)
            beta = np.sqrt(self.w)

            y = (self.w * x) ** 2 / 2 - self.Ej * np.cos(u * x + self.p_ext)
            ax.plot(x, y, color='k', linewidth=1.0, zorder=0)

        # plot lowest n functions
        for i in range(n):

            y = np.zeros_like(x)

            for j in range(self.N):
                # integral of wave function corresponds to hbar * w. For fine adjustments scale can be used
                y += scale * self.v[j, i] * self.w * np.sqrt(beta) * sp.norm_hermite(j, beta * x)
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

    def plot_energy_sweep(self, ax, n):
        """
        Plot of the fluxonium energies as a function of the sweep parameter.

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
            else:
                raise AttributeError("Unsupported dipole attribute.")
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
            ax.set_zlabel(r"$\langle \psi_m|\phi| \psi_n\rangle$ ($\Phi_0$)", labelpad=8)
        elif dipole == "charge":
            ax.set_zlabel(r"$\langle \psi_m|q| \psi_n\rangle$ ($2e$)", labelpad=8)

    ##################
    #####  core  #####
    ##################

    def calc_hamiltonian_parameters(self):
        """
        Calculation of all relevant fluxonium parameters.

        :return: None.
        """

        # calculate energies of the fluxonium Hamiltonian expressed with the number and phase operator
        # Ec and El are scaled to be in units of 1GHz. Ej must be given in GHz.
        # Note that the natural definition of Ec is used such that w = sqrt(Ec * El)

        self.Ec = 4e-9 * pyc.e ** 2 / pyc.h / self.C
        self.El = pyc.h / (16e9 * np.pi ** 2 * pyc.e ** 2) / self.L

        # calculated Hamiltonian parameters for the natural harmonic oscillator basis, i.e., the particle has unit mass
        self.w = np.sqrt(self.Ec * self.El)
        self.u = np.sqrt(self.Ec)

        # Matrix elements:
        self.Z = np.sqrt(self.Ec) / (2 * np.pi * np.sqrt(self.El))  # in R_Q

        # <psi| phi / Phi_0 |psi> = flux_zpf * <psi| a_dagger + a| psi>
        self.flux_zpf = np.sqrt(self.Z / (4 * np.pi))  # in Phi_0

        # <psi| q / (2 * e) |psi> = charge_zpf * <psi| a_dagger + a| psi>
        self.charge_zpf = 1.0 / np.sqrt(4 * np.pi * self.Z)  # in 2 * e

    def diagonalize_hamiltonian(self):
        """
        Numerical diagonalization of the fluxonium Hamiltonian.

        :return: None.
        """

        self.H = np.zeros((self.N, self.N))

        for m in range(self.N):

            for n in range(m):
                self.H[m, n] = - self.Ej * (
                        np.cos(self.p_ext) * self.special_integrals.cos_integral(m, n,
                                                                                   np.sqrt(8) * np.pi * self.flux_zpf) -
                        np.sin(self.p_ext) * self.special_integrals.sin_integral(m, n,
                                                                                   np.sqrt(8) * np.pi * self.flux_zpf))

                self.H[n, m] = self.H[m, n]

            self.H[m, m] = self.w * (m + 0.5) - self.Ej * (
                    np.cos(self.p_ext) * self.special_integrals.cos_integral(m, m,
                                                                               np.sqrt(8) * np.pi * self.flux_zpf) -
                    np.sin(self.p_ext) * self.special_integrals.sin_integral(m, m,
                                                                               np.sqrt(8) * np.pi * self.flux_zpf))

        self.E, self.v = sc.linalg.eigh(self.H)
        self.Eg = self.E[0]

        # calculate the creator an annihilator array
        self.sqrts = np.sqrt(np.arange(1, self.N))

    def draw_circuit(self):
        """
        Creates a figure with the circuit.

        :return: matplotlib figure instance
        """

        figwidth = 6.0
        fig, ax = plt.subplots(1, 1, figsize=(figwidth, 2.2 / 6.0 * figwidth))
        ax.axis("off")

        d = schemdraw.Drawing(canvas=ax)
        d.config(unit=2.0, fontsize=15)
        d.move(2.0, 0.0)
        d += (elm.Line().left())
        d += elm.Inductor().up().label("$L$")
        d += elm.Line().right()
        d += elm.Capacitor().down().label("$C$")
        d += elm.Line().right()
        d += elm.Josephson().up().label(r"$E_\mathrm{J}$")
        d += elm.Line().left()

        d.draw()

        #ax.axvline(-1.0)
        #ax.axvline(5.0)
        #ax.axhline(-0.1)
        #ax.axhline(2.1)

        ax.set_xlim(-1.0, 5.0)
        ax.set_ylim(-0.1, 2.1)

        return fig

    def show_formulas(self):
        """
        Creates a figure showing the most relevant formulas.

        :return: matplotlib figure instance
        """

        figwidth = 10.0
        fig, ax = plt.subplots(1, 1, figsize=(figwidth, 4.5 / 10.0 * figwidth))
        ax.axis("off")

        ax.text(0.5, 2.0, "Fluxonium Hamiltonian:", va="top", fontsize=12)
        ax.text(0.75, 1.42, "$H$", va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(1.0, 1.5, r"$=\frac{1}{2C}q^2 + \frac{1}{2L}(\phi - \Phi_\mathrm{ext})^2"
                           r" - E_\mathrm{J}\cos\left(\frac{2 \pi}{\Phi_0}\phi\right)$" + "\n" +
                           r"$ = \frac{1}{2} E_C n^2 + \frac{1}{2} E_L (\varphi - \varphi_\mathrm{ext})^2"
                           r" - E_\mathrm{J}\cos(\varphi)$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        ax.text(0.5, 0.0, "A time-independet external flux allows shifting the coordinate system:",
                va="top", fontsize=12)
        ax.text(0.75, -0.58, "$H$", va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(1.0, -0.5, r"$= \frac{1}{2} E_C n^2 + \frac{1}{2} E_L \varphi^2"
                            r" - E_\mathrm{J}\cos(\varphi + \varphi_\mathrm{ext})$" + "\n" +
                            r"$= \frac{1}{2} p^2 + \frac{\hbar^2\omega^2}{2} x^2"
                            r"- E_\mathrm{J}\cos(u x + \varphi_\mathrm{ext}), "
                            r"\quad \hbar\omega = \sqrt{E_L E_C}, \quad u = \sqrt{E_C}$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        #ax.axvline(0.0)
        #ax.axvline(10.0)
        #ax.axhline(-2.4)
        #ax.axhline(2.1)

        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(- 2.4, 2.1)

        return fig

    def __repr__(self):
        """
        Returns a representation of the fluxonium parameters.

        :return: string.
        """

        return (f"L = {self.L:.4e}\n"
                f"C = {self.C:.4e}\n"
                f"Ec = {self.Ec:.4e}\n"
                f"El = {self.El:.4e}\n"
                f"Ej = {self.Ej:.4e}\n"
                f"Ejs = {self.Ejs:.4e}\n"
                f"Ejd = {self.Ejd:.4e}\n"
                f"ratio = {self.ratio:.4e}\n"
                f"w = {self.w:.4e}\n"
                f"u = {self.u:.4e}\n"
                f"Z = {self.Z:.4e}\n"
                f"flux_zpf = {self.flux_zpf:.4e}\n"
                f"charge_zpf = {self.charge_zpf:.4e}\n"
                )

    ########################################
    ##### parameter storage for sweeps #####
    ########################################

    class Parameters:

        def __init__(self, fluxonium):

            self.C = fluxonium.C
            self.Ec = fluxonium.Ec
            self.L = fluxonium.L
            self.El = fluxonium.El
            self.Ej = fluxonium.Ej
            self.p_ext = fluxonium.p_ext
            self.w = fluxonium.w
            self.u = fluxonium.u
            self.Z = fluxonium.Z
            self.flux_zpf = fluxonium.flux_zpf
            self.charge_zpf = fluxonium.charge_zpf
            self.N = fluxonium.N
