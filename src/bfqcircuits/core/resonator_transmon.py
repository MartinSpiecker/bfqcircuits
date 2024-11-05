#  Copyright (c) 2024. Martin Spiecker. All rights reserved.

import numpy as np
import scipy.constants as pyc
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm

from bfqcircuits.utils.resonator_atom import ResonatorAtom


class ResonatorTransmon(ResonatorAtom):

    def __init__(self):

        ResonatorAtom.__init__(self)

        self.Lr = 0.0
        self.Cr = 0.0
        self.Ca = 0.0
        self.Cs = 0.0

        self.Elr = 0.0
        self.Ecr = 0.0
        self.Eca = 0.0
        self.Ecs = 0.0
        self.Ej = 0
        self.ng = 0

        self.wr = 0.0
        self.wa = 0.0
        self.g = 0.0

        self.flux_zpf = None
        self.charge_zpf = None

        self.flux_operator = None
        self.charge_operator = None
        self.calc_sin_phi_over_operator = None

    def set_parameters(self, Lr=None, Cr=None, Ca=None, Cs=None, Ej=None, ng=None, Na=None, Nr=None):
        """
        Helper routine to set relevant system parameters.

        :param Lr: resonator inductance in [H].
        :param Cr: resonator capacitance in [F].
        :param Ca: transmon capacitance (including the shared capacitance) in [F].
        :param Cs: shared capacitance in [F].
        :param Ej: Josephson energy in [GHz].
        :param ng: offset charge in [2e].
        :param Na: number of transmon basis states.
        :param Nr: number of resonator basis states.
        :return: None.
        """

        if Lr is not None:
            self.Lr = Lr
        if Cs is not None:
            self.Cs = Cs
        if Cr is not None:
            self.Cr = Cr
        if Ca is not None:
            self.Ca = Ca
        if Ej is not None:
            self.Ej = Ej
        if ng is not None:
            self.ng = ng
        if Na is not None:
            self.Na = Na
        if Nr is not None:
            self.Nr = Nr

    def sweep_offset_charge(self, ng_sweep):
        """
        Sweep of offset charge.

        :param ng_sweep: 1D numpy array with the offset charge values in [2e].
        :return: None.
        """

        self._initialize_sweep(ng_sweep)

        for i in range(self.steps):
            self.ng = self.par_sweep[i]
            self._calc_sweep(i)

    def sweep_parameter(self, par_sweep, par_name=None):
        """
        Sweep of system parameters.

        :param par_sweep: either dict with keys "Lr", "La", "Cr", "Ca", "Cs" "Ej", "ng" and corresponding
                          1D numpy parameter arrays,
                          or 1D numpy array with parameter sweep for the parameter specified by par_name.
        :param par_name: "Lr", "La", "Cr", "Ca", "Cs", "Ej", "ng".
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
                self._calc_sweep(i)
        else:
            if par_name in ["Lr", "La", "Cr", "Ca", "Cs", "Ej", "ng"]:

                self._initialize_sweep(par_sweep)
                for i in range(self.steps):
                    setattr(self, par_name, self.par_sweep[i])
                    self.calc_hamiltonian_parameters()
                    self._calc_sweep(i)
            else:
                print("Parameter does not exist or should not be swept in this routine. \n" +
                      "For instance, for sweeping the number of basis functions use the convergence_sweep routine.")

    def inspect_sweep(self, step):
        """
        Reset system to a specific sweep step. The routine assumes that the sweep has already been performed.

        :param: step: step of the sweep.
        :return: None.
        """

        self.Lr = self.system_pars_sweep[step].Lr
        self.Cr = self.system_pars_sweep[step].Cr
        self.Ca = self.system_pars_sweep[step].Ca
        self.Cs = self.system_pars_sweep[step].Cs
        self.Elr = self.system_pars_sweep[step].Elr
        self.Ecr = self.system_pars_sweep[step].Ecr
        self.Eca = self.system_pars_sweep[step].Eca
        self.Ecs = self.system_pars_sweep[step].Ecs
        self.Ej = self.system_pars_sweep[step].Ej
        self.ng = self.system_pars_sweep[step].ng
        self.Na = self.system_pars_sweep[step].Na
        self.Nr = self.system_pars_sweep[step].Nr
        self._operators()

        self.wr = self.system_pars_sweep[step].wr
        self.wa = self.system_pars_sweep[step].wa
        self.g = self.system_pars_sweep[step].g

        self.flux_zpf = self.system_pars_sweep[step].flux_zpf
        self.charge_zpf = self.system_pars_sweep[step].charge_zpf

        super().inspect_sweep(step)

    def _calc_sweep(self, step):
        """
        Private routine that diagonalizes the Hamiltonian and stores the data of the given sweep step.

        :param step: step of the parameter sweep.
        :return: None.
        """

        self.system_pars_sweep[step] = self.Parameters(self)
        super()._calc_sweep(step)

    ######################
    #####  analysis  #####
    ######################

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

    ###################
    #####  plots  #####
    ###################

    def plot_potential(self, ax, x_range=2.0):
        """
        3D plot of the resonator transmon potential.

        :param ax: matplotlib axes instance with projection="3d".
        :param x_range: x-axis range of the resonator that will be plotted in units of Phi_0.
        :return: None.
        """

        sample = 1001
        x = np.linspace(-x_range, x_range, sample)
        y = np.linspace(-np.pi, np.pi, sample)

        x, y = np.meshgrid(x, y, sparse=False, indexing='xy')

        z = 4 * np.pi**2 * self.Elr**2 * x**2 / 2 - self.Ej * np.cos(y)

        ax.plot_wireframe(x, y, z)

        ax.set_xlabel(r"$\phi_\mathrm{r}$ $(\Phi_0)$")
        ax.set_ylabel(r"$\phi_\mathrm{q}$ $(\Phi_0)$")
        ax.set_zlabel(r"$E$ (GHz)")

    ##################
    #####  core  #####
    ##################

    def calc_resonator_dipole_moments(self, state1, state2):
        """
        Calculation of the resonator flux and charge dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: integer or tupel (na, nr), which requires that the states have been sorted.
        :param state2: integer or tupel (na, nr), which requires that the states have been sorted.
        :return: absolute flux dipole moment, absolute charge dipole moment.
        """

        if hasattr(state1, '__iter__'):
            v1 = self.v_sort[*state1, :]
        else:
            v1 = self.v[:, state1]
        if hasattr(state2, '__iter__'):
            v2 = self.v_sort[*state2, :]
        else:
            v2 = self.v[:, state2]

        # For the resonator we essentially use the flux eigenstates with complex prefactors (-i)^n such that
        # we can avoid a complex Hamiltonian

        b_daggar = np.sum(v1[self.Na:] * self.sqrts_r * v2[:-self.Na])
        b = np.sum(v1[:-self.Na] * self.sqrts_r * v2[self.Na:])

        return np.abs(self.flux_zpf[0, 0] * (b_daggar - b)), np.abs(self.charge_zpf[0, 0] * (b_daggar + b))

    def calc_atom_dipole_moments(self, state1, state2):
        """
        Calculation of the transmon flux and charge dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: integer or tupel (na, nr), which requires that the states have been sorted.
        :param state2: integer or tupel (na, nr), which requires that the states have been sorted.
        :return: absolute flux dipole moment, absolute charge dipole moment.
        """

        if hasattr(state1, '__iter__'):
            v1 = self.v_sort[*state1, :]
        else:
            v1 = self.v[:, state1]
        if hasattr(state2, '__iter__'):
            v2 = self.v_sort[*state2, :]
        else:
            v2 = self.v[:, state2]

        fdm = np.dot(v1, np.dot(self.flux_operator, v2))
        cdm = np.sum(v1 * self.charge_operator * v2)

        return np.abs(fdm), np.abs(cdm)

    def calc_sin_phi_over_two(self, state1, state2):
        """
        Calculation of the sin(phi/2) matrix element between two states.

        :param state1: first state.
        :param state2: second state.
        :return: sin(phi/2) matrix element.
        """

        mel = np.dot(self.v[:, state1], np.dot(self.calc_sin_phi_over_operator, self.v[:, state2]))

        return np.abs(mel)

    def calc_hamiltonian_parameters(self):
        """
        Calculation of all relevant Hamiltonian parameters.

        :return: None.
        """

        # Calculate energies of the transmon Hamiltonian expressed with the number and phase operator.
        # El and Ec are scaled to be in units of 1GHz. Ej must be given in GHz.
        # Note that the natural definition of Ec is used such that w = sqrt(Ec * El)

        self.Elr = pyc.h / (16e9 * np.pi ** 2 * pyc.e ** 2) / self.Lr

        # Compute capacitance matrix
        C = self.Cr * self.Ca - self.Cs ** 2
        Cr = self.Ca / C
        Ca = self.Cr / C
        Cs = self.Cs / C

        self.Ecr = 4e-9 * pyc.e**2 / pyc.h * Cr
        self.Eca = 4e-9 * pyc.e**2 / pyc.h * Ca
        self.Ecs = 4e-9 * pyc.e**2 / pyc.h * Cs

        self.wr = np.sqrt(self.Ecr * self.Elr)
        self.wa = np.sqrt(self.Eca * self.Ej)

        self.wr_approx = self.wr  # for sorting the levels

        Zr = np.sqrt(self.Ecr) / (2 * np.pi * np.sqrt(self.Elr))  # in R_Q
        Zq = np.sqrt(self.Eca) / (2 * np.pi * np.sqrt(self.Ej))  # in R_Q

        self.flux_zpf = np.zeros((2, 2))
        self.flux_zpf[0, 0] = np.sqrt(Zr / (4 * np.pi))  # in Phi_0
        self.flux_zpf[1, 1] = np.sqrt(Zq / (4 * np.pi))  # in Phi_0

        self.charge_zpf = np.zeros((2, 2))
        self.charge_zpf[0, 0] = 1 / np.sqrt(4 * np.pi * Zr)  # in 2 * e
        self.charge_zpf[1, 1] = 1 / np.sqrt(4 * np.pi * Zq)  # in 2 * e

        self.g = self.Ecs * self.charge_zpf[0, 0] * self.charge_zpf[1, 1]

    def diagonalize_hamiltonian(self):
        """
        Numerical diagonalization of the system Hamiltonian.

        :return: None.
        """

        # For the resonator we essentially use the flux eigenstates the with complex prefactors (-i)^n such that
        # we can avoid a complex Hamiltonian

        charge_zpf_r = self.charge_zpf[0, 0]
        cs = np.arange(self.Na) - (self.Na - 1) // 2  # charge states transmon

        self.H = np.zeros((self.Nr * self.Na, self.Nr * self.Na))

        for m in range(self.Nr * self.Na):

            i = m // self.Na
            g = m % self.Na

            for n in range(m):

                j = n // self.Na
                h = n % self.Na

                if i == j and g == h + 1:
                    self.H[m, n] = - 0.5 * self.Ej

                if i == j + 1 and g == h:          # minus sign from complex prefactors
                    self.H[m, n] = - self.Ecs * charge_zpf_r * np.sqrt(i) * (cs[g] - self.ng)

                self.H[n, m] = self.H[m, n]

            self.H[m, m] = self.wr * (i + 0.5) + 0.5 * self.Eca * (cs[g] - self.ng)**2

        super().diagonalize_hamiltonian()

        # calculate the creator an annihilator array
        self._operators()

    def _operators(self):
        """
        Private routine that creates the flux and charge dipole operators and the sin(phi / 2) operator.

        :return: None.
        """

        self.sqrts_r = np.zeros((self.Nr - 1) * self.Na)
        for i in range((self.Nr - 1) * self.Na):
            r = i // self.Na
            self.sqrts_r[i] = np.sqrt(r + 1)

        cs = np.arange(self.Na) - (self.Na - 1) // 2  # charge states

        self.charge_operator = np.tile(cs, self.Nr)

        k = np.add.outer(-cs, cs)
        b = k != 0
        flx_op = np.zeros((self.Na, self.Na))
        # We neglect in the following line the prefactor/rotation - 1j.
        # The factor 2 pi is needed to convert from the phase operator to the charge operator
        flx_op[b] = (1 - 2 * (k[b] % 2)) / (2 * np.pi * k[b])
        self.flux_operator = np.tile(flx_op, (self.Nr, self.Nr))

        self.calc_sin_phi_over_operator = np.tile(8 * k * (1 - 2 * (k % 2)) / (1 - 4 * k ** 2), (self.Nr, self.Nr))

    def draw_circuit(self):
        """
        Creates a figure with the circuit.

        :return: matplotlib figure instance
        """

        figwidth = 8.0
        fig, ax = plt.subplots(1, 1, figsize=(figwidth, 2.8 / 8.0 * figwidth))
        ax.axis("off")

        d = schemdraw.Drawing(canvas=ax)
        d.config(unit=2.0, fontsize=15)
        d.move(2.0, 0.0)
        d += elm.Line().left()
        d += elm.Inductor().up().label(r"$L_\mathrm{r}$")
        d += elm.Line().right()
        d += elm.Capacitor().down().label(r"$C_\mathrm{r} - C_\mathrm{s}$")
        d += elm.Line().right()
        d += elm.Capacitor().up().label(r"$C_\mathrm{a} - C_\mathrm{s}$")
        d += elm.Capacitor().left().label(r"$C_\mathrm{s}$")
        d.move(2.0, 0.0)
        d += elm.Line().right()
        d += elm.Josephson().down().label(r"$E_\mathrm{J}$")
        d += elm.Line().left()

        d.draw()

        #ax.axvline(-1.0)
        #ax.axvline(7.0)
        #ax.axhline(-0.1)
        #ax.axhline(2.7)

        ax.set_xlim(-1.0, 7.0)
        ax.set_ylim(-0.1, 2.7)

        return fig

    def show_formulas(self):
        """
        Creates a figure showing the most relevant formulas.

        :return: matplotlib figure instance
        """

        figwidth = 10.0
        fig, ax = plt.subplots(1, 1, figsize=(figwidth, 5.0 / 10.0 * figwidth))
        ax.axis("off")

        ax.text(0.5, 2.0, "Resonator transmon Hamiltonian:", va="top", fontsize=12)
        ax.text(1.0, 1.5, r"$H =\frac{1}{2}\frac{1}{C_\mathrm{r}C_\mathrm{a} - C_\mathrm{s}^2}\mathbf{q}^T "
                          r"\genfrac{(}{)}{0}{1}{\,C_\mathrm{a}\;\,\,C_\mathrm{s}}{\,C_\mathrm{s}\;\;\,C_\mathrm{r}}\mathbf{q}"
                          r"+ \frac{1}{2L_\mathrm{r}}\phi_\mathrm{r}^2"
                          r" - E_\mathrm{J}\cos\left(\frac{2 \pi}{\Phi_0}\phi_\mathrm{a}\right),$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(0.5, 0.9, r"with", va="top", fontsize=12)
        ax.text(1.05, 0.96, r"$\mathbf{q} = (q_\mathrm{r}, q_\mathrm{a} - q_\mathrm{g})^T.$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        ax.text(0.5, 0.4, "With dimensionless operators the Hamiltonian reads:",
                va="top", fontsize=12)
        ax.text(1.0, -0.1, r"$H =\frac{1}{2}\mathbf{n}^T "
                           r"\genfrac{(}{)}{0}{1}{E_{C_\mathrm{r}}\;\;\;E_{C_\mathrm{s}}}"
                           r"{\,E_{C_\mathrm{s}}\;\;\,E_{C_\mathrm{q}}}\mathbf{n} + "
                           r"\frac{1}{2}E_{L_\mathrm{r}}\varphi_\mathrm{r}^2"
                           r" - E_\mathrm{J}\cos\left(\varphi_\mathrm{a}\right)$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(0.5, -0.8, "The wave functions are subject to the periodic boundary condition:",
                va="top", fontsize=12)
        ax.text(1.0, -1.2, r"$\psi(\varphi_\mathrm{r}, \varphi_\mathrm{a} + 2 \pi)="
                           r"\psi(\varphi_\mathrm{r}, \varphi_\mathrm{a})$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        #ax.axvline(0.0)
        #ax.axvline(10.0)
        #ax.axhline(-1.9)
        #ax.axhline(2.1)

        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(- 1.9, 2.1)

        return fig

    def __repr__(self):
        """
        Returns a representation of the resonator transmon parameters.

        :return: string
        """

        return (f"Lr = {self.Lr:.4e}\n"
                f"Cr = {self.Cr:.4e}\n"
                f"Ca = {self.Ca:.4e}\n"
                f"Cs = {self.Cs:.4e}\n"
                f"\n"
                f"Elr = {self.Elr:.4e}\n"
                f"Ecr = {self.Ecr:.4e}\n"
                f"Eca = {self.Eca:.4e}\n"
                f"Ecs = {self.Ecs:.4e}\n"
                f"Ej = {self.Ej:.4e}\n"
                f"\n"
                f"ℏωᵣ = {self.wr:.4e}\n"
                f"ℏωₐ = {self.wa:.4e}\n"
                f"ℏg = {self.g:.4e}\n"
                f"Ej / Ec = {self.Ej / self.Eca:.4e}\n"
                f"\n"
                f"flux_zpf = [[{self.flux_zpf[0, 0]:.4e}, {self.flux_zpf[0, 1]:.4e}], "
                f"[{self.flux_zpf[1, 0]:.4e}, {self.flux_zpf[1, 1]:.4e}]]\n"
                f"charge_zpf = [[{self.charge_zpf[0, 0]:.4e}, {self.charge_zpf[0, 1]:.4e}], "
                f"[{self.charge_zpf[1, 0]:.4e}, {self.charge_zpf[1, 1]:.4e}]]\n"
                )

    class Parameters:

        def __init__(self, resonator_transmon):

            self.Lr = resonator_transmon.Lr
            self.Cr = resonator_transmon.Cr
            self.Ca = resonator_transmon.Ca
            self.Cs = resonator_transmon.Cs
            self.Elr = resonator_transmon.Elr
            self.Ecr = resonator_transmon.Ecr
            self.Eca = resonator_transmon.Eca
            self.Ecs = resonator_transmon.Ecs
            self.Ej = resonator_transmon.Ej
            self.ng = resonator_transmon.ng
            self.Na = resonator_transmon.Na
            self.Nr = resonator_transmon.Nr

            self.wr = resonator_transmon.wr
            self.wa = resonator_transmon.wa
            self.g = resonator_transmon.g

            self.flux_zpf = resonator_transmon.flux_zpf
            self.charge_zpf = resonator_transmon.charge_zpf












