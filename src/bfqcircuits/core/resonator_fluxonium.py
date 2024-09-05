#  Copyright (c) 2024. Martin Spiecker. All rights reserved.

import numpy as np
import scipy.constants as pyc
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm

from bfqcircuits.utils.resonator_atom import ResonatorAtom
from bfqcircuits.utils import special as sp


class ResonatorFluxonium:

    def __new__(cls, coupling="inductive", basis="product"):
        """
        Helper class creating the desired resonator fluxonium class with the specified coupling and basis.

        :param coupling: either "inductive" or "capacitive".
        :param basis: either "product", which uses the resonator and fluxonium product basis.
                      or "normal", which uses a product basis of the normal modes of the linear Hamiltonian.
        """

        if coupling == "inductive":
            if basis == "product":
                return ResonatorFluxoniumInductiveProduct()
            elif basis == "normal":
                return ResonatorFluxoniumInductiveNormal()
        elif coupling == "capacitive":
            if basis == "product":
                return ResonatorFluxoniumCapacitiveProduct()
            elif basis == "normal":
                return ResonatorFluxoniumCapacitiveNormal()


#############################
#### Inductive coupling ####
#############################

class ResonatorFluxoniumInductive(ResonatorAtom):

    def __init__(self):

        ResonatorAtom.__init__(self)

        self.Lr = 0.0
        self.La = 0.0
        self.Ls = 0.0
        self.Cr = 0.0
        self.Ca = 0.0

        self.Elr = 0.0
        self.Ela = 0.0
        self.Els = 0.0
        self.Ecr = 0.0
        self.Eca = 0.0
        self.Ej = 0
        self.Ejs = 0
        self.Ejd = 0
        self.ratio = 0
        self.p_ext = 0

        self.wr = 0.0
        self.wrp = 0.0
        self.wa = 0.0
        self.wap = 0.0
        self.g_lin_sq = 0.0

        self.S = None
        self.flux_zpf = None
        self.charge_zpf = None

        self.sqrts_r = None
        self.sqrts_a = None

    def set_parameters(self, Lr=None, La=None, Ls=None, Cr=None, Ca=None, Ej=None, Ejs=None, Ejd=None, ratio=None,
                       p_ext=None, Na=None, Nr=None):
        """
        Helper routine to set relevant fluxonium parameters.
        When updating inductances or capacitances self.calc_hamiltonian_parameters() has to be called.

        :param Lr: resonator inductance (including the shared inductance) in [H].
        :param La: fluxonium inductance (including the shared inductance) in [H].
        :param Ls: shared inductance in [H].
        :param Cr: resonator capacitance in [F].
        :param Ca: qubit capacitance in [F].
        :param Ej: Josephson energy in [GHz].
        :param Ejs: sum of the Josephson energies of the SQUID [GHz].
        :param Ejd: difference of the Josephson energies (outer junction  - inner junction) of the SQUID [GHz].
        :param ratio: ratio of the loop areas - fluxonium loop area (enclosed by the inner junction) / SQUID loop area
        :param p_ext: external flux bias in [rad].
        :param Nr: number of resonator basis states.
        :param Nr: number of qubit basis states.
        :return: None.
        """

        if Lr is not None:
            self.Lr = Lr
        if La is not None:
            self.La = La
        if Ls is not None:
            self.Ls = Ls
        if Cr is not None:
            self.Cr = Cr
        if Ca is not None:
            self.Ca = Ca
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
        if Na is not None:
            self.Na = Na
        if Nr is not None:
            self.Nr = Nr

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
        Sweep of the externally applied magentic flux.

        :param pext_sweep: 1D numpy array with the external flux bias in [rad].
        :return: None.
        """

        self._initialize_sweep(pext_sweep)

        self._init_look_up_table(self.Na, self.Nr)
        for i in range(self.steps):
            self.p_ext = self.par_sweep[i]
            self._calc_sweep(i)
        self._destroy_look_up_table()

    def sweep_external_flux_squid(self, pext_sweep):
        """
        Sweep of the externally applied global magentic flux to a SQUID-junction fluxonium.

        The routine assumes that sum of the SQUID-junction Josephson energies self.Ejs and asymmetry self.Ejd as well as
        the ratio of fluxonium loop area to SQUID-area (self.ratio) have been defined.

        :param pext_sweep: 1D numpy array with the external flux bias in [rad].
        :return: None.
        """

        self._initialize_sweep(pext_sweep)

        self._init_look_up_table(self.Na, self.Nr)
        for i in range(self.steps):
            self.set_flux_squid(self.par_sweep[i])
            self._calc_sweep(i)
        self._destroy_look_up_table()

    def sweep_parameter(self, par_sweep, par_name=None):
        """
        Sweep of a resonator fluxonium parameters.

        :param par_sweep: either dict with keys "Lr", "La", "Ls", "Cr", "Ca", "Ej", "Ejs", "Ejd", "ratio",
                           "p_ext" and corresponding 1D numpy parameter arrays,
                           or 1D numpy array with parameter sweep for the parameter specified by par_name.
        :param par_name: "Lr", "La", "Ls", "Cr", "Ca", "Ej", "Ejs", "Ejd", "ratio", "p_ext".
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
            if par_name in ["Lr", "La", "Ls", "Cr", "Ca"]:

                self._initialize_sweep(par_sweep)
                for i in range(self.steps):
                    setattr(self, par_name, self.par_sweep[i])
                    self.calc_hamiltonian_parameters()
                    self._calc_sweep(i)

            elif par_name in ["Ej", "p_ext"]:

                self._initialize_sweep(par_sweep)
                self._init_look_up_table(self.Na, self.Nr)
                for i in range(self.steps):
                    setattr(self, par_name, self.par_sweep[i])
                    self._calc_sweep(i)
                self._destroy_look_up_table()

            elif par_name in ["Ejs", "Ejd", "ratio"]:

                self._initialize_sweep(par_sweep)
                self._init_look_up_table(self.Na, self.Nr)
                for i in range(self.steps):
                    setattr(self, par_name, self.par_sweep[i])
                    self.set_flux_squid(self.p_ext)
                    self._calc_sweep(i)
                self._destroy_look_up_table()

            else:
                print("Parameter does not exist or should not be swept in this routine. \n" +
                      "For instance, for sweeping the number of basis functions use the convergence_sweep routine.")

    def convergence_sweep(self, N_sweep):
        """
        Convergences of the wave functions and corresponding energies with the number of basis states.

        :param N_sweep: 2D numpy array of numbers of basis states of the form [(Na, Nr), ...].
        :return: None.
        """

        self.steps = N_sweep.shape[0]
        self.par_sweep = N_sweep
        size = np.max(N_sweep[:, 0] * N_sweep[:, 1])
        self.system_pars_sweep = np.zeros(self.steps, dtype=object)
        self.H_sweep = np.full((size, size, self.steps), np.nan)
        self.E_sweep = np.full((size, self.steps), np.nan)
        self.v_sweep = np.full((size, size, self.steps), np.nan)

        self._init_look_up_table(np.max(N_sweep[:, 0]), np.max(N_sweep[:, 1]))
        for i in range(self.steps):
            self.Na = N_sweep[i, 0]
            self.Nr = N_sweep[i, 1]
            self._calc_sweep(i)
        self._destroy_look_up_table()

    def inspect_sweep(self, step):
        """
        Reset system to a specific sweep step. The routine assumes that the sweep has already been performed.

        :param: step: step of the sweep.
        :return: None.
        """

        self.Lr = self.system_pars_sweep[step].Lr
        self.La = self.system_pars_sweep[step].La
        self.Ls = self.system_pars_sweep[step].Ls
        self.Cr = self.system_pars_sweep[step].Cr
        self.Ca = self.system_pars_sweep[step].Ca
        self.Elr = self.system_pars_sweep[step].Elr
        self.Ela = self.system_pars_sweep[step].Ela
        self.Els = self.system_pars_sweep[step].Els
        self.Ecr = self.system_pars_sweep[step].Ecr
        self.Eca = self.system_pars_sweep[step].Eca
        self.Ej = self.system_pars_sweep[step].Ej
        self.Ejs = self.system_pars_sweep[step].Ejs
        self.Ejd = self.system_pars_sweep[step].Ejd
        self.p_ext = self.system_pars_sweep[step].p_ext
        self.Na = self.system_pars_sweep[step].Na
        self.Nr = self.system_pars_sweep[step].Nr
        self._creator_annihilator()

        self.wr = self.system_pars_sweep[step].wr
        self.wrp = self.system_pars_sweep[step].wrp
        self.wa = self.system_pars_sweep[step].wa
        self.wap = self.system_pars_sweep[step].wap
        self.g_lin_sq = self.system_pars_sweep[step].g_lin_sq

        self.S = self.system_pars_sweep[step].S
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

    def _init_look_up_table(self, Na, Nr):  # will be overwritten
        pass

    def _destroy_look_up_table(self):  # will be overwritten
        pass

    ############################
    #####  sweep analysis  #####
    ############################

    ###################
    #####  plots  #####
    ###################

    def plot_potential(self, ax, xy_range=2.0, unit_mass=True):
        """
        3D plot of the resonator fluxonium potential.

        :param ax: matplotlib axes instance with projection="3d".
        :param xy_range: scaler or tupel of the x, y range that will be plotted.
        :param unit_mass: if False the flux potential is plotted in units of Phi_0.
                          if True the potential is transformed to correspond to a particle of uniform and unit mass and
                          is plotted in units of 1 / sqrt(GHz).
        :return: None.
        """

        sample = 1001
        if hasattr(xy_range, '__iter__'):
            x = np.linspace(-xy_range[0], xy_range[0], sample)  # resonator
            y = np.linspace(-xy_range[1], xy_range[1], sample)  # qubit
        else:
            x = np.linspace(-xy_range, xy_range, sample)
            y = np.linspace(-xy_range, xy_range, sample)

        x, y = np.meshgrid(x, y, sparse=False, indexing='xy')

        if not unit_mass:

            z = (4 * np.pi ** 2 * self.Elr ** 2 * x ** 2 / 2 + 4 * np.pi ** 2 * self.Ela ** 2 * y ** 2 / 2
                 + 4 * np.pi ** 2 * self.Els * x * y - self.Ej * np.cos(y))

            ax.plot_wireframe(x, y, z)

            ax.set_xlabel(r"$\phi_\mathrm{r}$ $(\Phi_0)$")
            ax.set_ylabel(r"$\phi_\mathrm{q}$ $(\Phi_0)$")

        else:
            z = (self.wr ** 2 * x ** 2 / 2 + self.wa ** 2 * y ** 2 / 2 + self.g_lin_sq * x * y
                 - self.Ej * np.cos(np.sqrt(self.Eca) * y))

            ax.plot_wireframe(x, y, z)

            ax.set_xlabel(r"$x_\mathrm{r}$ $(1 / \sqrt{GHz})$")
            ax.set_ylabel(r"$x_\mathrm{q}$ $(1 / \sqrt{GHz})$")

        ax.set_zlabel(r"$E$ (GHz)")

    ##################
    #####  core  #####
    ##################

    def _creator_annihilator(self):
        """
        Private routine that creates the creation and annihilations operators.

        :return: None
        """

        self.sqrts_r = np.zeros((self.Nr - 1) * self.Na)
        for i in range((self.Nr - 1) * self.Na):
            r = i // self.Na
            self.sqrts_r[i] = np.sqrt(r + 1)

        self.sqrts_a = np.zeros(self.Nr * self.Na - 1)
        for i in range(self.Nr * self.Na - 1):
            q = i % self.Na
            if q < self.Na - 1:
                self.sqrts_a[i] = np.sqrt(q + 1)

    def calc_hamiltonian_parameters(self):
        """
        Calculation of all relevant Hamiltonian parameters.

        :return: None.
        """

        # Compute inductance matrix
        L = self.Lr * self.La - self.Ls ** 2
        lr = self.La / L
        la = self.Lr / L
        ls = - self.Ls / L

        # Calculate energies of the fluxonium Hamiltonian expressed with the number and phase operator.
        # El and Ec are scaled to be in units of 1GHz. Ej must be given in GHz.
        # note that th natural definition of Ec is used such that w = sqrt(Ec * El)

        self.Elr = pyc.h / (16e9 * np.pi ** 2 * pyc.e ** 2) * lr
        self.Ela = pyc.h / (16e9 * np.pi ** 2 * pyc.e ** 2) * la
        self.Els = pyc.h / (16e9 * np.pi ** 2 * pyc.e ** 2) * ls

        self.Ecr = 4e-9 * pyc.e**2 / pyc.h / self.Cr
        self.Eca = 4e-9 * pyc.e ** 2 / pyc.h / self.Ca

        # Calculated Hamiltonian parameters for the natural harmonic oscillator basis, i.e., the particle has unit mass
        wr_sq = self.Ecr * self.Elr
        wa_sq = self.Eca * self.Ela
        self.g_lin_sq = self.Els * np.sqrt(self.Ecr * self.Eca)

        self.wr = np.sqrt(wr_sq)
        self.wa = np.sqrt(wa_sq)

        # Calculate normal modes of the harmonic part of the Hamiltonian.
        # This basis is used in the normal mode diagonalization.
        # The coordinates x1/2 are chosen to be resonator/qubit like
        # The resonator frequency wrp is needed for the sorting algorithm.

        sqrt = np.sqrt((wr_sq - wa_sq)**2 + 4 * self.g_lin_sq ** 2)

        if wr_sq >= wa_sq:
            self.wrp = np.sqrt(0.5 * (wr_sq + wa_sq + sqrt))  # resonator like
            self.wap = np.sqrt(0.5 * (wr_sq + wa_sq - sqrt))  # qubit like
            s1 = (wr_sq - wa_sq) + sqrt
            s2 = - 2 * self.g_lin_sq
        else:
            self.wrp = np.sqrt(0.5 * (wr_sq + wa_sq - sqrt))  # resonator like
            self.wap = np.sqrt(0.5 * (wr_sq + wa_sq + sqrt))  # qubit like
            s1 = (wa_sq - wr_sq) + sqrt
            s2 = 2 * self.g_lin_sq

        self.S = np.array([[s1, s2], [- s2, s1]]) / np.sqrt(s1**2 + s2**2)

        self.wr_approx = self.wrp   # for sorting the levels

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
        d += (elm.Line().left())
        d += elm.Capacitor().up().label(r"$C_\mathrm{r}$")
        d += elm.Inductor().right().label(r"$L_\mathrm{r} - L_\mathrm{s}$")
        d += elm.Inductor().down().label(r"$L_\mathrm{s}$")
        d += elm.Line().right()
        d += elm.Capacitor().up().label(r"$C_\mathrm{a}$")
        d += elm.Line().right()
        d += elm.Josephson().down().label(r"$E_\mathrm{J}$")
        d += elm.Line().left()
        d.move(- 2.0, 2.0)
        d += elm.Inductor().right().label(r"$L_\mathrm{a} - L_\mathrm{s}$")

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
        fig, ax = plt.subplots(1, 1, figsize=(figwidth, 6.5 / 10.0 * figwidth))
        ax.axis("off")

        ax.text(0.5, 2.0, "Resonator fluxonium Hamiltonian:", va="top", fontsize=12)
        #ax.text(0.75, 1.42, "$H$", va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(1.0, 1.5, r"$H =\frac{1}{2}\mathbf{q}^T "
                r"\genfrac{(}{)}{0}{1}{\frac{1}{C_\mathrm{r}}\;\,\,0}{\;0\;\;\frac{1}{C_\mathrm{a}}}\mathbf{q}"
                r"+ \frac{1}{2}\frac{1}{L_\mathrm{r}L_\mathrm{a} - L_\mathrm{s}^2}\mathbf{\phi}^T"
                r"\genfrac{(}{)}{0}{1}{\;\;L_\mathrm{a}\;\;-L_\mathrm{s}}{-L_\mathrm{s}\;\;\;\;L_\mathrm{r}}\mathbf{\phi}"
                r" - E_\mathrm{J}\cos\left(\frac{2 \pi}{\Phi_0}\phi_\mathrm{a}\right),$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(0.5, 0.5, r"with", va="top", fontsize=12)
        ax.text(1.05, 0.56, r"$\mathbf{\phi} = (\phi_\mathrm{r}, \phi_\mathrm{a} - \Phi_\mathrm{ext})^T,\quad"
                r"\mathbf{q} = (q_\mathrm{r}, q_\mathrm{a})^T.$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        ax.text(0.5, -0.2, "A time-independet external flux allows shifting the coordinate system.\n"
                "With dimensionless operators the Hamiltonian reads:",
                va="top", fontsize=12)
        ax.text(0.75, -1.11, "$H$", va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(1.0, -1.0, r"$=\frac{1}{2}\mathbf{n}^T "
                r"\genfrac{(}{)}{0}{1}{\!\!\!\!E_{C_\mathrm{r}}\;\;\;0}{\;\,0\;\;\;\;\;E_{C_\mathrm{q}}}\mathbf{n}"
                r" + \mathbf{\varphi}^T\genfrac{(}{)}{0}{1}{\!E_{L_\mathrm{r}}\;\;\;E_{L_\mathrm{s}}}"
                r"{\;E_{L_\mathrm{s}}\;\;\;E_{L_\mathrm{a}}}\mathbf{\varphi}"
                r" - E_\mathrm{J}\cos\left(\varphi_\mathrm{a} + \varphi_\mathrm{ext}\right)$" + "\n" +
                r"$=\frac{1}{2}\mathbf{p}^T\mathbf{p} + \frac{\hbar^2}{2}\mathbf{x}^T"
                r"\genfrac{(}{)}{0}{1}{\;\omega_\mathrm{r}^2\;\;\;\;\;g_\mathrm{lin}^2}"
                r"{\,g_\mathrm{lin}^2\;\;\;\;\,\omega_\mathrm{a}^2}\mathbf{x}"
                r" - E_\mathrm{J}\cos\left(\sqrt{E_{C_\mathrm{a}}} x_\mathrm{a} + \varphi_\mathrm{ext}\right),$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(0.5, -2.46, r"with", va="top", fontsize=12)
        ax.text(1.05, -2.34, r"$\omega_\mathrm{r} = \sqrt{E_{L_\mathrm{r}} E_{C_\mathrm{r}}},\quad"
                r"\omega_\mathrm{a} = \sqrt{E_{L_\mathrm{a}} E_{C_\mathrm{a}}},\quad"
                r"g_\mathrm{lin}^2 = E_{L_\mathrm{s}}\sqrt{E_{C_\mathrm{r}} E_{C_\mathrm{a}}}.$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(0.5, -3.2, "A coordinate transformation to the normal basis yields:",
                va="top", fontsize=12)
        ax.text(1.0, -3.7, r"$H=\frac{1}{2}\mathbf{p}'^T\mathbf{p}' + \frac{\hbar^2}{2}\mathbf{x}'^T"
                r"\genfrac{(}{)}{0}{1}{\!\!\!\!\omega_\mathrm{r}'^2\;\;\;\;0}"
                r"{\;\,0\;\;\;\;\;\;\omega_\mathrm{a}'^2}\mathbf{x}'"
                r" - E_\mathrm{J}\cos\left(u x_\mathrm{r}' + v x_\mathrm{a}' + \varphi_\mathrm{ext}\right).$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        #ax.axvline(0.0)
        #ax.axvline(10.0)
        #ax.axhline(-4.4)
        #ax.axhline(2.1)

        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(- 4.4, 2.1)

        return fig

    def __repr__(self):
        """
        Returns a representation of the resonator fluxonium parameters.

        :return: string
        """

        return (f"Lr = {self.Lr:.4e}\n"
                f"La = {self.La:.4e}\n"
                f"Ls = {self.Ls:.4e}\n"
                f"Cr = {self.Cr:.4e}\n"
                f"Ca = {self.Ca:.4e}\n"
                f"\n"
                f"Elr = {self.Elr:.4e}\n"
                f"Ela = {self.Ela:.4e}\n"
                f"Els = {self.Els:.4e}\n"
                f"Ecr = {self.Ecr:.4e}\n"
                f"Eca = {self.Eca:.4e}\n"
                f"Ej = {self.Ej:.4e}\n"
                f"Ejs = {self.Ejs:.4e}\n"
                f"Ejd = {self.Ejd:.4e}\n"
                f"ratio = {self.ratio:.4e}\n"
                f"\n"
                f"wr = {self.wr:.4e}\n"
                f"wrp = {self.wrp:.4e}\n"
                f"wa = {self.wa:.4e}\n"
                f"wap = {self.wap:.4e}\n"
                f"g_lin_sq = {self.g_lin_sq:.4e}\n"
                f"\n"
                f"S = [[{self.S[0, 0]:.4e}, {self.S[0, 1]:.4e}], [{self.S[1, 0]:.4e}, {self.S[1, 1]:.4e}]]\n"
                f"flux_zpf = [[{self.flux_zpf[0, 0]:.4e}, {self.flux_zpf[0, 1]:.4e}], "
                f"[{self.flux_zpf[1, 0]:.4e}, {self.flux_zpf[1, 1]:.4e}]]\n"
                f"charge_zpf = [[{self.charge_zpf[0, 0]:.4e}, {self.charge_zpf[0, 1]:.4e}], "
                f"[{self.charge_zpf[1, 0]:.4e}, {self.charge_zpf[1, 1]:.4e}]]\n"
                )

    class Parameters:

        def __init__(self, resonator_fluxonium):

            self.Lr = resonator_fluxonium.Lr
            self.La = resonator_fluxonium.La
            self.Ls = resonator_fluxonium.Ls
            self.Cr = resonator_fluxonium.Cr
            self.Ca = resonator_fluxonium.Ca
            self.Elr = resonator_fluxonium.Elr
            self.Ela = resonator_fluxonium.Ela
            self.Els = resonator_fluxonium.Els
            self.Ecr = resonator_fluxonium.Ecr
            self.Eca = resonator_fluxonium.Eca
            self.Ej = resonator_fluxonium.Ej
            self.Ejs = resonator_fluxonium.Ejs
            self.Ejd = resonator_fluxonium.Ejd
            self.p_ext = resonator_fluxonium.p_ext
            self.Na = resonator_fluxonium.Na
            self.Nr = resonator_fluxonium.Nr

            self.wr = resonator_fluxonium.wr
            self.wrp = resonator_fluxonium.wrp
            self.wa = resonator_fluxonium.wa
            self.wap = resonator_fluxonium.wap
            self.g_lin_sq = resonator_fluxonium.g_lin_sq

            self.S = resonator_fluxonium.S
            self.flux_zpf = resonator_fluxonium.flux_zpf
            self.charge_zpf = resonator_fluxonium.charge_zpf


class ResonatorFluxoniumInductiveProduct(ResonatorFluxoniumInductive):

    def __init__(self):

        ResonatorFluxoniumInductive.__init__(self)

        self._coupling = "inductive"
        self._basis = "product"

        self.g = 0.0

        self.special_integrals = sp.SpecialIntegrals()

    def inspect_sweep(self, step):
        """
        Reset system to a specific sweep step. The routine assumes that the sweep has already been performed.

        :param: step: step of the parameter sweep.
        :return: None.
        """

        super().inspect_sweep(step)

        self.g = self.system_pars_sweep[step].g

    def _init_look_up_table(self, Na, Nr):
        """
        Private routine that initializes the integral look-up table.

        :param Na: number of fluxonium basis states.
        :param Nr: not used, dummy from the parent class
        :return: None.
        """
        self.special_integrals.init_look_up_table_fluxonium(Na)

    def _destroy_look_up_table(self):
        """
        Destruction of the integral look-up table.

        :return: None.
        """

        self.special_integrals.destroy_look_up_table()

    def calc_resonator_dipole_moments(self, state1, state2):
        """
        Calculation of the resonator flux and charge dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: first state.
        :param state2: second state.
        :return: absolute flux dipole moment, absolute charge dipole moment.
        """

        b_daggar = np.sum(self.sqrts_r * self.v[self.Na:, state1] * self.v[:-self.Na, state2])
        b = np.sum(self.sqrts_r * self.v[:-self.Na, state1] * self.v[self.Na:, state2])

        return np.abs(self.flux_zpf[0, 0] * (b_daggar + b)), np.abs(self.charge_zpf[0, 0] * (b_daggar - b))

    def calc_atom_dipole_moments(self, state1, state2):
        """
        Calculation of the fluxonium flux and charge dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: first state.
        :param state2: second state.
        :return: absolute flux dipole moment, absolute charge dipole moment.
        """

        a_daggar = np.sum(self.sqrts_a * self.v[1:, state1] * self.v[:-1, state2])
        a = np.sum(self.sqrts_a * self.v[:-1, state1] * self.v[1:, state2])

        return np.abs(self.flux_zpf[1, 1] * (a_daggar + a)), np.abs(self.charge_zpf[1, 1] * (a_daggar - a))

    def calc_hamiltonian_parameters(self):
        """
        Calculation of all relevant Hamiltonian parameters.

        :return: None.
        """

        ResonatorFluxoniumInductive.calc_hamiltonian_parameters(self)

        Zr = np.sqrt(self.Ecr) / (2 * np.pi * np.sqrt(self.Elr))  # in R_Q
        Zq = np.sqrt(self.Eca) / (2 * np.pi * np.sqrt(self.Ela))  # in R_Q

        self.flux_zpf = np.zeros((2, 2))
        self.flux_zpf[0, 0] = np.sqrt(Zr / (4 * np.pi))  # in Phi_0
        self.flux_zpf[1, 1] = np.sqrt(Zq / (4 * np.pi))  # in Phi_0

        self.charge_zpf = np.zeros((2, 2))
        self.charge_zpf[0, 0] = 1 / np.sqrt(4 * np.pi * Zr)  # in 2 * e
        self.charge_zpf[1, 1] = 1 / np.sqrt(4 * np.pi * Zq)  # in 2 * e

        self.g = 4 * np.pi**2 * self.Els * self.flux_zpf[0, 0] * self.flux_zpf[1, 1]

    def diagonalize_hamiltonian(self):
        """
        Numerical diagonalization of the system Hamiltonian.

        :return: None.
        """

        self.H = np.zeros((self.Nr * self.Na, self.Nr * self.Na))

        flux_q = np.sqrt(8) * np.pi * self.flux_zpf[1, 1]

        for m in range(self.Nr * self.Na):

            i = m // self.Na
            g = m % self.Na

            for n in range(m):

                j = n // self.Na
                h = n % self.Na

                if i == j:
                    self.H[m, n] = - self.Ej * (
                            np.cos(self.p_ext) * self.special_integrals.cos_integral(g, h, flux_q) +
                            np.sin(self.p_ext) * self.special_integrals.sin_integral(g, h, flux_q))

                if j == i - 1 and h == g - 1:
                    self.H[m, n] += self.g * np.sqrt(i) * np.sqrt(g)
                if j == i - 1 and h == g + 1:
                    self.H[m, n] += self.g * np.sqrt(i) * np.sqrt(h)

                self.H[n, m] = self.H[m, n]

            self.H[m, m] = self.wr * (i + 0.5) + self.wa * (g + 0.5) - self.Ej * (
                    np.cos(self.p_ext) * self.special_integrals.cos_integral(g, g, flux_q) +
                    np.sin(self.p_ext) * self.special_integrals.sin_integral(g, g, flux_q))

        super().diagonalize_hamiltonian()

        # calculate the creator an annihilator array
        self._creator_annihilator()

    def __repr__(self):
        """
        Returns a representation of the resonator fluxonium parameters.

        :return: string.
        """

        str = super().__repr__()

        str += (f"\n"
                f"g = {self.g:.4e}\n")

        return str

    class Parameters(ResonatorFluxoniumInductive.Parameters):

        def __init__(self, resonator_fluxonium):

            ResonatorFluxoniumInductive.Parameters.__init__(self, resonator_fluxonium)

            self.g = resonator_fluxonium.g


class ResonatorFluxoniumInductiveNormal(ResonatorFluxoniumInductive):

    def __init__(self):

        ResonatorFluxoniumInductive.__init__(self)

        self._coupling = "inductive"
        self._basis = "normal"

        self.special_integrals_1 = sp.SpecialIntegrals()
        self.special_integrals_2 = sp.SpecialIntegrals()

    def _init_look_up_table(self, Na, Nr):
        """
        Private routine that initializes the integral look-up table.

        :param Na: number of fluxonium basis states.
        :param Nr: number of resonator basis states.
        :return: None.
        """

        self.special_integrals_1.init_look_up_table_fluxonium(Nr)
        self.special_integrals_2.init_look_up_table_fluxonium(Na)

    def _destroy_look_up_table(self):
        """
        Destruction of the integral look-up table.

        :return: None.
        """

        self.special_integrals_1.destroy_look_up_table()
        self.special_integrals_2.destroy_look_up_table()

    ##################
    #####  core  #####
    ##################

    def calc_resonator_dipole_moments(self, state1, state2):
        """
        Calculation of the resonator flux and charge dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: first state.
        :param state2: second state.
        :return: absolute flux dipole moment, absolute charge dipole moment.
        """

        # qr = S11 * q1 + S12 * q2
        b_daggar = np.sum(self.sqrts_r * self.v[self.Na:, state1] * self.v[:-self.Na, state2])
        b = np.sum(self.sqrts_r * self.v[:-self.Na, state1] * self.v[self.Na:, state2])

        a_daggar = np.sum(self.sqrts_a * self.v[1:, state1] * self.v[:-1, state2])
        a = np.sum(self.sqrts_a * self.v[:-1, state1] * self.v[1:, state2])

        fdm = self.flux_zpf[0, 0] * (b_daggar + b) + self.flux_zpf[0, 1] * (a_daggar + a)
        qdm = self.charge_zpf[0, 0] * (b_daggar - b) + self.charge_zpf[0, 1] * (a_daggar - a)

        return np.abs(fdm), np.abs(qdm)

    def calc_atom_dipole_moments(self, state1, state2):
        """
        Calculation of the fluxonium flux and charge dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: first state.
        :param state2: second state.
        :return: absolute flux dipole moment, absolute charge dipole moment.
        """

        # qr = S11 * q1 + S12 * q2
        b_daggar = np.sum(self.sqrts_r * self.v[self.Na:, state1] * self.v[:-self.Na, state2])
        b = np.sum(self.sqrts_r * self.v[:-self.Na, state1] * self.v[self.Na:, state2])

        a_daggar = np.sum(self.sqrts_a * self.v[1:, state1] * self.v[:-1, state2])
        a = np.sum(self.sqrts_a * self.v[:-1, state1] * self.v[1:, state2])

        fdm = self.flux_zpf[1, 0] * (b_daggar + b) + self.flux_zpf[1, 1] * (a_daggar + a)
        qdm = self.charge_zpf[1, 0] * (b_daggar - b) + self.charge_zpf[1, 1] * (a_daggar - a)

        return np.abs(fdm), np.abs(qdm)

    def calc_hamiltonian_parameters(self):
        """
        Calculation of all relevant Hamiltonian parameters.

        :return: None.
        """

        ResonatorFluxoniumInductive.calc_hamiltonian_parameters(self)

        self.flux_zpf = np.zeros((2, 2))
        self.flux_zpf[0, 0] = np.sqrt(self.Ecr / (2 * self.wrp)) / (2 * np.pi) * self.S[0, 0]  # in Phi_0
        self.flux_zpf[0, 1] = np.sqrt(self.Ecr / (2 * self.wap)) / (2 * np.pi) * self.S[0, 1]  # in Phi_0
        self.flux_zpf[1, 0] = np.sqrt(self.Eca / (2 * self.wrp)) / (2 * np.pi) * self.S[1, 0]  # in Phi_0
        self.flux_zpf[1, 1] = np.sqrt(self.Eca / (2 * self.wap)) / (2 * np.pi) * self.S[1, 1]  # in Phi_0

        self.charge_zpf = np.zeros((2, 2))
        self.charge_zpf[0, 0] = np.sqrt(self.wrp / (2 * self.Ecr)) * self.S[0, 0]  # in 2 * e
        self.charge_zpf[0, 1] = np.sqrt(self.wap / (2 * self.Ecr)) * self.S[0, 1]  # in 2 * e
        self.charge_zpf[1, 0] = np.sqrt(self.wrp / (2 * self.Eca)) * self.S[1, 0]  # in 2 * e
        self.charge_zpf[1, 1] = np.sqrt(self.wap / (2 * self.Eca)) * self.S[1, 1]  # in 2 * e

    def diagonalize_hamiltonian(self):
        """
        Numerical diagonalization of the system Hamiltonian.

        :return: None.
        """

        self.H = np.zeros((self.Nr * self.Na, self.Nr * self.Na))

        flux_q1 = np.sqrt(8) * np.pi * self.flux_zpf[1, 0]
        flux_q2 = np.sqrt(8) * np.pi * self.flux_zpf[1, 1]

        for m in range(self.Nr * self.Na):

            i = m // self.Na
            g = m % self.Na

            for n in range(0, m):
                j = n // self.Na
                h = n % self.Na

                self.H[m, n] = - self.Ej * (
                        np.cos(self.p_ext) * (
                                self.special_integrals_1.cos_integral(i, j, flux_q1) *
                                self.special_integrals_2.cos_integral(g, h, flux_q2) -
                                self.special_integrals_1.sin_integral(i, j, flux_q1) *
                                self.special_integrals_2.sin_integral(g, h, flux_q2)) +
                        np.sin(self.p_ext) * (
                                self.special_integrals_1.sin_integral(i, j, flux_q1) *
                                self.special_integrals_2.cos_integral(g, h, flux_q2) +
                                self.special_integrals_1.cos_integral(i, j, flux_q1) *
                                self.special_integrals_2.sin_integral(g, h, flux_q2)))

                self.H[n, m] = self.H[m, n]

            self.H[m, m] = self.wrp * (i + 0.5) + self.wap * (g + 0.5) - self.Ej * (
                    np.cos(self.p_ext) * (
                            self.special_integrals_1.cos_integral(i, i, flux_q1) *
                            self.special_integrals_2.cos_integral(g, g, flux_q2) -
                            self.special_integrals_1.sin_integral(i, i, flux_q1) *
                            self.special_integrals_2.sin_integral(g, g, flux_q2)) +
                    np.sin(self.p_ext) * (
                            self.special_integrals_1.sin_integral(i, i, flux_q1) *
                            self.special_integrals_2.cos_integral(g, g, flux_q2) +
                            self.special_integrals_1.cos_integral(i, i, flux_q1) *
                            self.special_integrals_2.sin_integral(g, g, flux_q2)))

        super().diagonalize_hamiltonian()

        # calculate the creator an annihilator array
        self._creator_annihilator()


#############################
#### Capacitive coupling ####
#############################

class ResonatorFluxoniumCapacitive(ResonatorAtom):

    def __init__(self):

        ResonatorAtom.__init__(self)

        # Resonator Fluxonium Parameters
        self.Lr = 0.0
        self.La = 0.0
        self.Cr = 0.0
        self.Ca = 0.0
        self.Cs = 0.0

        self.Elr = 0.0
        self.Ela = 0.0
        self.Ecr = 0.0
        self.Eca = 0.0
        self.Ecs = 0.0
        self.Ej = 0
        self.Ejs = 0
        self.Ejd = 0
        self.ratio = 0
        self.p_ext = 0

        self.wr = 0.0
        self.wrp = 0.0
        self.wa = 0.0
        self.wap = 0.0
        self.g_lin_sq = 0.0

        self.S = None
        self.flux_zpf = None
        self.charge_zpf = None

        # Hamiltonian
        self.sqrts_r = None
        self.sqrts_a = None

    def set_parameters(self, Lr=None, La=None, Cs=None, Cr=None, Ca=None, Ej=None, Ejs=None, Ejd=None, ratio=None,
                       p_ext=None, Na=None, Nr=None):
        """
        Helper routine to set relevant fluxonium parameters.
        When updating inductances or capacitances self.calc_hamiltonian_parameters() has to be called.

        :param Lr: resonator inductance in [H].
        :param La: qubit inductance in [H].
        :param Cr: resonator capacitance (including the shared capacitance) in [F].
        :param Ca: fluxonium capacitance (including the shared capacitance) in [F].
        :param Cs: shared capacitance in [F].
        :param Ej: Josephson energy in [GHz].
        :param Ejs: sum of the Josephson energies of the SQUID [GHz].
        :param Ejd: difference of the Josephson energies (outer junction  - inner junction) of the SQUID [GHz].
        :param ratio: ratio of the loop areas - fluxonium loop area (enclosed by the inner junction) / SQUID loop area.
        :param p_ext: external flux bias in [rad].
        :param Nr: number of resonator basis states.
        :param Nr: number of qubit basis states.
        :return: None.
        """

        if Lr is not None:
            self.Lr = Lr
        if La is not None:
            self.La = La
        if Cs is not None:
            self.Cs = Cs
        if Cr is not None:
            self.Cr = Cr
        if Ca is not None:
            self.Ca = Ca
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
        if Na is not None:
            self.Na = Na
        if Nr is not None:
            self.Nr = Nr

    def set_flux_squid(self, p_ext):
        """
        Calculation the effective Josephson energy and effective flux bias.

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
        Sweep of the externally applied magentic flux.

        :param pext_sweep: 1D numpy array with the external flux bias in [rad].
        :return: None.
        """

        self._initialize_sweep(pext_sweep)

        self._init_look_up_table(self.Na, self.Nr)
        for i in range(self.steps):
            self.p_ext = self.par_sweep[i]
            self._calc_sweep(i)
        self._destroy_look_up_table()

    def sweep_external_flux_squid(self, pext_sweep):
        """
        Sweep of the externally applied global magentic flux to a SQUID-junction fluxonium.

        The routine assumes that sum of the SQUID-junction Josephson energies self.Ejs and asymmetry self.Ejd as well as
        the ratio of fluxonium loop area to SQUID-area (self.ratio) have been defined.
        For further details see explanation of self.set_flux_squid().

        :param pext_sweep: 1D numpy array with the external flux bias in [rad].
        :return: None.
        """

        self._initialize_sweep(pext_sweep)

        self._init_look_up_table(self.Na, self.Nr)
        for i in range(self.steps):
            self.set_flux_squid(self.par_sweep[i])
            self._calc_sweep(i)
        self._destroy_look_up_table()

    def sweep_parameter(self, par_sweep, par_name=None):
        """
        Sweep of system parameters.

        :param par_sweep: either dict with keys "Lr", "La", "Cr", "Cs", "Ca", "Ej", "Ejs", "Ejd", "ratio",
                          "p_ext" and corresponding 1D numpy parameter arrays,
                          or 1D numpy array with parameter sweep for the parameter specified by par_name.
        :param par_name: "Lr", "La", "Cr", "Ca", "Cs", "Ej", "Ejs", "Ejd", "ratio", "p_ext".
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
            if par_name in ["Lr", "La", "Ls", "Cr", "Ca"]:

                self._initialize_sweep(par_sweep)
                for i in range(self.steps):
                    setattr(self, par_name, self.par_sweep[i])
                    self.calc_hamiltonian_parameters()
                    self._calc_sweep(i)

            elif par_name in ["Ej", "p_ext"]:

                self._initialize_sweep(par_sweep)
                self._init_look_up_table(self.Na, self.Nr)
                for i in range(self.steps):
                    setattr(self, par_name, self.par_sweep[i])
                    self._calc_sweep(i)
                self._destroy_look_up_table()

            elif par_name in ["Ejs", "Ejd", "ratio"]:

                self._initialize_sweep(par_sweep)
                self._init_look_up_table(self.Na, self.Nr)
                for i in range(self.steps):
                    setattr(self, par_name, self.par_sweep[i])
                    self.set_flux_squid(self.p_ext)
                    self._calc_sweep(i)
                self._destroy_look_up_table()

            else:
                print("Parameter does not exist or should not be swept in this routine. \n" +
                      "For instance, for sweeping the number of basis functions use the convergence_sweep routine.")

    def convergence_sweep(self, N_sweep):
        """
        Convergences of the wave functions and corresponding energies with the number of basis states.

        :param N_sweep: 2D numpy array of numbers of basis states of the form [(Na, Nr), ...].
        :return: None.
        """

        self.steps = N_sweep.shape[0]
        self.par_sweep = N_sweep
        size = np.max(N_sweep[:, 0] * N_sweep[:, 1])
        self.system_pars_sweep = np.zeros(self.steps, dtype=object)
        self.H_sweep = np.full((size, size, self.steps), np.nan)
        self.E_sweep = np.full((size, self.steps), np.nan)
        self.v_sweep = np.full((size, size, self.steps), np.nan)

        self._init_look_up_table(np.max(N_sweep[:, 0]), np.max(N_sweep[:, 1]))
        for i in range(self.steps):
            self.Na = N_sweep[i, 0]
            self.Nr = N_sweep[i, 1]
            self._calc_sweep(i)
        self._destroy_look_up_table()

    def inspect_sweep(self, step):
        """
        Reset system to a specific sweep step. The routine assumes that the sweep has already been performed.

        :param: step: step of the sweep.
        :return: None.
        """

        self.Lr = self.system_pars_sweep[step].Lr
        self.La = self.system_pars_sweep[step].La
        self.Cs = self.system_pars_sweep[step].Cs
        self.Cr = self.system_pars_sweep[step].Cr
        self.Ca = self.system_pars_sweep[step].Ca
        self.Elr = self.system_pars_sweep[step].Elr
        self.Ela = self.system_pars_sweep[step].Ela
        self.Ecr = self.system_pars_sweep[step].Ecr
        self.Eca = self.system_pars_sweep[step].Eca
        self.Ecs = self.system_pars_sweep[step].Ecs
        self.Ej = self.system_pars_sweep[step].Ej
        self.Ejs = self.system_pars_sweep[step].Ejs
        self.Ejd = self.system_pars_sweep[step].Ejd
        self.p_ext = self.system_pars_sweep[step].p_ext
        self.Na = self.system_pars_sweep[step].Na
        self.Nr = self.system_pars_sweep[step].Nr
        self._creator_annihilator()

        self.wr = self.system_pars_sweep[step].wr
        self.wrp = self.system_pars_sweep[step].wrp
        self.wa = self.system_pars_sweep[step].wa
        self.wap = self.system_pars_sweep[step].wap
        self.g_lin_sq = self.system_pars_sweep[step].g_lin_sq

        self.S = self.system_pars_sweep[step].S
        self.flux_zpf = self.system_pars_sweep[step].flux_zpf
        self.charge_zpf = self.system_pars_sweep[step].charge_zpf

        super().inspect_sweep(step)

    def _calc_sweep(self, step):
        """
        Private routine that diagonalizes the Hamiltonian and stores the data of the given sweep step.

        :param step: step of the parameter sweep
        :return: None
        """

        self.system_pars_sweep[step] = self.Parameters(self)
        super()._calc_sweep(step)

    def _init_look_up_table(self, Na, Nr):  # will be overwritten
        pass

    def _destroy_look_up_table(self):  # will be overwritten
        pass

    ############################
    #####  sweep analysis  #####
    ############################

    ##################
    #####  plots  #####
    ##################

    def plot_potential(self, ax, xy_range=2.0, unit_mass=True):
        """
        3D plot of the resonator fluxonium potential.

        :param ax: matplotlib axes instance with projection="3d".
        :param xy_range: scaler or tupel of the x, y range that will be plotted.
        :param unit_mass: if False the flux potential is plotted in units of Phi_0.
                          if True the potential is transformed to correspond to a particle of uniform and unit mass and
                          is plotted in units of 1 / sqrt(GHz).
        :return: None.
        """

        sample = 1001
        if hasattr(xy_range, '__iter__'):
            x = np.linspace(-xy_range[0], xy_range[0], sample)  # resonator
            y = np.linspace(-xy_range[1], xy_range[1], sample)  # qubit
        else:
            x = np.linspace(-xy_range, xy_range, sample)
            y = np.linspace(-xy_range, xy_range, sample)

        x, y = np.meshgrid(x, y, sparse=False, indexing='xy')

        if not unit_mass:

            z = (4 * np.pi ** 2 * self.Elr ** 2 * x ** 2 / 2 + 4 * np.pi ** 2 * self.Ela ** 2 * y ** 2 / 2
                 - self.Ej * np.cos(2 * np.pi * y))

            ax.plot_wireframe(x, y, z)

            ax.set_xlabel(r"$\phi_\mathrm{r}$ $(\Phi_0)$")
            ax.set_ylabel(r"$\phi_\mathrm{q}$ $(\Phi_0)$")

        else:

            wr_sq, wa_sq, g_lin_sq = self.cholesky_transformation()

            z = (wr_sq * x ** 2 / 2 + wa_sq * y ** 2 / 2 + g_lin_sq * x * y
                 - self.Ej * np.cos(np.sqrt(self.Eca) * y))

            ax.plot_wireframe(x, y, z)

            ax.set_xlabel(r"$x_\mathrm{r}$ $(1 / \sqrt{GHz})$")
            ax.set_ylabel(r"$x_\mathrm{q}$ $(1 / \sqrt{GHz})$")

        ax.set_zlabel(r"$E$ (GHz)")

    ##################
    #####  core  #####
    ##################

    def _creator_annihilator(self):
        """
        Private routine that creates the creation and annihilations operators.

        :return: None.
        """

        self.sqrts_r = np.zeros((self.Nr - 1) * self.Na)
        for i in range((self.Nr - 1) * self.Na):
            r = i // self.Na
            self.sqrts_r[i] = np.sqrt(r + 1)

        self.sqrts_a = np.zeros(self.Nr * self.Na - 1)
        for i in range(self.Nr * self.Na - 1):
            q = i % self.Na
            if q < self.Na - 1:
                self.sqrts_a[i] = np.sqrt(q + 1)

    def calc_hamiltonian_parameters(self):
        """
        Calculation of all relevant Hamiltonian parameters.

        :return: None.
        """

        # Calculate energies of the fluxonium Hamiltonian expressed with the number and phase operator.
        # El and Ec are scaled to be in units of 1GHz. Ej must be given in GHz.
        # Note that the natural definition of Ec is used such that w = sqrt(Ec * El)

        self.Elr = pyc.h / (16e9 * np.pi ** 2 * pyc.e ** 2) / self.Lr
        self.Ela = pyc.h / (16e9 * np.pi ** 2 * pyc.e ** 2) / self.La

        # Compute capacitance matrix
        C = self.Cr * self.Ca - self.Cs ** 2
        Cr = self.Ca / C
        Ca = self.Cr / C
        Cs = self.Cs / C

        self.Ecr = 4e-9 * pyc.e**2 / pyc.h * Cr
        self.Eca = 4e-9 * pyc.e ** 2 / pyc.h * Ca
        self.Ecs = 4e-9 * pyc.e**2 / pyc.h * Cs

        # Calculated Hamiltonian parameters for the natural harmonic oscillator basis, i.e., the particle has unit mass
        wr_sq = self.Ecr * self.Elr
        wa_sq = self.Eca * self.Ela
        self.g_lin_sq = self.Ecs * np.sqrt(self.Elr * self.Ela)

        self.wr = np.sqrt(wr_sq)
        self.wa = np.sqrt(wa_sq)

        sqrt = np.sqrt((wr_sq - wa_sq)**2 + 4 * self.g_lin_sq**2)

        if wr_sq >= wa_sq:
            self.wrp = np.sqrt(0.5 * (wr_sq + wa_sq + sqrt))  # resonator like
            self.wap = np.sqrt(0.5 * (wr_sq + wa_sq - sqrt))  # qubit like
            s1 = (wr_sq - wa_sq) + sqrt
            s2 = - 2 * self.g_lin_sq
        else:
            self.wrp = np.sqrt(0.5 * (wr_sq + wa_sq - sqrt))  # resonator like
            self.wap = np.sqrt(0.5 * (wr_sq + wa_sq + sqrt))  # qubit like
            s1 = (wa_sq - wr_sq) + sqrt
            s2 = 2 * self.g_lin_sq

        self.S = np.array([[s1, s2], [- s2, s1]]) / np.sqrt(s1**2 + s2**2)

        self.wr_approx = self.wrp  # for sorting the levels

    def cholesky_transformation(self):
        """
        Cholesky decomposition of the Ec matrix.
        This allows to transform the Hamiltonian to a particle with uniform and unit mass.

        :return: wr2, wa2, g_lin_sq -  the entries of the harmonic potential matrix hbar**2 omega**2 in [GHz].
        """

        wr_sq = self.Elr * self.Ecr - self.Elr * self.Ecs**2 / self.Eca
        wa_sq = self.Ela * self.Eca + self.Elr * self.Ecs ** 2 / self.Eca
        g_lin_sq = self.Elr * self.Ecs * np.sqrt(self.Ecr / self.Eca + self.Ecs ** 2 / self.Eca ** 2)

        return wr_sq, wa_sq, g_lin_sq

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
        d.move(1.8, 0.0)
        d += elm.Line().left(length=1.8)
        d += elm.Inductor().up().label(r"$L_\mathrm{r}$")
        d += elm.Line().right(length=1.8)
        d += elm.Capacitor().down().label(r"$C_\mathrm{r} - C_\mathrm{s}$")
        d += elm.Line().right(length=1.8)
        d += elm.Capacitor().up().label(r"$C_\mathrm{a} - C_\mathrm{s}$")
        d += elm.Capacitor().left(length=1.8).label(r"$C_\mathrm{s}$")
        d.move(1.8, 0.0)
        d += elm.Line().right(length=1.2)
        d += elm.Josephson().down().label(r"$E_\mathrm{J}$")
        d += elm.Line().left(length=1.2)
        d.move(1.2, 2.0)
        d += elm.Line().right(length=1.5)
        d += elm.Inductor().down().label(r"$L_\mathrm{a}$")
        d += elm.Line().left(length=1.5)

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
        fig, ax = plt.subplots(1, 1, figsize=(figwidth, 7.5 / 10.0 * figwidth))
        ax.axis("off")

        ax.text(0.5, 2.0, "Resonator fluxonium Hamiltonian:", va="top", fontsize=12)
        #ax.text(0.75, 1.42, "$H$", va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(1.0, 1.5, r"$H =\frac{1}{2}\frac{1}{C_\mathrm{r}C_\mathrm{a} - C_\mathrm{s}^2}\mathbf{q}^T "
                r"\genfrac{(}{)}{0}{1}{\,C_\mathrm{a}\;\,\,C_\mathrm{s}}{\,C_\mathrm{s}\;\;\,C_\mathrm{r}}\mathbf{q}"
                r"+ \frac{1}{2}\mathbf{\phi}^T"
                r"\genfrac{(}{)}{0}{1}{\frac{1}{L_\mathrm{r}}\;\;\;\;0}{\;\,0\;\;\;\;\frac{1}{L_\mathrm{a}}}\mathbf{\phi}"
                r" - E_\mathrm{J}\cos\left(\frac{2 \pi}{\Phi_0}\phi_\mathrm{a}\right),$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(0.5, 0.6, r"with", va="top", fontsize=12)
        ax.text(1.05, 0.66, r"$\mathbf{\phi} = (\phi_\mathrm{r}, \phi_\mathrm{a} - \Phi_\mathrm{ext})^T,\quad"
                r"\mathbf{q} = (q_\mathrm{r}, q_\mathrm{a})^T.$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        ax.text(0.5, -0.0, "A time-independet external flux allows shifting the coordinate system.\n"
                "With dimensionless operators the Hamiltonian reads:",
                va="top", fontsize=12)
        ax.text(0.75, -0.91, "$H$", va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(1.0, -0.8, r"$=\frac{1}{2}\mathbf{n}^T "
                r"\genfrac{(}{)}{0}{1}{E_{C_\mathrm{r}}\;\;\;E_{C_\mathrm{s}}}"
                r"{\,E_{C_\mathrm{s}}\;\;\,E_{C_\mathrm{q}}}\mathbf{n}"
                r" + \mathbf{\varphi}^T\genfrac{(}{)}{0}{1}{\!E_{L_\mathrm{r}}\;\;\;\;0}"
                r"{\;\,0\;\;\;\;\;E_{L_\mathrm{a}}}\mathbf{\varphi}"
                r" - E_\mathrm{J}\cos\left(\varphi_\mathrm{a} + \varphi_\mathrm{ext}\right)$" + "\n" +
                r"$=\frac{\hbar^2}{2}\mathbf{p}^T"
                r"\genfrac{(}{)}{0}{1}{\;\omega_\mathrm{r}^2\;\;\;\;\;g_\mathrm{lin}^2}"
                r"{\,g_\mathrm{lin}^2\;\;\;\;\,\omega_\mathrm{a}^2}\mathbf{p}"
                r"+ \frac{1}{2}\mathbf{x}^T\mathbf{x} - E_\mathrm{J}\cos\left(1 / \sqrt{E_{L_\mathrm{a}}} x_\mathrm{a} "
                r"+ \varphi_\mathrm{ext}\right),$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(0.5, -2.26, r"with", va="top", fontsize=12)
        ax.text(1.05, -2.14, r"$\omega_\mathrm{r} = \sqrt{E_{L_\mathrm{r}} E_{C_\mathrm{r}}},\quad"
                r"\omega_\mathrm{a} = \sqrt{E_{L_\mathrm{a}} E_{C_\mathrm{a}}},\quad"
                r"g_\mathrm{lin}^2 = E_{C_\mathrm{s}}\sqrt{E_{L_\mathrm{r}} E_{L_\mathrm{a}}}.$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(0.5, -2.9, "A coordinate transformation to the normal basis yields:",
                va="top", fontsize=12)
        ax.text(1.0, -3.4, r"$H=\frac{\hbar^2}{2}\mathbf{p}'^T"
                r"\genfrac{(}{)}{0}{1}{\!\!\!\!\omega_\mathrm{r}'^2\;\;\;\;0}"
                r"{\;\,0\;\;\;\;\;\;\omega_\mathrm{a}'^2}\mathbf{p}' + "
                r"\frac{1}{2}\mathbf{x}'^T\mathbf{x}' - E_\mathrm{J}\cos\left(u x_\mathrm{r}' + v x_\mathrm{a}' + \varphi_\mathrm{ext}\right).$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')
        ax.text(0.5, -4.2, r"Alternatively, Cholesky decomposing the $\mathbf{E}_C$-matrix yields:",
                va="top", fontsize=12)
        ax.text(1.0, -4.7, r"$H=\frac{1}{2}\mathbf{p}^T\mathbf{p} + \frac{\hbar^2}{2}\mathbf{x}^T"
                r"\genfrac{(}{)}{0}{1}{\;\omega_\mathrm{r}^2\;\;\;\;\;g_\mathrm{lin}^2}"
                r"{\,g_\mathrm{lin}^2\;\;\;\;\,\omega_\mathrm{a}^2}\mathbf{x}"
                r" - E_\mathrm{J}\cos\left(\sqrt{E_{C_\mathrm{a}}} x_\mathrm{a} + \varphi_\mathrm{ext}\right).$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        #ax.axvline(0.0)
        #ax.axvline(10.0)
        #ax.axhline(-5.4)
        #ax.axhline(2.1)

        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(- 5.4, 2.1)

        return fig

    def __repr__(self):
        """
        Returns a representation of the resonator fluxonium parameters.

        :return: string.
        """

        return (f"Lr = {self.Lr:.4e}\n"
                f"La = {self.La:.4e}\n"
                f"Cr = {self.Cr:.4e}\n"
                f"Ca = {self.Ca:.4e}\n"
                f"Cs = {self.Cs:.4e}\n"
                f"\n"
                f"Elr = {self.Elr:.4e}\n"
                f"Ela = {self.Ela:.4e}\n"
                f"Ecr = {self.Ecr:.4e}\n"
                f"Eca = {self.Eca:.4e}\n"
                f"Ecs = {self.Ecs:.4e}\n"
                f"Ej = {self.Ej:.4e}\n"
                f"Ejs = {self.Ejs:.4e}\n"
                f"Ejd = {self.Ejd:.4e}\n"
                f"ratio = {self.ratio:.4e}\n"
                f"\n"
                f"wr = {self.wr:.4e}\n"
                f"wrp = {self.wrp:.4e}\n"
                f"wa = {self.wa:.4e}\n"
                f"wap = {self.wap:.4e}\n"
                f"g_lin_sq = {self.g_lin_sq:.4e}\n"
                f"\n"
                f"S = [[{self.S[0, 0]:.4e}, {self.S[0, 1]:.4e}], [{self.S[1, 0]:.4e}, {self.S[1, 1]:.4e}]]\n"
                f"flux_zpf = [[{self.flux_zpf[0, 0]:.4e}, {self.flux_zpf[0, 1]:.4e}], "
                f"[{self.flux_zpf[1, 0]:.4e}, {self.flux_zpf[1, 1]:.4e}]]\n"
                f"charge_zpf = [[{self.charge_zpf[0, 0]:.4e}, {self.charge_zpf[0, 1]:.4e}], "
                f"[{self.charge_zpf[1, 0]:.4e}, {self.charge_zpf[1, 1]:.4e}]]\n"
                )

    class Parameters:

        def __init__(self, resonator_fluxonium):

            self.Lr = resonator_fluxonium.Lr
            self.La = resonator_fluxonium.La
            self.Cr = resonator_fluxonium.Cr
            self.Ca = resonator_fluxonium.Ca
            self.Cs = resonator_fluxonium.Cs
            self.Elr = resonator_fluxonium.Elr
            self.Ela = resonator_fluxonium.Ela
            self.Ecr = resonator_fluxonium.Ecr
            self.Eca = resonator_fluxonium.Eca
            self.Ecs = resonator_fluxonium.Ecs
            self.Ej = resonator_fluxonium.Ej
            self.Ejs = resonator_fluxonium.Ejs
            self.Ejd = resonator_fluxonium.Ejd
            self.p_ext = resonator_fluxonium.p_ext
            self.Na = resonator_fluxonium.Na
            self.Nr = resonator_fluxonium.Nr

            self.wr = resonator_fluxonium.wr
            self.wrp = resonator_fluxonium.wrp
            self.wa = resonator_fluxonium.wa
            self.wap = resonator_fluxonium.wap
            self.g_lin_sq = resonator_fluxonium.g_lin_sq

            self.S = resonator_fluxonium.S
            self.flux_zpf = resonator_fluxonium.flux_zpf
            self.charge_zpf = resonator_fluxonium.charge_zpf


class ResonatorFluxoniumCapacitiveProduct(ResonatorFluxoniumCapacitive):

    def __init__(self):

        ResonatorFluxoniumCapacitive.__init__(self)

        self._coupling = "capacitive"
        self._basis = "product"

        self.g = 0.0

        self.special_integrals = sp.SpecialIntegrals()

    def inspect_sweep(self, step):
        """
        Reset system to a specific sweep step. The routine assumes that the sweep has already been performed.

        :param: step: step of the parameter sweep.
        :return: None.
        """

        super().inspect_sweep(step)

        self.g = self.system_pars_sweep[step].g

    def _init_look_up_table(self, Na, Nr):
        """
        Private routine that initializes the integral look-up table.

        :param Na: number of fluxonium basis states.
        :param Nr: not used, dummy from the parent class
        :return: None
        """
        self.special_integrals.init_look_up_table_fluxonium(Na)

    def _destroy_look_up_table(self):
        """
        Destruction of the integral look-up table.

        :return: None.
        """

        self.special_integrals.destroy_look_up_table()

    def calc_resonator_dipole_moments(self, state1, state2):
        """
        Calculation of the resonator flux and charge dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: first state.
        :param state2: second state.
        :return: absolute flux dipole moment, absolute charge dipole moment.
        """

        b_daggar = np.sum(self.sqrts_r * self.v[self.Na:, state1] * self.v[:-self.Na, state2])
        b = np.sum(self.sqrts_r * self.v[:-self.Na, state1] * self.v[self.Na:, state2])

        return np.abs(self.flux_zpf[0, 0] * (b_daggar + b)), np.abs(self.charge_zpf[1, 1] * (b_daggar - b))

    def calc_atom_dipole_moments(self, state1, state2):
        """
        Calculation of the fluxonium flux and charge dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: first state.
        :param state2: second state.
        :return: absolute flux dipole moment, absolute charge dipole moment.
        """

        a_daggar = np.sum(self.sqrts_a * self.v[1:, state1] * self.v[:-1, state2])
        a = np.sum(self.sqrts_a * self.v[:-1, state1] * self.v[1:, state2])

        return np.abs(self.flux_zpf[1, 1] * (a_daggar + a)), np.abs(self.charge_zpf[1, 1] * (a_daggar - a))

    def calc_hamiltonian_parameters(self):
        """
        Calculation of all relevant Hamiltonian parameters.

        :return: None.
        """

        ResonatorFluxoniumCapacitive.calc_hamiltonian_parameters(self)

        Zr = np.sqrt(self.Ecr) / (2 * np.pi * np.sqrt(self.Elr))  # in R_Q
        Zq = np.sqrt(self.Eca) / (2 * np.pi * np.sqrt(self.Ela))  # in R_Q

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

        self.H = np.zeros((self.Nr * self.Na, self.Nr * self.Na))

        flux_q = np.sqrt(8) * np.pi * self.flux_zpf[1, 1]

        for m in range(self.Nr * self.Na):

            i = m // self.Na
            g = m % self.Na

            for n in range(m):

                j = n // self.Na
                h = n % self.Na

                if i == j:
                    self.H[m, n] = - self.Ej * (
                            np.cos(self.p_ext) * self.special_integrals.cos_integral(g, h, flux_q) +
                            np.sin(self.p_ext) * self.special_integrals.sin_integral(g, h, flux_q))

                if j == i - 1 and h == g - 1:
                    self.H[m, n] -= self.g * np.sqrt(i) * np.sqrt(g)
                if j == i - 1 and h == g + 1:
                    self.H[m, n] += self.g * np.sqrt(i) * np.sqrt(h)

                self.H[n, m] = self.H[m, n]

            self.H[m, m] = self.wr * (i + 0.5) + self.wa * (g + 0.5) - self.Ej * (
                    np.cos(self.p_ext) * self.special_integrals.cos_integral(g, g, flux_q) +
                    np.sin(self.p_ext) * self.special_integrals.sin_integral(g, g, flux_q))

        super().diagonalize_hamiltonian()

        # calculate the creator an annihilator array
        self._creator_annihilator()

    def __repr__(self):
        """
        Returns a representation of the resonator fluxonium parameters.

        :return: string
        """

        str = super().__repr__()

        str += (f"\n"
                f"g = {self.g:.4e}\n")

        return str

    class Parameters(ResonatorFluxoniumCapacitive.Parameters):

        def __init__(self, resonator_fluxonium):

            ResonatorFluxoniumCapacitive.Parameters.__init__(self, resonator_fluxonium)

            self.g = resonator_fluxonium.g


class ResonatorFluxoniumCapacitiveNormal(ResonatorFluxoniumCapacitive):

    def __init__(self):

        self._coupling = "capacitive"
        self._basis = "normal"

        ResonatorFluxoniumCapacitive.__init__(self)

        self.special_integrals_1 = sp.SpecialIntegrals()
        self.special_integrals_2 = sp.SpecialIntegrals()

    def _init_look_up_table(self, Na, Nr):
        """
        Private routine that initializes the integral look-up table.

        :param Na: number of fluxonium basis states.
        :param Nr: number of resonator basis states.
        :return: None.
        """

        self.special_integrals_1.init_look_up_table_fluxonium(Nr)
        self.special_integrals_2.init_look_up_table_fluxonium(Na)

    def _destroy_look_up_table(self):
        """
        Destruction of the integral look-up table.

        :return: None.
        """

        self.special_integrals_1.destroy_look_up_table()
        self.special_integrals_2.destroy_look_up_table()

    ##################
    #####  core  #####
    ##################

    def calc_resonator_dipole_moments(self, state1, state2):
        """
        Calculation of the resonator flux and charge dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: first state.
        :param state2: second state.
        :return: absolute flux dipole moment, absolute charge dipole moment.
        """

        # qr = S11 * q1 + S12 * q2
        b_daggar = np.sum(self.sqrts_r * self.v[self.Na:, state1] * self.v[:-self.Na, state2])
        b = np.sum(self.sqrts_r * self.v[:-self.Na, state1] * self.v[self.Na:, state2])

        a_daggar = np.sum(self.sqrts_a * self.v[1:, state1] * self.v[:-1, state2])
        a = np.sum(self.sqrts_a * self.v[:-1, state1] * self.v[1:, state2])

        fdm = self.flux_zpf[0, 0] * (b_daggar + b) + self.flux_zpf[0, 1] * (a_daggar + a)
        qdm = self.charge_zpf[0, 0] * (b_daggar - b) + self.charge_zpf[0, 1] * (a_daggar - a)

        return np.abs(fdm), np.abs(qdm)

    def calc_atom_dipole_moments(self, state1, state2):
        """
        Calculation of the fluxonium flux and charge dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: first state.
        :param state2: second state.
        :return: absolute flux dipole moment, absolute charge dipole moment.
        """

        # qr = S11 * q1 + S12 * q2
        b_daggar = np.sum(self.sqrts_r * self.v[self.Na:, state1] * self.v[:-self.Na, state2])
        b = np.sum(self.sqrts_r * self.v[:-self.Na, state1] * self.v[self.Na:, state2])

        a_daggar = np.sum(self.sqrts_a * self.v[1:, state1] * self.v[:-1, state2])
        a = np.sum(self.sqrts_a * self.v[:-1, state1] * self.v[1:, state2])

        fdm = self.flux_zpf[1, 0] * (b_daggar + b) + self.flux_zpf[1, 1] * (a_daggar + a)
        qdm = self.charge_zpf[1, 0] * (b_daggar - b) + self.charge_zpf[1, 1] * (a_daggar - a)

        return np.abs(fdm), np.abs(qdm)

    def calc_hamiltonian_parameters(self):
        """
        Calculation of all relevant Hamiltonian parameters.

        :return: None.
        """

        ResonatorFluxoniumCapacitive.calc_hamiltonian_parameters(self)

        self.flux_zpf = np.zeros((2, 2))
        self.flux_zpf[0, 0] = np.sqrt(self.wrp / (2 * self.Elr)) / (2 * np.pi) * self.S[0, 0]  # in Phi_0
        self.flux_zpf[0, 1] = np.sqrt(self.wap / (2 * self.Elr)) / (2 * np.pi) * self.S[0, 1]  # in Phi_0
        self.flux_zpf[1, 0] = np.sqrt(self.wrp / (2 * self.Ela)) / (2 * np.pi) * self.S[1, 0]  # in Phi_0
        self.flux_zpf[1, 1] = np.sqrt(self.wap / (2 * self.Ela)) / (2 * np.pi) * self.S[1, 1]  # in Phi_0

        self.charge_zpf = np.zeros((2, 2))
        self.charge_zpf[0, 0] = np.sqrt(self.Elr / (2 * self.wrp)) * self.S[0, 0]  # in 2 * e
        self.charge_zpf[0, 1] = np.sqrt(self.Elr / (2 * self.wap)) * self.S[0, 1]  # in 2 * e
        self.charge_zpf[1, 0] = np.sqrt(self.Ela / (2 * self.wrp)) * self.S[1, 0]  # in 2 * e
        self.charge_zpf[1, 1] = np.sqrt(self.Ela / (2 * self.wap)) * self.S[1, 1]  # in 2 * e

    def diagonalize_hamiltonian(self):
        """
        Numerical diagonalization of the system Hamiltonian.

        :return: None.
        """

        self.H = np.zeros((self.Nr * self.Na, self.Nr * self.Na))

        flux_q1 = np.sqrt(8) * np.pi * self.flux_zpf[1, 0]
        flux_q2 = np.sqrt(8) * np.pi * self.flux_zpf[1, 1]

        for m in range(self.Nr * self.Na):

            i = m // self.Na
            g = m % self.Na

            for n in range(0, m):
                j = n // self.Na
                h = n % self.Na

                self.H[m, n] = - self.Ej * (
                        np.cos(self.p_ext) * (
                                self.special_integrals_1.cos_integral(i, j, flux_q1) *
                                self.special_integrals_2.cos_integral(g, h, flux_q2) -
                                self.special_integrals_1.sin_integral(i, j, flux_q1) *
                                self.special_integrals_2.sin_integral(g, h, flux_q2)) +
                        np.sin(self.p_ext) * (
                                self.special_integrals_1.sin_integral(i, j, flux_q1) *
                                self.special_integrals_2.cos_integral(g, h, flux_q2) +
                                self.special_integrals_1.cos_integral(i, j, flux_q1) *
                                self.special_integrals_2.sin_integral(g, h, flux_q2)))

                self.H[n, m] = self.H[m, n]

            self.H[m, m] = self.wrp * (i + 0.5) + self.wap * (g + 0.5) - self.Ej * (
                    np.cos(self.p_ext) * (
                            self.special_integrals_1.cos_integral(i, i, flux_q1) *
                            self.special_integrals_2.cos_integral(g, g, flux_q2) -
                            self.special_integrals_1.sin_integral(i, i, flux_q1) *
                            self.special_integrals_2.sin_integral(g, g, flux_q2)) +
                    np.sin(self.p_ext) * (
                            self.special_integrals_1.sin_integral(i, i, flux_q1) *
                            self.special_integrals_2.cos_integral(g, g, flux_q2) +
                            self.special_integrals_1.cos_integral(i, i, flux_q1) *
                            self.special_integrals_2.sin_integral(g, g, flux_q2)))

        super().diagonalize_hamiltonian()

        # calculate the creator an annihilator array
        self._creator_annihilator()


