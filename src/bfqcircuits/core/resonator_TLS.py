import numpy as np
import scipy.constants as pyc
import matplotlib.pyplot as plt

from bfqcircuits.utils.resonator_atom import ResonatorAtom


class ResonatorTLS(ResonatorAtom):

    def __init__(self):

        ResonatorAtom.__init__(self)

        self.wr = 0.0
        self.wa = 0.0
        self.wa_x = 0.0
        self.wa_y = 0.0
        self.wa_z = 0.0
        self.g = 0.0
        self.RWA = False

        self.Na = 2
        self.Nr = 1

    def set_parameters(self, wr=None, wa_x=None, wa_y=None, wa_z=None, g=None, RWA=None, Nr=None):
        """
        Helper routine to set relevant system parameters.

        :param wr: resonator frequency in [GHz].
        :param wa_x: qubit energy in x-direction in [GHz].
        :param wa_y: qubit energy in y-direction in [GHz].
        :param wa_z: qubit energy in z-direction in [GHz].
        :param g: transverse (sigma_x * x) coupling strength in [GHz].
        :param RWA: boolean toggling the rotating wave approximation.
        :param Nr: number of resonator basis states.
        :return: None.
        """

        if wr is not None:
            self.wr = wr
        if wa_x is not None:
            self.wa_x = wa_x
        if wa_y is not None:
            self.wa_y = wa_y
        if wa_z is not None:
            self.wa_z = wa_z
        if g is not None:
            self.g = g
        if RWA is not None:
            self.RWA = RWA
        if Nr is not None:
            self.Nr = Nr

        self.wa = np.sqrt(self.wa_x**2 + self.wa_y**2 + self.wa_z**2)
        self.wr_approx = self.wr

    def sweep_parameter(self, par_sweep, par_name=None):
        """
        Sweep of system parameters.

        :param par_sweep: either dict with keys "wr", "wa_x", "wa_y", "wa_z", "g" and corresponding
                          1D numpy parameter arrays,
                          or 1D numpy array with parameter sweep for the parameter specified by par_name.
        :param par_name: "wr", "wa_x", "wa_y", "wa_z", "g".
        :return: None.
        """

        if type(par_sweep) == dict:

            keys = list(par_sweep.keys())

            if "wa_y" in keys or self.wa_y != 0.0:
                self._dtype = complex
            else:
                self._dtype = float

            self.steps = par_sweep[keys[0]].size
            self._initialize_sweep(np.arange(self.steps))
            for i in range(self.steps):
                for key in keys:
                    setattr(self, key, par_sweep[key][i])
                self._calc_sweep(i)
        else:
            if par_name in ["wr", "wa_x", "wa_y", "wa_z", "g"]:

                if par_name == "wa_y" or self.wa_y != 0.0:
                    self._dtype = complex
                else:
                    self._dtype = float
                self._initialize_sweep(par_sweep)
                for i in range(self.steps):
                    setattr(self, par_name, self.par_sweep[i])
                    self._calc_sweep(i)
            else:
                print("Parameter does not exist or should not be swept in this routine. \n" +
                      "For instance, for sweeping the number of basis functions use the convergence_sweep routine.")

    def convergence_sweep(self, Nr):
        """
        Convergences of the wave functions and corresponding energies with the number of basis states.

        :param Nr: number of resonator basis states.
        :return: None.
        """

        if self.wa_y == 0.0:
            self._dtype = float
        else:
            self._dtype = complex

        self.steps = Nr
        self.par_sweep = np.arange(1, Nr + 1)
        self.system_pars_sweep = np.zeros(self.steps, dtype=object)
        size = self.Na * Nr
        self.H_sweep = np.full((size, size, self.steps), np.nan, dtype=self._dtype)
        self.E_sweep = np.full((size, self.steps), np.nan)
        self.Eg_sweep = np.zeros(self.steps)
        self.v_sweep = np.full((size, size, self.steps), np.nan, dtype=self._dtype)

        for i in range(self.steps):
            self.Nr = self.par_sweep[i]
            self._calc_sweep(i)

    def inspect_sweep(self, step):
        """
        Reset system to a specific sweep step. The routine assumes that the sweep has already been performed.

        :param: step: step of the sweep.
        :return: None.
        """

        self.wr = self.system_pars_sweep[step].wr
        self.wa = self.system_pars_sweep[step].wa
        self.wa_x = self.system_pars_sweep[step].wa_x
        self.wa_y = self.system_pars_sweep[step].wa_y
        self.wa_z = self.system_pars_sweep[step].wa_z
        self.g = self.system_pars_sweep[step].g
        self.RWA = self.system_pars_sweep[step].RWA
        self.Nr = self.system_pars_sweep[step].Nr

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

    def calc_resonator_dipole_moments(self, state1, state2):
        """
        Calculation of the resonator position and momentum dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: integer or tupel (na, nr), which requires that the states have been sorted.
        :param state2: integer or tupel (na, nr), which requires that the states have been sorted.
        :return: absolute position dipole moment, absolute momentum dipole moment.
        """

        if hasattr(state1, '__iter__'):
            v1 = self.v_sort[*state1, :]
        else:
            v1 = self.v[:, state1]
        if hasattr(state2, '__iter__'):
            v2 = self.v_sort[*state2, :]
        else:
            v2 = self.v[:, state2]

        if self._dtype == float:
            b_daggar = np.sum(v1[self.Na:] * self.sqrts_r * v2[:-self.Na])
            b = np.sum(v1[:-self.Na] * self.sqrts_r * v2[self.Na:])
        else:
            b_daggar = np.sum(np.conjugate(v1[self.Na:]) * self.sqrts_r * v2[:-self.Na])
            b = np.sum(np.conjugate(v1[:-self.Na]) * self.sqrts_r * v2[self.Na:])

        return np.abs((b_daggar + b)), np.abs((b_daggar - b))

    def calc_atom_dipole_moments(self, state1, state2):
        """
        Calculation of the TLS x- and y-dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: integer or tupel (na, nr), which requires that the states have been sorted.
        :param state2: integer or tupel (na, nr), which requires that the states have been sorted.
        :return: absolute x-dipole moment, absolute y-dipole moment.
        """

        if hasattr(state1, '__iter__'):
            v1 = self.v_sort[*state1, :]
        else:
            v1 = self.v[:, state1]
        if hasattr(state2, '__iter__'):
            v2 = self.v_sort[*state2, :]
        else:
            v2 = self.v[:, state2]

        if self._dtype == float:
            a_daggar = np.sum(v1[1:] * self.sqrts_a * v2[:-1])
            a = np.sum(v1[:-1] * self.sqrts_a * v2[1:])
        else:
            a_daggar = np.sum(np.conjugate(v1[1:]) * self.sqrts_a * v2[:-1])
            a = np.sum(np.conjugate(v1[:-1]) * self.sqrts_a * v2[1:])

        return np.abs((a_daggar + a) / np.sqrt(2)), np.abs((a_daggar - a) / np.sqrt(2))

    ###################
    #####  plots  #####
    ###################

    def plot_res_dipole_to_various_states_sweep(self, ax, ref_state, state_list, dipole="x"):
        """
        3D plot of dipole moments as a function of the sweep parameter.

        :param ax: matplotlib axes instance with projection="3d".
        :param ref_state: state of interest.
        :param state_list: list of other states.
        :param dipole: either "x" or "y" for flux and charge dipole moments, respectively.
        :return: None.
        """

        if dipole == "x":
            super().plot_res_dipole_to_various_states_sweep(ax, ref_state, state_list, dipole="flux")
        elif dipole == "y":
            super().plot_res_dipole_to_various_states_sweep(ax, ref_state, state_list, dipole="charge")

        ax.set_xlabel("sweep parameter")
        ax.set_ylabel("$E$ (GHz)")
        if dipole == "x":
            ax.set_zlabel(r'$\langle x \rangle$')
        elif dipole == "y":
            ax.set_zlabel(r'$\langle y \rangle$')

    ##################
    #####  core  #####
    ##################

    def diagonalize_hamiltonian(self):
        """
        Numerical and analytical diagonalization of the system Hamiltonian.

        :return: None.
        """

        if self.wa_y == 0.0:
            self.H = np.zeros((self.Nr * self.Na, self.Nr * self.Na), dtype=float)
        else:
            self.H = np.zeros((self.Nr * self.Na, self.Nr * self.Na), dtype=complex)

        self.H += np.diag(np.repeat(self.wr * (np.arange(self.Nr) + 0.5), self.Na) +
                          np.tile(self.wa_z * (np.arange(self.Na) - 0.5), self.Nr))

        self.H += (np.diag(np.tile(0.5 * self.wa_x * np.arange(self.Na), self.Nr)[1::], 1) +
                   np.diag(np.tile(0.5 * self.wa_x * np.arange(self.Na), self.Nr)[1::], -1))

        if self.wa_y != 0.0:
            self.H += (np.diag(np.tile(-0.5j * self.wa_y * np.arange(self.Na), self.Nr)[1::], 1) +
                       np.diag(np.tile(0.5j * self.wa_y * np.arange(self.Na), self.Nr)[1::], -1))

        if self.RWA:
            self.H += (np.diag((self.g * np.tile(np.arange(self.Na), self.Nr) *
                                np.repeat(np.sqrt(np.arange(1, self.Nr + 1)), self.Na))[:-1], 1) +
                       np.diag((self.g * np.tile(np.arange(self.Na), self.Nr) *
                                np.repeat(np.sqrt(np.arange(1, self.Nr + 1)), self.Na))[:-1], -1))
        else:
            self.H += (np.diag((self.g * np.tile(np.arange(self.Na), self.Nr) *
                                np.repeat(np.sqrt(np.arange(1, self.Nr + 1)), self.Na))[:-1], 1) +
                       np.diag((self.g * np.tile(np.arange(self.Na), self.Nr) *
                                np.repeat(np.sqrt(np.arange(1, self.Nr + 1)), self.Na))[:-1], -1) +
                       np.diag((self.g * np.tile(np.arange(self.Na), self.Nr) *
                                np.repeat(np.sqrt(np.arange(1, self.Nr + 1)), self.Na))[1:-2], 3) +
                       np.diag((self.g * np.tile(np.arange(self.Na), self.Nr) *
                                np.repeat(np.sqrt(np.arange(1, self.Nr + 1)), self.Na))[1:-2], -3))

        super().diagonalize_hamiltonian()

        self._creator_annihilator()

    def _creator_annihilator(self):
        """
        Private routine that creates the creation and annihilations operators.

        :return: None
        """

        self.sqrts_r = np.zeros((self.Nr - 1) * self.Na)
        for i in range((self.Nr - 1) * self.Na):
            self.sqrts_r[i] = np.sqrt(i // self.Na + 1)

        self.sqrts_a = np.tile((1.0, 0.0), self.Nr)

    def show_formulas(self):
        """
        Creates a figure showing the most relevant formulas.

        :return: matplotlib figure instance
        """

        figwidth = 8.0
        fig, ax = plt.subplots(1, 1, figsize=(figwidth, 3.0 / 8.0 * figwidth))
        ax.axis("off")

        ax.text(0.5, 2.0, "Assymetric quantum Rabi model:", va="top", fontsize=12)
        ax.text(1.0, 1.5, r"$H=\hbar \omega_\mathrm{r} \left(a^\dagger a + \frac{1}{2}\right)"
                          r" + \frac{\hbar \omega_\mathrm{a}^\mathrm{x}}{2} \sigma_x"
                          r" + \frac{\hbar \omega_\mathrm{a}^\mathrm{y}}{2} \sigma_y"
                          r" + \frac{\hbar \omega_\mathrm{a}^\mathrm{z}}{2} \sigma_z$" + "\n" +
                          r"$\qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad"
                          r"+ \hbar g (a^\dagger + a)\sigma_x$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        ax.text(0.5, 0.5, "Rotating wave approximation:",
                va="top", fontsize=12)
        ax.text(1.0, 0.0, r"$H=\hbar \omega_\mathrm{r} \left(a^\dagger a + \frac{1}{2}\right)"
                          r" + \frac{\hbar \omega_\mathrm{a}^\mathrm{x}}{2} \sigma_x"
                          r" + \frac{\hbar \omega_\mathrm{a}^\mathrm{y}}{2} \sigma_y"
                          r" + \frac{\hbar \omega_\mathrm{a}^\mathrm{z}}{2} \sigma_z$" + "\n" +
                          r"$\qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad"
                          r" + \hbar g (a^\dagger \sigma^- + a \sigma^+)$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        #ax.axvline(0.0)
        #ax.axvline(8.0)
        #ax.axhline(-0.9)
        #ax.axhline(2.1)

        ax.set_xlim(0.0, 8.0)
        ax.set_ylim(-0.9, 2.1)

        return fig

    def __repr__(self):
        """
        Returns a representation of the resonator TLS parameters.

        :return: string
        """

        return (f"ℏωᵣ = {self.wr:.4e}\n"
                f"ℏωₐ = {self.wa:.4e}\n"
                f"ℏωₐ_x = {self.wa_x:.4e}\n"
                f"ℏωₐ_y = {self.wa_y:.4e}\n"
                f"ℏωₐ_z = {self.wa_z:.4e}\n"
                f"ℏg = {self.g:.4e}\n"
                f"RWA = {self.RWA}\n"
                )

    class Parameters:

        def __init__(self, resonator_TLS):

            self.wr = resonator_TLS.wr
            self.wa = resonator_TLS.wa
            self.wa_x = resonator_TLS.wa_x
            self.wa_y = resonator_TLS.wa_y
            self.wa_z = resonator_TLS.wa_z
            self.g = resonator_TLS.g
            self.RWA = resonator_TLS.RWA
            self.Nr = resonator_TLS.Nr


