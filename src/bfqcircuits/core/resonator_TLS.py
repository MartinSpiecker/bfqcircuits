import numpy as np
import scipy.constants as pyc
import matplotlib.pyplot as plt

from bfqcircuits.utils.resonator_atom import ResonatorAtom


class ResonatorTLS(ResonatorAtom):

    def __init__(self):

        ResonatorAtom.__init__(self)

        self.wr = 0.0
        self.wa = 0.0
        self.g = 0.0
        self.RWA = True

        self.Na = 2

    def set_parameters(self, wr=None, wa=None, g=None, RWA=None, Nr=None):
        """
        Helper routine to set relevant system parameters.

        :param wr: resonator frequency in [GHz].
        :param wa: qubit frequency in [GHz].
        :param g: coupling in [GHz].
        :param Nr: number of resonator basis states.
        :return: None.
        """

        if wr is not None:
            self.wr = wr
        if wa is not None:
            self.wa = wa
        if g is not None:
            self.g = g
        if RWA is not None:
            self.RWA = RWA
        if Nr is not None:
            self.Nr = Nr

        self.wr_approx = self.wr

    def sweep_parameter(self, par_sweep, par_name=None):
        """
        Sweep of system parameters.

        :param par_sweep: either dict with keys "wr", "wa", "g" and corresponding
                          1D numpy parameter arrays,
                          or 1D numpy array with parameter sweep for the parameter specified by par_name.
        :param par_name: "wr", "wa", "g".
        :return: None.
        """

        if type(par_sweep) == dict:

            keys = list(par_sweep.keys())

            self.steps = par_sweep[keys[0]].size
            self._initialize_sweep(np.arange(self.steps))
            for i in range(self.steps):
                for key in keys:
                    setattr(self, key, par_sweep[key][i])
                self._calc_sweep(i)
        else:
            if par_name in ["wr", "wa", "g"]:

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

        self.steps = Nr.shape[0]
        self.par_sweep = Nr
        size = self.Na * Nr
        self.system_pars_sweep = np.zeros(self.steps, dtype=object)
        self.H_sweep = np.full((size, size, self.steps), np.nan)
        self.E_sweep = np.full((size, self.steps), np.nan)
        self.Eg_sweep = np.zeros(self.steps)
        self.v_sweep = np.full((size, size, self.steps), np.nan)

        for i in range(self.steps):
            self.Nr = Nr[i, 1]
            self._calc_sweep(i)

    def inspect_sweep(self, step):
        """
        Reset system to a specific sweep step. The routine assumes that the sweep has already been performed.

        :param: step: step of the sweep.
        :return: None.
        """

        self.wr = self.system_pars_sweep[step].wr
        self.wa = self.system_pars_sweep[step].wa
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

    def calc_resonator_dipole_moments(self, state1, state2, sorted_states):
        """
        Calculation of the resonator position and momentum dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: integer if sorted_states is False, tupel (na, nr) if sorted_states is True.
        :param state2: integer if sorted_states is False, tupel (na, nr) if sorted_states is True.
        :param sorted_states: True or False. If True the states must have been sorted already
                              with self.associated_levels().
        :return: absolute position dipole moment, absolute momentum dipole moment.
        """

        if sorted_states:
            v1 = self.v_sort[*state1, :]
            v2 = self.v_sort[*state2, :]
        else:
            v1 = self.v[:, state1]
            v2 = self.v[:, state2]

        b_daggar = np.sum(v1[self.Na:] * self.sqrts_r * v2[:-self.Na])
        b = np.sum(v1[:-self.Na] * self.sqrts_r * v2[self.Na:])

        return np.abs((b_daggar + b)), np.abs((b_daggar - b))

    def calc_atom_dipole_moments(self, state1, state2, sorted_states):
        """
        Calculation of the TLS x- and y-dipole moments between two states.
        This routine assumes that the Hamiltonian has been diagonalized using self.diagonalize_hamiltonian().

        :param state1: integer if sorted_states is False, tupel (na, nr) if sorted_states is True.
        :param state2: integer if sorted_states is False, tupel (na, nr) if sorted_states is True.
        :param sorted_states: True or False. If True the states must have been sorted already
                              with self.associated_levels().
        :return: absolute x-dipole moment, absolute y-dipole moment.
        """

        if sorted_states:
            v1 = self.v_sort[*state1, :]
            v2 = self.v_sort[*state2, :]
        else:
            v1 = self.v[:, state1]
            v2 = self.v[:, state2]

        a_daggar = np.sum(v1[1:] * self.sqrts_a * v2[:-1])
        a = np.sum(v1[:-1] * self.sqrts_a * v2[1:])

        return np.abs((a_daggar + a) / np.sqrt(2)), np.abs((a_daggar - a) / np.sqrt(2))

    def associate_levels(self, dE=0.1, na_max=-1, nr_max=-1):

        if self.RWA:

            d = self.wa - self.wr

            self.associated_levels = np.full((self.Na * self.Nr, 2), 0)

            self.associated_levels[0, :] = (0, 0)
            for i in range(1, self.Nr):

                if d <= 0:
                    self.associated_levels[self.Na * i - 1, :] = (i - 1, 1)
                    self.associated_levels[self.Na * i, :] = (i, 0)
                else:
                    self.associated_levels[self.Na * i - 1, :] = (i, 0)
                    self.associated_levels[self.Na * i, :] = (i - 1, 1)
            self.associated_levels[-1, :] = (self.Nr - 1, 1)

            self.E_sort = np.empty((self.Na, self.Nr))
            self.v_sort = np.empty((self.Na, self.Nr, self.Na * self.Nr))

            for i in range(self.Na):

                arg = np.argwhere(self.associated_levels[:, 1] == i)[:, 0]
                self.E_sort[i, :] = self.E[arg]
                self.v_sort[i, :, :] = self.v[:, arg].T

            self.E_trust = np.inf

        else:
            super().associate_levels(dE=dE, na_max=na_max, nr_max=nr_max)

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

        self.H = np.zeros((self.Nr * self.Na, self.Nr * self.Na))

        if not self.RWA:

            self.H = (np.diag(np.repeat(self.wr * (np.arange(self.Nr) + 0.5), self.Na) +
                              np.tile(self.wa * (np.arange(self.Na) - 0.5), self.Nr)) +
                      np.diag((self.g * np.tile(np.arange(self.Na), self.Nr) *
                               np.repeat(np.sqrt(np.arange(1, self.Nr + 1)), self.Na))[:-1], 1) +
                      np.diag((self.g * np.tile(np.arange(self.Na), self.Nr) *
                               np.repeat(np.sqrt(np.arange(1, self.Nr + 1)), self.Na))[:-1], -1) +
                      np.diag((self.g * np.tile(np.arange(self.Na), self.Nr) *
                               np.repeat(np.sqrt(np.arange(1, self.Nr + 1)), self.Na))[1:-1], 2) +
                      np.diag((self.g * np.tile(np.arange(self.Na), self.Nr) *
                               np.repeat(np.sqrt(np.arange(1, self.Nr + 1)), self.Na))[1:-1], -2))

            super().diagonalize_hamiltonian()
        else:
            self.H = (np.diag(np.repeat(self.wr * (np.arange(self.Nr) + 0.5), self.Na) +
                              np.tile(self.wa * (np.arange(self.Na) - 0.5), self.Nr)) +
                      np.diag((self.g * np.tile(np.arange(self.Na), self.Nr) *
                               np.repeat(np.sqrt(np.arange(1, self.Nr + 1)), self.Na))[:-1], 1) +
                      np.diag((self.g * np.tile(np.arange(self.Na), self.Nr) *
                               np.repeat(np.sqrt(np.arange(1, self.Nr + 1)), self.Na))[:-1], -1))

            d = self.wa - self.wr
            self.v = np.eye(self.Na * self.Nr)
            for i in range(1, self.Nr):

                sqrt = np.sqrt(d**2 + 4 * self.g**2 * i)

                if d <= 0:
                    s1 = - d + sqrt
                    s2 = 2 * self.g * np.sqrt(i)
                else:
                    s2 = d + sqrt
                    s1 = 2 * self.g * np.sqrt(i)

                v1 = s1 / np.sqrt(s1**2 + s2**2)
                v2 = s2 / np.sqrt(s1**2 + s2**2)

                self.v[self.Na * i - 1, self.Na * i - 1] = v1
                self.v[self.Na * i, self.Na * i - 1] = - v2
                self.v[self.Na * i - 1, self.Na * i] = v2
                self.v[self.Na * i, self.Na * i] = v1

            if d <= 0:
                sgn = -1.0
            else:
                sgn = 1.0

            Eg = (self.wr * np.arange(self.Nr)
                  - 0.5 * sgn * np.sqrt(d**2 + 4 * self.g**2 * np.arange(self.Nr)))
            Ee = (self.wr * np.arange(1, self.Nr + 1)
                  + 0.5 * sgn * np.sqrt(d**2 + 4 * self.g**2 * np.arange(1, self.Nr + 1)))

            # self.E = np.sort(np.hstack((Eg, Ee)))

            self.E = np.empty(self.Na * self.Nr)
            self.E[0] = Eg[0]
            for i in range(1, self.Nr):
                if d <= 0:
                    self.E[self.Na * i - 1:self.Na * i + 1] = (Ee[i - 1], Eg[i])
                else:
                    self.E[self.Na * i - 1:self.Na * i + 1] = (Eg[i], Ee[i - 1])
            self.E[-1] = Ee[-1]
            self.Eg = self.E[0]

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

        ax.text(0.5, 2.0, "Quantum Rabi model:", va="top", fontsize=12)
        ax.text(1.0, 1.5, r"$H=\hbar \omega_\mathrm{r} \left(a^\dagger a + \frac{1}{2}\right)"
                           r" + \frac{\hbar \omega_\mathrm{a}}{2} \sigma_z + \hbar g (a^\dagger + a)\sigma_x$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        ax.text(0.5, 0.5, "Jaynes-Cummings model:",
                va="top", fontsize=12)
        ax.text(1.0, 0.0, r"$H=\hbar \omega_\mathrm{r} \left(a^\dagger a + \frac{1}{2}\right)"
                          r" + \frac{\hbar \omega_\mathrm{a}}{2} \sigma_z + \hbar g (a^\dagger \sigma^- + a \sigma^+)$",
                va="top", fontsize=15, math_fontfamily='dejavuserif')

        #ax.axvline(0.0)
        #ax.axvline(6.0)
        #ax.axhline(-0.9)
        #ax.axhline(2.1)

        ax.set_xlim(0.0, 6.0)
        ax.set_ylim(-0.9, 2.1)

        return fig

    def __repr__(self):
        """
        Returns a representation of the resonator TLS parameters.

        :return: string
        """

        return (f"wr = {self.wr:.4e}\n"
                f"wa = {self.wa:.4e}\n"
                f"g = {self.g:.4e}\n"
                f"RWA = {self.RWA}\n"
                )

    class Parameters:

        def __init__(self, resonator_TLS):

            self.wa = resonator_TLS.wa
            self.wr = resonator_TLS.wr
            self.g = resonator_TLS.g
            self.RWA = resonator_TLS.RWA
            self.Nr = resonator_TLS.Nr


