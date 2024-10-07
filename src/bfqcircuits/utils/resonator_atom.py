#  Copyright (c) 2024. Martin Spiecker. All rights reserved.

import numpy as np
import scipy as sc

import matplotlib.colors as cl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class ResonatorAtom:

    def __init__(self):

        self.H = None
        self.E = None
        self.Eg = 0.0
        self.v = None
        self.Nr = 0
        self.Na = 0

        # state association
        self.wr_approx = 0.0
        self.E_sort = None
        self.v_sort = None
        self.associated_levels = None
        self.E_trust = 0.0

        self.resonator_transitions = None
        self.chi = None
        self.atom_spectrum = None
        self.atom_stark_shift = None

        # sweeps
        self.steps = 0
        self.par_sweep = None
        self.system_pars_sweep = None
        self.H_sweep = None
        self.E_sweep = None
        self.Eg_sweep = None
        self.v_sweep = None

        self._sweep_sorted = False
        self.E_sort_sweep = None
        self.v_sort_sweep = None
        self.associated_levels_sweep = None
        self.E_trust_sweep = None

        self.flux_dm_sweep = None
        self.charge_dm_sweep = None

        self.resonator_transitions_sweep = None
        self.atom_spectrum_sweep = None
        self.chi_sweep = None
        self.atom_stark_shift_sweep = None

        # plots
        self.colors = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])


    #############################
    #####  sweep parameters #####
    #############################

    def _initialize_sweep(self, par_sweep):
        """
        Private routine that creates the data storage for the parameter sweep.

        :param par_sweep: 1D-numpy array.
        :return: None.
        """

        self.steps = par_sweep.size
        self.par_sweep = par_sweep
        self.system_pars_sweep = np.zeros(self.steps, dtype=object)
        self.H_sweep = np.zeros((self.Nr * self.Na, self.Nr * self.Na, self.steps))
        self.E_sweep = np.zeros((self.Nr * self.Na, self.steps))
        self.Eg_sweep = np.zeros(self.steps)
        self.v_sweep = np.zeros((self.Nr * self.Na, self.Nr * self.Na, self.steps))

        self._sweep_sorted = False
        self.E_sort_sweep = None
        self.v_sort_sweep = None
        self.E_trust_sweep = None
        self.associated_levels_sweep = None

    def inspect_sweep(self, step):
        """
        Reset system to a specific sweep step. The routine assumes that the sweep has already been performed.

        :param: step: step of the sweep.
        :return: None.
        """

        self.H = self.H_sweep[:, :, step]
        self.E = np.empty(self.Na * self.Nr)
        self.E[:] = self.E_sweep[:, step]
        self.Eg = self.Eg_sweep[step]
        self.v = self.v_sweep[:, :, step]

        if self._sweep_sorted:
            self.E_sort = self.E_sort_sweep[step]
            self.v_sort = self.v_sort_sweep[step]
            self.associated_levels = self.associated_levels_sweep[step]
            self.E_trust = self.E_trust_sweep[step]

    def _calc_sweep(self, step):
        """
        Private routine that diagonalizes the Hamiltonian and stores the data of the given sweep step.

        :param step: step of the parameter sweep.
        :return: None.
        """

        self.diagonalize_hamiltonian()
        self.H_sweep[:, :, step] = self.H
        self.E_sweep[:, step] = self.E
        self.Eg_sweep[step] = self.Eg
        self.v_sweep[:, :, step] = self.v

    ############################
    #####  sweep analysis  #####
    ############################

    def substract_groundstate_energy(self):
        """
        Subtracts ground state energy from self.E.

        :return: None.
        """

        self.E -= self.Eg

    def add_groundstate_energy(self):
        """
        Adds ground state energy to self.E.

        :return: None.
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

    def calc_resonator_dipole_moments_sweep(self, state1, state2, sorted_states):
        """
        Calculation of the resonator flux and charge dipole moments between two states.

        :param state1: integer if sorted_states is False, tupel (na, nr) if sorted_states is True.
        :param state2: integer if sorted_states is False, tupel (na, nr) if sorted_states is True.
        :param sorted_states: True or False. If True the states must have been sorted already
                              with self.associated_levels_sweep().
        :return: None.
        """

        self.flux_dm_sweep = np.zeros(self.steps)
        self.charge_dm_sweep = np.zeros(self.steps)

        for i in range(self.steps):

            self.inspect_sweep(i)
            self.flux_dm_sweep[i], self.charge_dm_sweep[i] = self.calc_resonator_dipole_moments(state1, state2,
                                                                                                sorted_states)

    def calc_atom_dipole_moments_sweep(self, state1, state2, sorted_states):
        """
        Calculation of the atom flux and charge dipole moments between two states.

        :param state1: integer if sorted_states is False, tupel (na, nr) if sorted_states is True.
        :param state2: integer if sorted_states is False, tupel (na, nr) if sorted_states is True.
        :param sorted_states: True or False. If True the states must have been sorted already
                              with self.associated_levels_sweep().
        :return: None.
        """

        self.flux_dm_sweep = np.zeros(self.steps)
        self.charge_dm_sweep = np.zeros(self.steps)

        for i in range(self.steps):

            self.inspect_sweep(i)
            self.flux_dm_sweep[i], self.charge_dm_sweep[i] = self.calc_atom_dipole_moments(state1, state2,
                                                                                           sorted_states)

    def associate_levels_sweep(self, dE=0.1, na_max=-1, nr_max=-1):
        """
        Association of resonator and atom excitations to the system eigenstates for each sweep parameter.
        The states are identified by their mutual resonator charge dipole moment when climbing up the excitation ladder.
        This association is only meaningful for a weak resonator atom coupling.

        :param dE: allowed frequency variation of the resonator photons when climbing up the excitation ladder. A large
                   value for dE slows down the sorting as more resonator charge dipole moments to all the states within
                   dE have to be calculated. If no next state is found within dE, self.E_trust is set to the current
                   state energy.
        :param na_max: restricts the number of qubit states that are identified. By default, na_max is set to self.Na.
        :param nr_max: restricts the number of resonator states that are identified. This parameter will certainly
                       reduce self.E_trust and with that might limit the number of qubit states that will be identified.
                       By default, nr_max is set to self.Nr.
        :return: None
        """

        self.E_sort_sweep = np.zeros(self.steps, dtype="object")
        self.v_sort_sweep = np.zeros(self.steps, dtype="object")
        self.associated_levels_sweep = np.zeros(self.steps, dtype="object")
        self.E_trust_sweep = np.zeros(self.steps)

        for i in range(self.steps):

            self.inspect_sweep(i)
            self.associate_levels(dE=dE, na_max=na_max, nr_max=nr_max)
            self.E_sort_sweep[i] = self.E_sort
            self.v_sort_sweep[i] = self.v_sort
            self.associated_levels_sweep[i] = self.associated_levels
            self.E_trust_sweep[i] = self.E_trust

        self.E_trust = np.min(self.E_trust_sweep)
        self._sweep_sorted = True

    def derive_spectrum_properties_sweep(self):
        """
        Calculation of various spectrum properties (resonator transitions, atom spectrum, dispersive/stark shift)
        as a function of the sweep parameter.

        :return: None.
        """

        self.resonator_transitions_sweep = np.zeros(self.steps, dtype="object")
        self.atom_spectrum_sweep = np.zeros(self.steps, dtype="object")
        self.chi_sweep = np.zeros(self.steps, dtype="object")
        self.atom_stark_shift_sweep = np.zeros(self.steps, dtype="object")

        for i in range(self.steps):

            self.E_sort = self.E_sort_sweep[i]
            self.derive_spectrum_properties()

            self.resonator_transitions_sweep[i] = self.resonator_transitions
            self.chi_sweep[i] = self.chi
            self.atom_spectrum_sweep[i] = self.atom_spectrum
            self.atom_stark_shift_sweep[i] = self.atom_stark_shift

    ######################
    #####  analysis  #####
    ######################

    def diagonalize_hamiltonian(self):
        """
        Numerical diagonalization of the Hamiltonian.

        :return: None.
        """

        self.E, self.v = sc.linalg.eigh(self.H)
        self.Eg = self.E[0]

    def associate_levels(self, dE, na_max=-1, nr_max=-1):
        """
        Associating resonator and atom excitations to the system eigenstates.
        The states are identified by their mutual resonator charge dipole moment when climbing up the excitation ladder.
        This association is only meaningful for a weak resonator atom coupling.

        :param dE: allowed frequency variation of the resonator photons when climbing up the excitation ladder. A large
                   value for dE slows down the sorting as more resonator charge dipole moments to all the states within
                   dE have to be calculated. If no next state is found within dE, self.E_trust is set to the current
                   state energy.
        :param na_max: restricts the number of qubit states that are identified. By default, na_max is set to self.Na.
        :param nr_max: restricts the number of resonator states that are identified. This parameter will certainly
                       reduce self.E_trust and with that might limit the number of qubit states that will be identified.
                       By default, nr_max is set to self.Nr.
        :return: None.
        """

        if na_max == -1:
            na_max = self.Na
        if nr_max == -1:
            nr_max = self.Nr

        self.associated_levels = np.full((self.Na * self.Nr, 2), -1)
        dm = np.zeros(self.Na * self.Nr)
        used = np.zeros(self.Na * self.Nr, dtype=bool)

        self.E_trust = np.inf
        k = 0
        nq = 0
        while nq < na_max and k < self.Nr * self.Na:

            if not used[k] and self.E[k] < self.E_trust:

                e_last = np.inf
                q = k

                # start with the lowest unidentified level and
                # declare it as the next fluxonium state with zero photons in the resonator
                nr = 0
                self.associated_levels[q, 0] = nr
                self.associated_levels[q, 1] = nq
                used[q] = True

                # climb up the ladder
                terminate = False
                while nr < nr_max - 1 and not terminate:

                    dm[:] = 0

                    arg = np.argwhere(~used &
                                      ((self.E - self.E[q] > self.wr_approx - dE) &
                                       (self.E - self.E[q] < self.wr_approx + dE)))[:, 0]

                    if arg.size > 0:
                        for j in arg:
                            _, dm[j] = self.calc_resonator_dipole_moments(q, j, False)

                        # find the biggest overlap
                        q = np.argmax(dm)

                        if self.E[q] < self.E_trust:  # consider energy as correct if smaller than e_trust

                            nr += 1
                            self.associated_levels[q, 0] = nr
                            self.associated_levels[q, 1] = nq
                            used[q] = True

                            e_last = self.E[q]

                        else:
                            terminate = True

                    else:
                        if e_last < self.E_trust:
                            self.E_trust = e_last
                        terminate = True

                if nr_max < self.Nr and not terminate:  # sorting terminated by nr_max

                    if e_last < self.E_trust:
                        self.E_trust = e_last

                nq += 1

            k += 1

        self.E_sort = np.full((self.Na, self.Nr), np.nan)
        self.v_sort = np.full((self.Na, self.Nr, self.Na * self.Nr), np.nan)

        terminate = False
        i = 0
        while i < self.Na and not terminate:

            arg = np.argwhere(self.associated_levels[:, 1] == i)[:, 0]
            n_res = arg.size

            if n_res > 0:

                self.E_sort[i, :n_res] = self.E[arg]
                self.v_sort[i, :n_res, :] = self.v[:, arg].T

            else:
                terminate = True

            i += 1

    def derive_spectrum_properties(self):
        """
        Calculation of various spectrum properties (resonator transitions, atom spectrum, dispersive/stark shift).

        :return: None.
        """

        self.resonator_transitions = self.E_sort[:, 1:] - self.E_sort[:, :-1]
        # in comparison to qubit in ground state
        self.chi = self.resonator_transitions - self.resonator_transitions[0, :]

        self.atom_spectrum = self.E_sort - self.E_sort[0, :]
        # in comparison to resonator in ground state
        self.atom_stark_shift = self.atom_spectrum - self.atom_spectrum[:, 0, np.newaxis]

    def calc_resonator_dipole_moments(self, state1, state2, sorted_states) -> (float, float):
        pass

    def calc_atom_dipole_moments(self, state1, state2, sorted_states) -> (float, float):
        pass

    def plot_energy_sweep(self, ax, n):
        """
        Plot of unsorted energies as a function of the sweep parameter.

        :param ax: matplotlib axes instance.
        :param n: list/array of states that will be plotted.
        :return: None.
        """

        if not hasattr(n, '__iter__'):
            n = [n]

        for i in n:
            ax.plot(self.par_sweep, self.E_sweep[i, :], self.colors[i % 10])

        ax.set_xlabel("sweep parameter")
        ax.set_ylabel("$E$ (GHz)")

    def plot_energy_sweep_wrapped(self, ax, n_col, n_tot, E_ref):
        """
        Plot of unsorted energies modulo the reference energy as a function of the sweep parameter.

        :param ax: matplotlib axes instance.
        :param n_col: number of lower states that will be plotted with colors.
        :param n_tot: number of lower states that will be plotted.
        :param E_ref: int or 1D numpy array of size self.par_sweep with reference energies.
        :return: E_max: maximum plotted energy.
        """

        E_max = - np.inf

        for k in range(n_tot - 1, -1, -1):
            i = int(np.floor(np.min(self.E_sweep[k, :] / E_ref)))
            j = int(np.ceil(np.min(self.E_sweep[k, :] / E_ref)))

            E_max = max((max(self.E_sweep[k, :]), E_max))

            for l in range(i, j + 1):

                if k >= n_col:
                    ax.plot(self.par_sweep, self.E_sweep[k, :] - l * E_ref, color=(0.8, 0.8, 0.8))
                else:
                    ax.plot(self.par_sweep, self.E_sweep[k, :] - l * E_ref, self.colors[k % 10])

        if isinstance(E_ref, np.ndarray):
            ax.set_ylim(-0.5 * np.max(E_ref), 0.5 * np.max(E_ref))
        else:
            ax.set_ylim(-0.5 * E_ref, 0.5 * E_ref)

        ax.set_xlabel("sweep parameter")
        ax.set_ylabel(r"$E$ mod $E_\mathrm{ref}$ (GHz)")

        return E_max

    def plot_energy_convergence(self, ax, n):
        """
        Plot of energy convergence. This routine assumes that a convergence sweep has been performed beforehand.

        :param ax: matplotlib axes instance.
        :param n: number of lower states that will be plotted.
        :return: None.
        """

        converged_energy = self.E_sweep - self.E_sweep[:, -1, np.newaxis]

        for i in range(n):
            ax.plot(np.arange(self.par_sweep.shape[0]), converged_energy[i, :], self.colors[i % 10])

        ax.set_xlabel("$N$")
        ax.set_ylabel(r"$\Delta E$ (GHz)")

    def plot_sorted_energy_sweep(self, ax, na, nr):
        """
        Plot of sorted energies as a function of the sweep parameter.

        :param ax: matplotlib axes instance.
        :param nr: int or list of resonator states.
        :param na: int or list of atom states.
        :return: None.
        """

        if not hasattr(nr, '__iter__'):
            nr = [nr]
        if not hasattr(na, '__iter__'):
            na = [na]

        for i in na:
            for j in nr:
                array = np.asarray([item[i][j] for item in self.E_sort_sweep])
                ax.plot(self.par_sweep, array, self.colors[i % 10])

        ax.set_xlabel("sweep parameter")
        ax.set_ylabel(r"$E$ (GHz)")

    def plot_sorted_energy_sweep_wrapped(self, ax, E_ref, na_max=-1, nr_max=-1, n=-1, gap=True):
        """
        Plot of sorted energies modulo the reference energy as a function of the sweep parameter.

        :param ax: matplotlib axes instance.
        :param E_ref: int or 1D numpy array of size self.par_sweep with reference energies.
        :param na_max: restricts maximum atom excitation. Default is all excitations.
        :param nr_max: restricts maximum resonator excitation. Default is all excitations.
        :param n: number of lower system states that will be plotted. Default is all states.
        :param gap: lines separated (True) or connected (False) at avoided level crossings.
        :return: na_max_plotted: maximum plotted atom excitation.
                 nr_max_plotted: maximum plotted resonator excitation.
                 n_max_plotted: maximum plotted system excitation.
                 E_max: maximum plotted energy.
        """

        if n == -1:
            n = self.Na * self.Nr
        if na_max == -1:
            na_max = self.Na - 1
        if nr_max == -1:
            nr_max = self.Nr - 1

        na_max_plotted = 0
        nr_max_plotted = 0
        n_max_plotted = 0
        E_max = - np.inf

        als = np.stack(self.associated_levels_sweep)
        b = als == -1
        als[b[:, :, 0], 0] = self.Nr
        als[b[:, :, 1], 1] = self.Na

        for k in range(n):

            if np.any((als[:, k, 0] <= nr_max) & (als[:, k, 1] <= na_max)):

                n_max_plotted = max((n_max_plotted, k))

                s = int(np.floor(np.min(self.E_sweep[k, :] / E_ref)))
                t = int(np.ceil(np.min(self.E_sweep[k, :] / E_ref)))

                for l in range(s, t + 1):

                    for j in range(na_max + 1):

                        b = (als[:, k, 0] <= nr_max) & (als[:, k, 1] == j)

                        if np.sum(b) > 0:

                            na_max_plotted = max((na_max_plotted, j))
                            nr_max_plotted = max((max(als[b, k, 0]), nr_max_plotted))
                            E_max = max((max(self.E_sweep[k, b]), E_max))

                            up = (np.argwhere((b[1:] == True) & (b[:-1] == False)) + 1).tolist()
                            down = (np.argwhere((b[1:] == False) & (b[:-1] == True)) + 1).tolist()

                            if b[0]:
                                up.insert(0, [0])
                            if b[-1]:
                                down.append([b.size])
                            up = np.hstack(up)
                            down = np.hstack(down)

                            if j < 10:                    # first 10 qubit states are plotted in color
                                color = self.colors[j]
                            else:
                                color = (0.8, 0.8, 0.8)  # grays[j - 10]

                            for q in range(up.size):
                                if up[q] == 0:
                                    up[q] = 1
                                if down[q] == b.size:
                                    down[q] = b.size - 1

                                if gap:
                                    x = self.par_sweep[up[q]:down[q]]
                                    y = (self.E_sweep[k, :] - l * E_ref)[up[q]:down[q]]
                                else:
                                    x1 = self.par_sweep[up[q] - 1:down[q]]
                                    x2 = self.par_sweep[up[q]:down[q] + 1]
                                    x = 0.5 * (x2 + x1)

                                    y1 = (self.E_sweep[k, :] - l * E_ref)[up[q] - 1:down[q]]
                                    y2 = (self.E_sweep[k, :] - l * E_ref)[up[q]:down[q] + 1]
                                    y = 0.5 * (y1 + y2)

                                ax.plot(x, y, color=color, zorder=-j)

        if isinstance(E_ref, np.ndarray):
            ax.set_ylim(-0.5 * np.max(E_ref), 0.5 * np.max(E_ref))
        else:
            ax.set_ylim(-0.5 * E_ref, 0.5 * E_ref)

        ax.set_xlabel("sweep parameter")
        ax.set_ylabel(r"$E$ mod $E_\mathrm{ref}$ (GHz)")

        return na_max_plotted, nr_max_plotted, n_max_plotted, E_max

    def plot_resonator_transitions_sweep(self, ax, na, nr):
        """
        Plot resonator transition energies as a function of the sweep parameter.

        :param ax: matplotlib axes instance.
        :param nr: int or list of resonator states.
        :param na: int or list of atom states.
        :return: None.
        """

        if not hasattr(nr, '__iter__'):
            nr = [nr]
        if not hasattr(na, '__iter__'):
            na = [na]

        for i in na:
            for j in nr:
                array = np.asarray([item[i][j] for item in self.resonator_transitions_sweep])

                ax.plot(self.par_sweep, array, self.colors[j % 10])

    def plot_spectrum_sweep(self, ax, na, nr):
        """
        Plot atom energy spectrum as a function of the sweep parameter.

        :param ax: matplotlib axes instance.
        :param nr: int or list of resonator states.
        :param na: int or list of atom states.
        :return: None.
        """

        if not hasattr(nr, '__iter__'):
            nr = [nr]
        if not hasattr(na, '__iter__'):
            na = [na]

        for i in na:
            for j in nr:
                array = np.asarray([item[i][j] for item in self.atom_spectrum_sweep])

                ax.plot(self.par_sweep, array, self.colors[i % 10])

    def plot_chi_sweep(self, ax, na, nr):
        """
        Plot dispersive shift as a function of the sweep parameter.

        :param ax: matplotlib axes instance.
        :param nr: int or list of resonator states.
        :param na: int or list of atom states.
        :return: None.
        """

        if not hasattr(nr, '__iter__'):
            nr = [nr]
        if not hasattr(na, '__iter__'):
            na = [na]

        for i in na:
            for j in nr:
                array = np.asarray([item[i][j] for item in self.chi_sweep])

                ax.plot(self.par_sweep, 1e3 * array, self.colors[j % 10])

    def plot_stark_shift_sweep(self, ax, na, nr):
        """
        Plot atom stark shift as a function of the sweep parameter.

        :param ax: matplotlib axes instance.
        :param nr: int or list of resonator states.
        :param na: int or list of atom states.
        :return: None.
        """

        if not hasattr(nr, '__iter__'):
            nr = [nr]
        if not hasattr(na, '__iter__'):
            na = [na]

        for i in na:
            for j in nr:
                array = np.asarray([item[i][j] for item in self.atom_stark_shift_sweep])

                ax.plot(self.par_sweep, 1e3 * array, self.colors[j % 10])

    def plot_res_dipole_to_various_states_sweep(self, ax, ref_state, state_list, dipole="flux"):
        """
        3D plot of dipole moments as a function of the sweep parameter.

        :param ax: matplotlib axes instance with projection="3d".
        :param ref_state: state of interest.
        :param state_list: list of other states.
        :param dipole: either "flux" or "charge" for flux and charge dipole moments, respectively.
        :return: None.
        """

        dmax = 0.0
        for i in state_list:

            self.calc_resonator_dipole_moments_sweep(ref_state, i, False)

            if dipole == "flux":
                dm_sweep = self.flux_dm_sweep
            elif dipole == "charge":
                dm_sweep = self.charge_dm_sweep

            dmax = max((dmax, np.max(dm_sweep)))

            # Code to convert data in 3D polygons
            v = []
            for k in range(self.steps):
                v.append([self.par_sweep[k], self.E_sweep[i, k], np.abs(dm_sweep[k])])
            for k in range(self.steps - 1, -1, -1):
                v.append([self.par_sweep[k], self.E_sweep[i, k], 0.0])

            r, g, b = cl.to_rgb(self.colors[i % 10])
            poly3dCollection = Poly3DCollection([v], linewidths=0.0)
            poly3dCollection.set_facecolor((r, g, b, 0.5))

            # Code to plot the 3D polygons
            ax.add_collection3d(poly3dCollection)

        self.flux_dm_sweep = None
        self.charge_dm_sweep = None

        ax.set_xlim(np.min(self.par_sweep), np.max(self.par_sweep))
        ax.set_ylim(np.min(self.E_sweep[state_list, :]), np.max(self.E_sweep[state_list, :]))
        ax.set_zlim(0, dmax)

        ax.set_xlabel("sweep parameter")
        ax.set_ylabel("$E$ (GHz)")
        if dipole == "flux":
            ax.set_zlabel(r'$\langle \phi_\mathrm{r} \rangle$ (\Phi_0)')
        else:
            ax.set_zlabel(r'$\langle q_\mathrm{r} \rangle$ (2e)')
