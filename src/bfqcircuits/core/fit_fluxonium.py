#  Copyright (c) 2024. Martin Spiecker. All rights reserved.

import numpy as np
import scipy.constants as pyc
from scipy import optimize

from bfqcircuits.core.fluxonium import Fluxonium


class FitFluxonium(Fluxonium):

    def __init__(self):

        Fluxonium.__init__(self)

        self.p_ext_data = None
        self.spectrum_data = None
        self.level_index_data = None
        self.data_points = 0

        self.E_fit = None
        self.cost = None

    def set_initial_value(self, data, L, C, Ej, N):
        """
        Initialization of fit routine.

        :param data: numpy array of shape (n, 3) -> [[external flux in (rad), energy in (GHz), state], ...]
        :param L: initial guess inductance in [H]
        :param C: initial guess capacitance in [F]
        :param Ej: initial guess Josephson energy in [GHz]
        :param N: number of basis states
        :return: None
        """

        self.p_ext_data = data[:, 0]
        self.spectrum_data = data[:, 1]
        self.level_index_data = data[:, 2].astype(int)

        self.data_points = data.shape[0]
        self.E_fit = np.zeros(self.data_points)

        self.set_parameters(L=L, C=C, Ej=Ej, N=N)
        self.calc_hamiltonian_parameters()

        error = self._errorfunc(np.array([self.w, self.Ej, self.flux_zpf]))
        self.cost = 0.5 * np.sum(error ** 2)

        print("Cost: {}".format(self.cost))

    def fit(self):
        """
        Run the fit routine. The fit is performed on the fluxonium parameters w, Ej, and flux_zpf.
        The initial guess with respect to this parameters must be correct within a factor of 4.

        :return: None
        """

        lb = np.array([0.25 * self.w, 0.25 * self.Ej, 0.25 * self.flux_zpf])
        ub = np.array([4.0 * self.w, 4.0 * self.Ej, 4.0 * self.flux_zpf])
        bounds = (lb, ub)

        fit_result = optimize.least_squares(self._errorfunc, np.array([self.w, self.Ej, self.flux_zpf]),
                                            method='trf', bounds=bounds, xtol=1e-6, verbose=2)

        self.w = fit_result['x'][0]
        self.Ej = fit_result['x'][1]
        self.flux_zpf = fit_result['x'][2]

        self.L = pyc.h * self.flux_zpf ** 2 / (2e9 * pyc.e ** 2 * self.w)
        self.C = pyc.e**2 / (2e9 * np.pi**2 * pyc.h * self.w * self.flux_zpf**2)
        self.calc_hamiltonian_parameters()  # calc all other parameters

        print("Cost: {}".format(fit_result['cost']))
        print("")
        print("L: {}nH".format(self.L * 1e9))
        print("C: {}fF".format(self.C * 1e15))
        print("Ej: {}GHz".format(self.Ej))

    def _calc_energies(self, par):
        """
        Private routine called by errorfunc

        :param par: fit parameters
        :return: None
        """

        self.w = par[0]
        self.Ej = par[1]
        self.flux_zpf = par[2]
        self.sweep_external_flux(self.p_ext_data)

        for i in range(self.steps):
            self.E_fit[i] = self.E_sweep[self.level_index_data[i], i] - self.Eg_sweep[i]

    def _errorfunc(self, par):
        """
        Private routine for calculation of the errors.

        :param par: fit parameters
        :return: 1D numpy array with errors
        """

        self._calc_energies(par)

        return self.E_fit - self.spectrum_data

    def plot_fit(self, ax, n=0, pext_sweep=None):
        """
        Plot of the measured and calculated fluxonium spectrum.

        :param ax: matplotlib axes instance
        :param n: optional number of lower states that will be plotted, if not given inferred from the measured data
        :param pext_sweep: optional 1D-numpy array with the external flux bias in [rad]
        :return: None
        """

        if n == 0:
            n = max(self.level_index_data)

        if pext_sweep is None:
            pext_sweep = np.linspace(- 0.1, 1.1, 121) * np.pi

        self.sweep_external_flux(pext_sweep)

        for i in range(n):
            ax.plot(pext_sweep, self.E_sweep[i, :] - self.Eg_sweep, self.colors[i % 10])

            b = self.level_index_data == i
            ax.plot(self.p_ext_data[b], self.spectrum_data[b], color=self.colors[i % 10], ls="", marker='+')

        ax.set_xlabel(r"$\varphi_\mathrm{ext}$ (rad)")
        ax.set_ylabel(r"$E$ (GHz)")
