#  Copyright (c) 2024. Martin Spiecker. All rights reserved.

import numpy as np
import scipy.special as sp


def norm_hermite(n, x):

    n = int(n)

    return sp.eval_hermite(n, x) * np.exp(- x**2 / 2) / np.sqrt(2**n * sp.factorial(n, exact=True) * np.sqrt(np.pi))


class SpecialIntegrals:

    def __init__(self):

        self.look_up_table_cos = None
        self.look_up_table_sin = None
        self.look_up_table_ovl = None
        self.look_up_table_cos_shift = None
        self.look_up_table_sin_shift = None

    def init_look_up_table_fluxonium(self, n_max):

        self.look_up_table_cos = np.full((n_max + 1, n_max + 1), np.nan)
        self.look_up_table_sin = np.full((n_max + 1, n_max + 1), np.nan)

    def init_look_up_table_fluxoniumLCAO(self, n_max, max_dist):

        self.init_look_up_table_fluxonium(n_max)

        self.look_up_table_ovl = np.full((n_max + 1, n_max + 1, 2 * max_dist + 1), np.nan)
        self.look_up_table_cos_shift = np.full((n_max + 1, n_max + 1, 2 * max_dist + 1), np.nan)
        self.look_up_table_sin_shift = np.full((n_max + 1, n_max + 1, 2 * max_dist + 1), np.nan)

    def destroy_look_up_table(self):

        self.look_up_table_cos = None
        self.look_up_table_sin = None
        self.look_up_table_ovl = None
        self.look_up_table_cos_shift = None
        self.look_up_table_sin_shift = None

    def cos_integral(self, m, n, b):

        m = int(m)
        n = int(n)

        if n > m:
            min = m
            m = n
            n = min

        if self.look_up_table_cos is None or np.isnan(self.look_up_table_cos[m, n]):

            if (m + n) % 2 == 0:

                val = (-1)**((m - n) / 2) / np.sqrt(float(2**(m - n) * sp.perm(m, m - n, exact=True))) \
                       * b**(m - n) * np.exp(- b**2 / 4) * sp.eval_genlaguerre(n, m - n, b**2 / 2)
            else:
                val = 0.0

            if self.look_up_table_cos is not None:
                self.look_up_table_cos[m, n] = val
        else:
            val = self.look_up_table_cos[m, n]

        return val

    def sin_integral(self, m, n, b):

        m = int(m)
        n = int(n)

        if n > m:
            min = m
            m = n
            n = min

        if self.look_up_table_sin is None or np.isnan(self.look_up_table_sin[m, n]):

            if (m + n) % 2 == 1:

                val = (-1)**((m - n - 1) / 2) / np.sqrt(float(2**(m - n) * sp.perm(m, m - n, exact=True))) \
                       * b**(m - n) * np.exp(- b**2 / 4) * sp.eval_genlaguerre(n, m - n, b**2 / 2)
            else:
                val = 0.0

            if self.look_up_table_sin is not None:
                self.look_up_table_sin[m, n] = val
        else:
            val = self.look_up_table_sin[m, n]

        return val

    #####################################
    #####  FluxoniumLCAO integrals  #####
    #####################################

    def overlap_shift_integral(self, m, n, dist, d):

        m = int(m)
        n = int(n)

        if self.look_up_table_ovl is None or np.isnan(self.look_up_table_ovl[m, n, dist]):

            N = np.exp(- d**2 / 4) * np.sqrt((sp.factorial(n, exact=True) / 2**n) * (sp.factorial(m, exact=True) / 2**m))

            A = 0
            for k in range(np.min((n, m)) + 1):
                A += 2**k * d**(m - k) * (- d)**(n - k) / (
                        sp.factorial(m - k, exact=True) * sp.factorial(n - k, exact=True) * sp.factorial(k, exact=True))
            val = N * A

            if self.look_up_table_ovl is not None:
                self.look_up_table_ovl[m, n, dist] = val
        else:
            val = self.look_up_table_ovl[m, n, dist]

        return val

    def x_shift_integral(self, m, n, dist, d):

        A = 0

        if m > 0:
            A += np.sqrt(m / 2) * self.overlap_shift_integral(m - 1, n, dist, d)

        if n > 0:
            A += np.sqrt(n / 2) * self.overlap_shift_integral(m, n - 1, dist, d)

        return A

    def x2_shift_integral(self, m, n, dist, d):

        A = 0

        if m > 0:
            A += np.sqrt(m / 2) * self.x_shift_integral(m - 1, n, dist, d)

        if n > 0:
            A += np.sqrt(n / 2) * self.x_shift_integral(m, n - 1, dist, d)

        return A + self.overlap_shift_integral(m, n, dist, d) / 2

    def cos_shift_integral(self, m, n, dist, d, b):

        m = int(m)
        n = int(n)

        if self.look_up_table_cos_shift is None or np.isnan(self.look_up_table_cos_shift[m, n, dist]):

            N = np.exp(- d**2 / 4)

            C = 0
            for k in range(m + 1):
                for l in range(n + 1):
                    C += sp.comb(m, k, exact=True) * sp.comb(n, l, exact=True) * \
                         d**(m - k) * (- d)**(n - l) * self.cos_integral(k, l, b) / \
                         np.sqrt(2**(m + n - (k + l)) *
                                 float(sp.factorial(m, exact=True) / sp.factorial(k, exact=True)) *
                                 float(sp.factorial(n, exact=True) / sp.factorial(l, exact=True)))
            val = N * C

            if self.look_up_table_cos_shift is not None:
                self.look_up_table_cos_shift[m, n, dist] = val
        else:
            val = self.look_up_table_cos_shift[m, n, dist]

        return val

    def sin_shift_integral(self, m, n, dist, d, b):

        m = int(m)
        n = int(n)

        if self.look_up_table_sin_shift is None or np.isnan(self.look_up_table_sin_shift[m, n, dist]):

            N = np.exp(- d**2 / 4)

            S = 0
            for k in range(m + 1):
                for l in range(n + 1):
                    S += sp.comb(m, k, exact=True) * sp.comb(n, l, exact=True) * \
                         d**(m - k) * (- d)**(n - l) * self.sin_integral(k, l, b) / \
                         np.sqrt(2**(m + n - (k + l)) *
                                 float(sp.factorial(m, exact=True) / sp.factorial(k, exact=True)) *
                                 float(sp.factorial(n, exact=True) / sp.factorial(l, exact=True)))
            val = N * S

            if self.look_up_table_sin_shift is not None:
                self.look_up_table_sin_shift[m, n, dist] = val
        else:
            val = self.look_up_table_sin_shift[m, n, dist]

        return val
