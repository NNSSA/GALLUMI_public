from likelihood_class import Likelihood

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simps
import matplotlib

matplotlib.use("TkAgg")


class Galaxies_Poisson_Mstar(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Load data
        self.JWST_Mstar = np.loadtxt(data.path["data"] + "/Labbe_Mstar/Labbe_Mstar.txt")

        # Define order of Gaussian quadrature integration
        self.points, self.weights = np.polynomial.legendre.leggauss(25)
        self.points_highres, self.weights_highres = np.polynomial.legendre.leggauss(150)

        # Halo masses which we integrate over
        self.Mhalos = np.geomspace(1e9, 1e14, 600)

        return

    # Comoving radial distance
    def rcomoving(self, z, Omega_m, h):
        return (
            self.c
            * self.integrator(
                lambda x: 1 / np.sqrt(Omega_m * np.power(1 + x, 3) + 1.0 - Omega_m),
                0.0,
                z,
            )
            / (100.0 * h)
        )

    # Gaussian integrator
    def integrator(self, f, a, b, highres=False):
        sub = (b - a) / 2.0
        add = (b + a) / 2.0
        if sub == 0:
            return 0.0
        if not highres:
            return sub * np.dot(f(sub * self.points + add), self.weights)
        return sub * np.dot(f(sub * self.points_highres + add), self.weights_highres)

    # The Gaussian bump added to the power spectrum
    def Gaussian_bump(self, k, amp, mean, sigma):
        k1 = 0.5
        if k < k1:
            return 0.0
        return amp * np.exp(-((np.log(k) - mean) ** 2) / (2.0 * sigma**2))
        # return (amp / (k * sigma * np.sqrt(2. * np.pi))) * np.exp(-(np.log(k) - mean)**2 / (2. * sigma**2))

    # Calculate the mass variance manually intead from CLASS
    def calculate_sigma(self, R, kmin, kmax, pk, z, amp, mean, sigma):
        mps = np.vectorize(
            lambda y: pk(y, z) * (1.0 + self.Gaussian_bump(y, amp, mean, sigma))
        )
        # window = lambda x: np.heaviside(1. - x * R, 1.)
        window = (
            lambda x: 3.0 * (np.sin(x * R) - (x * R) * np.cos(x * R)) / (x * R) ** 3
        )
        return np.sqrt(
            self.integrator(
                lambda lnx: window(np.exp(lnx)) ** 2
                * mps(np.exp(lnx))
                * np.exp(lnx) ** 3,
                np.log(kmin),
                np.log(kmax),
                True,
            )
            / 2.0
            / np.pi**2
        )
        # return np.sqrt(self.integrator(lambda x: window(x)**2 * mps(x) * x**2, kmin, kmax, True) / 2. / np.pi**2)

    # Compute the HMF and average <MUV> given halo mass Mh
    def HMF(self, cosmo, data, z, Mh, ks):
        rhoM = (
            np.power(data.mcmc_parameters["h"]["current"], 2)
            * data.mcmc_parameters["Omega_m"]["current"]
            * self.rho_crit
        )
        deltaM = 100
        Gaussian_amp = data.mcmc_parameters["Gaussian_amp"]["current"]
        Gaussian_mean = data.mcmc_parameters["Gaussian_mean"]["current"]
        Gaussian_sigma = data.mcmc_parameters["Gaussian_sigma"]["current"]

        sigma = self.calculate_sigma(
            np.power(3.0 * Mh / (4.0 * np.pi * rhoM), 1.0 / 3),
            min(ks),
            max(ks),
            cosmo.pk_cb_lin,
            z,
            Gaussian_amp,
            Gaussian_mean,
            Gaussian_sigma,
        )
        sigma_plus_deltaM = self.calculate_sigma(
            np.power(3.0 * (Mh + deltaM) / (4.0 * np.pi * rhoM), 1.0 / 3),
            min(ks),
            max(ks),
            cosmo.pk_cb_lin,
            z,
            Gaussian_amp,
            Gaussian_mean,
            Gaussian_sigma,
        )
        sigma_minus_deltaM = self.calculate_sigma(
            np.power(3.0 * (Mh - deltaM) / (4.0 * np.pi * rhoM), 1.0 / 3),
            min(ks),
            max(ks),
            cosmo.pk_cb_lin,
            z,
            Gaussian_amp,
            Gaussian_mean,
            Gaussian_sigma,
        )
        dsigmadM = (sigma_plus_deltaM - sigma_minus_deltaM) / (2.0 * deltaM)

        return (
            -self.AST
            * np.sqrt(2.0 * self.aST / np.pi)
            * (
                1.0
                + np.power(
                    np.power(sigma, 2) / (self.aST * np.power(self.deltaST, 2)),
                    self.pST,
                )
            )
            * (self.deltaST / sigma)
            * np.exp(-self.aST * np.power(self.deltaST, 2) / (2.0 * np.power(sigma, 2)))
            * (rhoM / (Mh * sigma))
            * dsigmadM
        )

    # Log-likelihood
    def loglkl(self, cosmo, data):

        log_lkl_poisson = 0

        # Iterate over redshifts of data
        for z, log10Mstar, Vol in self.JWST_Mstar:

            # Array of wavenumbers
            ks = (
                cosmo.get_transfer(z)["k (h/Mpc)"]
                * data.mcmc_parameters["h"]["current"]
            )

            # HMFs and average MUVs only need to be computed at each redshift slice
            HMFs = np.array(
                [self.HMF(cosmo, data, z, mass, ks) for mass in self.Mhalos]
            )
            HMF_interp = PchipInterpolator(self.Mhalos, HMFs)

            M_lower = 10 ** (log10Mstar) / 0.157 / 0.1  # epsilon = 0.1
            Mh_array_integrate = np.geomspace(M_lower, 1e14, 500)
            N_gal = simps(HMF_interp(Mh_array_integrate), Mh_array_integrate) * Vol
            log_lkl_poisson += -N_gal + 1.0 * np.log(N_gal) - np.log(1.0)

        return log_lkl_poisson
