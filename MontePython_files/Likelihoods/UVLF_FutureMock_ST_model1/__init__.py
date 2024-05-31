########################################################################
# Written by Nashwan Sabti - 2021
# Please cite 2110.XXXX and 2110.XXXX when using this likelihood
########################################################################

from likelihood_class import Likelihood

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simps
from scipy.special import erf


class UVLF_FutureMock_ST_model1(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Define order of Gaussian quadrature integration
        self.points, self.weights = np.polynomial.legendre.leggauss(25)

        # Import future mock UV LF data and impose 20% minimal error
        self.UVLF_FutureMock = np.loadtxt(
            data.path["data"] + "/UVLF/UVLF_FutureMock_WideField.txt"
        )  # Wide field mock data
        # self.UVLF_FutureMock = np.loadtxt(data.path["data"]+"/UVLF/UVLF_FutureMock_DeepField.txt") # Deep field mock data
        self.zs = np.unique(self.UVLF_FutureMock[:, 0])

        # Halo masses which we integrate over
        self.Mhalos = np.geomspace(1e8, 1e14, 1000)

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
    def integrator(self, f, a, b):
        sub = (b - a) / 2.0
        add = (b + a) / 2.0

        if sub == 0:
            return 0.0

        return sub * np.dot(f(sub * self.points + add), self.weights)

    # Alcock-Paczynski corrections
    def AP_effect(self, data):
        deltaz = 0.5  # 1/2 width of redshift slice
        Omega_m = data.mcmc_parameters["Omega_m"]["current"]
        hubble = data.mcmc_parameters["h"]["current"]

        self.UVLF_data = self.UVLF_FutureMock.copy()
        for z in self.zs:
            # Vratio is the correction to the UV LF
            Vratio = (
                np.power(
                    self.rcomoving(
                        z + deltaz, self.Omega_m_FutureMock, self.h_FutureMock
                    ),
                    3,
                )
                - np.power(
                    self.rcomoving(
                        z - deltaz, self.Omega_m_FutureMock, self.h_FutureMock
                    ),
                    3,
                )
            ) / (
                np.power(self.rcomoving(z + deltaz, Omega_m, hubble), 3)
                - np.power(self.rcomoving(z - deltaz, Omega_m, hubble), 3)
            )

            # Apply correction to UV LF and error
            self.UVLF_data[self.UVLF_data[:, 0] == z, 3:] *= Vratio

            # Apply correction to magntiudes, since luminosity distances are affected too
            self.UVLF_data[self.UVLF_data[:, 0] == z, 1] = self.UVLF_data[
                self.UVLF_data[:, 0] == z, 1
            ] - 5.0 * np.log10(
                self.rcomoving(z, Omega_m, hubble)
                / self.rcomoving(z, self.Omega_m_FutureMock, self.h_FutureMock)
            )

    # Sheth-Mo-Tormen HMF (using sigma_cb as massive neutrinos are considered in the runs)
    def HMF(self, cosmo, data, z, M):
        rhoM = (
            np.power(data.mcmc_parameters["h"]["current"], 2)
            * data.mcmc_parameters["Omega_m"]["current"]
            * self.rho_crit
        )
        deltaM = 100
        sigma = cosmo.sigma_cb(np.power(3.0 * M / (4.0 * np.pi * rhoM), 1.0 / 3), z)
        dsigmadM = (
            cosmo.sigma_cb(
                np.power(3.0 * (M + deltaM) / (4.0 * np.pi * rhoM), 1.0 / 3), z
            )
            - cosmo.sigma_cb(
                np.power(3.0 * (M - deltaM) / (4.0 * np.pi * rhoM), 1.0 / 3), z
            )
        ) / (2.0 * deltaM)

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
            * (rhoM / (M * sigma))
            * dsigmadM
        )

    # Obtain average <MUV> from given halo mass Mh using EPS accretion model (Eq. 6 in 1409.5228)
    def MUV_from_Mh(self, cosmo, data, z, Mh):
        alphastar = data.mcmc_parameters["alphastar"]["current"]
        betastar = data.mcmc_parameters["betastar"]["current"]
        epsilonstar = 10 ** (
            data.mcmc_parameters["epsilonstar_slope"]["current"]
            * np.log10((1 + z) / (1 + 6))
            + data.mcmc_parameters["epsilonstar_icept"]["current"]
        )
        Mc = 10 ** (
            data.mcmc_parameters["Mc_slope"]["current"] * np.log10((1 + z) / (1 + 6))
            + data.mcmc_parameters["Mc_icept"]["current"]
        )

        Q = data.mcmc_parameters["Q"]["current"]
        rhoM = (
            np.power(data.mcmc_parameters["h"]["current"], 2)
            * data.mcmc_parameters["Omega_m"]["current"]
            * self.rho_crit
        )
        sigma1 = cosmo.sigma_cb(
            np.power(3.0 * Mh / Q / (4.0 * np.pi * rhoM), 1.0 / 3), z
        )
        sigma2 = cosmo.sigma_cb(np.power(3.0 * Mh / (4.0 * np.pi * rhoM), 1.0 / 3), z)
        functionf = 1.0 / np.sqrt(sigma1**2 - sigma2**2)
        dgrowthdz = (
            -cosmo.scale_independent_growth_factor_f(z)
            * cosmo.scale_independent_growth_factor(z)
            / (1.0 + z)
        )
        Mhdot = (
            -(1 + z)
            * cosmo.Hubble(z)
            * self.invMpctoinvYear
            * 1.686
            * np.sqrt(2.0 / np.pi)
            * Mh
            * functionf
            * dgrowthdz
            / cosmo.scale_independent_growth_factor(z) ** 2
        )

        return (
            -2.5
            * np.log10(
                epsilonstar
                * Mhdot
                / ((Mh / Mc) ** alphastar + (Mh / Mc) ** betastar)
                / self.kappaUV
            )
            + 51.63
        )

    # Integrated Gaussian distribution for given average magnitude
    def first_integrand(self, MUV, width, MUV_av, sigma_MUV):
        return 0.5 * (
            erf((MUV_av - MUV + width / 2.0) / (sigma_MUV * np.sqrt(2)))
            - erf((MUV_av - MUV - width / 2.0) / (sigma_MUV * np.sqrt(2)))
        )

    # Log-likelihood
    def loglkl(self, cosmo, data):

        # Compute AP effect
        self.AP_effect(data)

        chisq = 0
        # Iterate over redshifts of data
        for z in self.zs:

            # HMFs and average MUVs only need to be computed at each redshift slice
            HMFs = np.array([self.HMF(cosmo, data, z, mass) for mass in self.Mhalos])
            MUV_avs = np.array(
                [self.MUV_from_Mh(cosmo, data, z, mass) for mass in self.Mhalos]
            )

            # Iterate over data at redshift z
            for item in self.UVLF_data[self.UVLF_data[:, 0] == z, 1:]:
                MUV, width, UVLF_data, UVLF_error = item
                second_integrand = (
                    HMFs
                    * self.first_integrand(
                        MUV,
                        width,
                        MUV_avs,
                        data.mcmc_parameters["sigma_MUV"]["current"],
                    )
                    / width
                )
                chisq += (
                    (simps(second_integrand, self.Mhalos) - UVLF_data) / UVLF_error
                ) ** 2

        # Add Gaussian prior from BBN
        chisq += (
            (
                data.mcmc_parameters["omega_b"]["current"]
                * data.mcmc_parameters["omega_b"]["scale"]
                - self.omegab_BBN
            )
            / self.omegab_BBN_error
        ) ** 2

        return -0.5 * chisq
