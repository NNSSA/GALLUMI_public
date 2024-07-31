from __future__ import division
from likelihood_class import Likelihood

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from scipy.integrate import simps
from scipy.special import erf, sici
from mcfit import Hankel
import time

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


class MTNG_Clustering_2halo(Likelihood):
    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

        # Define order of Gaussian quadrature integration
        self.points, self.weights = np.polynomial.legendre.leggauss(25)

        self.MTNG_redshifts = [4, 5, 7]
        self.magnitudes = [
            [-23.0, -22.0],
            [-22.0, -21.0],
            [-21.0, -20.0],
            [-20.0, -19.0],
            [-19.0, -18.0]]
        # arcsecs_to_rad = np.pi / 3600. / 180.

        # Import Millenium-TNG galaxy clustering data and covariances
        data_MTNG_theta_z4 = (
            np.load(data.path["data"] + "/Clustering/rth_full_80_mtng.npy")[:] * 60.0
        )
        data_MTNG_omega_z4 = np.load(
            data.path["data"] + "/Clustering/wth_full_80_mtng.npy"
        )[:]
        data_MTNG_cov_z4 = np.load(
            data.path["data"] + "/Clustering/wth_cov_80_mtng.npy"
        )[:, :]

        data_MTNG_theta_z5 = (
            np.load(data.path["data"] + "/Clustering/rth_full_69_mtng.npy") * 60.0
        )[:]
        data_MTNG_omega_z5 = np.load(
            data.path["data"] + "/Clustering/wth_full_69_mtng.npy"
        )[:]
        data_MTNG_cov_z5 = np.load(
            data.path["data"] + "/Clustering/wth_cov_69_mtng.npy"
        )[:, :]

        data_MTNG_theta_z7 = (
            np.load(data.path["data"] + "/Clustering/rth_full_51_mtng.npy") * 60.0
        )[:]
        data_MTNG_omega_z7 = np.load(
            data.path["data"] + "/Clustering/wth_full_51_mtng.npy"
        )[:]
        data_MTNG_cov_z7 = np.load(
            data.path["data"] + "/Clustering/wth_cov_51_mtng.npy"
        )[:, :]

        # Cut the data in the same places as in Goldrush IV
        data_z4 = self.cut_data(
            data_MTNG_theta_z4, data_MTNG_omega_z4, data_MTNG_cov_z4
        )
        data_z5 = self.cut_data(
            data_MTNG_theta_z5, data_MTNG_omega_z5, data_MTNG_cov_z5
        )
        data_z7 = self.cut_data(
            data_MTNG_theta_z7, data_MTNG_omega_z7, data_MTNG_cov_z7
        )

        # data_z4[0] *= arcsecs_to_rad
        # data_z5[0] *= arcsecs_to_rad
        # data_z7[0] *= arcsecs_to_rad

        # data_z4[2] = np.linalg.inv(data_z4[2])
        # data_z5[2] = np.linalg.inv(data_z5[2])
        # data_z7[2] = np.linalg.inv(data_z7[2])

        self.MTNG_data = {"4": data_z4 , "5": data_z5, "7": data_z7}

        # Halo masses which we integrate over
        self.Mhalos = np.geomspace(1e6, 1e16, 600)
        self.Msubhalos = np.geomspace(1e6, 1e16, 601)

        # Number of elements in k-array and the array itself
        self.k_length = 100
        self.k_array = np.geomspace(1e-3, 200.0, self.k_length)

        # Initialise Hankel transform function from mcfit
        self.Hankel_transform = Hankel(self.k_array, lowring=True)

        return

    def cut_data(self, thetas, omegas, covs):
        pos_cuts = np.where((thetas > 100) & (thetas < 800))[0]
        # print(pos_single)
        theta_new = thetas[pos_cuts]
        # pos = np.concatenate([i * len(thetas) + pos_single for i in range(len(magnitudes))])
        omega_new = omegas[pos_cuts]
        cov_new = covs[pos_cuts, :][:, pos_cuts]
        theta_new *= np.pi / 3600.0 / 180.0  # arcsec to rad
        minError = 0.1
        sigmasq = np.array(
            list(map(max, zip((minError * omega_new) ** 2, np.diag(cov_new))))
        )
        return [theta_new, omega_new, sigmasq]

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

    def drcomovingdz(self, z, Omega_m, h):
        return (
            self.c / np.sqrt(Omega_m * np.power(1 + z, 3) + 1.0 - Omega_m) / (100.0 * h)
        )

    def integrator(self, f, a, b):
        sub = (b - a) / 2.0
        add = (b + a) / 2.0

        if sub == 0:
            return 0.0

        return sub * np.dot(f(sub * self.points + add), self.weights)

    def sigma_table(self, cosmo, data):
        rhoM = (
            np.power(data.mcmc_parameters["h"]["current"], 2)
            * data.mcmc_parameters["Omega_m"]["current"]
            * self.rho_crit
        )
        Mhalos = np.geomspace(
            min(self.Mhalos) / 3.0, max(self.Mhalos) * 1.1, 200
        )  # /3. because Mh/Q < Mh with Qmax = 2.5; *1.1 because of deltaMh
        zs = np.linspace(min(self.MTNG_redshifts), max(self.MTNG_redshifts) * 1.01, 10)
        sigmas = np.zeros((len(Mhalos), len(zs)))
        for numz, z in enumerate(zs):
            for numMh, Mh in enumerate(Mhalos):
                sigmas[numMh, numz] = cosmo.sigma(
                    np.power(3.0 * Mh / (4.0 * np.pi * rhoM), 1.0 / 3), z
                )
        return RegularGridInterpolator(
            (Mhalos, zs), sigmas, bounds_error=False, fill_value=None
        )

    def sigma_interp(self, interpolator, z, Mh):
        Mhgrid, zgrid = np.meshgrid(np.asarray(Mh), np.asarray(z), indexing="ij")
        return interpolator((Mhgrid, zgrid)).ravel()

    def dsigmadMh_interp(self, interpolator, z, Mh):
        deltaMh = 100
        sigma_plus_deltaMh = self.sigma_interp(interpolator, z, Mh + deltaMh)
        sigma_min_deltaMh = self.sigma_interp(interpolator, z, Mh - deltaMh)
        return (sigma_plus_deltaMh - sigma_min_deltaMh) / (2.0 * deltaMh)

    def halo_bias(self, interpolator, z, Mh):
        delta_c = 1.686
        nu = delta_c / self.sigma_interp(interpolator, z, Mh)
        return 1.0 + 1.0 / (np.sqrt(0.707) * delta_c) * (
            np.sqrt(0.707) * (0.707 * nu**2)
            + np.sqrt(0.707) * 0.5 * (0.707 * nu**2) ** (0.4)
            - (0.707 * nu**2) ** 0.6 / ((0.707 * nu**2) ** 0.6 + 0.14)
        )

    def HMF(self, data, interpolator, z, Mh):
        rhoM = (
            np.power(data.mcmc_parameters["h"]["current"], 2)
            * data.mcmc_parameters["Omega_m"]["current"]
            * self.rho_crit
        )
        sigma = self.sigma_interp(interpolator, z, Mh)
        dsigmadM = self.dsigmadMh_interp(interpolator, z, Mh)
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

    def SHMF(self, z, Mh, Ms):
        a = (
            6e-4
            * Mh
            * (2.0 - 0.08 * np.log10(Mh)) ** 2
            * (np.log10(1e5 * Mh)) ** (-0.1)
            * (1 + z)
        )
        b = (
            8e-5
            * Mh
            * (2.0 - 0.08 * np.log10(Mh))
            * (np.log10(1e5 * Mh)) ** (-0.08 * z)
            * (np.log10(1e8 * Mh)) ** (-1.0)
            * (np.log10(1e-18 * Mh)) ** (2.0)
        )
        Mt = 0.05 * (1.0 + z) * Mh
        alpha = 0.2 + 0.02 * z
        beta = 3.0
        return (a + b * Ms**alpha) * np.exp(-((Ms / Mt) ** beta)) / Ms**2

    def fourier_of_NFW(self, data, z, Mh, k):
        Delta_vir = 200.0
        c = data.mcmc_parameters["A_c"]["current"] * 10 ** (
            1.3081
            - 0.1078 * (1 + z)
            + 0.00398 * (1 + z) ** 2
            + (0.0223 - 0.0944 * (1 + z) ** (-0.3907)) * np.log10(Mh)
        )
        rhoC = np.power(data.mcmc_parameters["h"]["current"], 2) * self.rho_crit
        Rs = (3.0 * Mh / (4 * np.pi * Delta_vir * rhoC)) ** (1.0 / 3) / c
        rhoS = Mh / (4.0 * np.pi * Rs**3 * (np.log(1 + c) - c / (1.0 + c)))
        return (
            4
            * np.pi
            * rhoS
            * Rs**3
            / Mh
            * (
                np.sin(k * Rs) * (sici((1 + c) * k * Rs)[0] - sici(k * Rs)[0])
                - np.sin(c * k * Rs) / ((1 + c) * k * Rs)
                + np.cos(k * Rs) * (sici((1 + c) * k * Rs)[1] - sici(k * Rs)[1])
            )
        )

    def N_central(self, P_central, Mhalos, Mmin_central):
        return P_central * np.exp(-Mmin_central / Mhalos)

    def N_satellite(self, SHMFs, P_satellite, Msubhalos, Mmin_satellite):
        return simps(
            SHMFs * P_satellite * np.exp(-Mmin_satellite / Msubhalos), Msubhalos, axis=1
        )

    def ngal_of_z(self, HMFs, Mhalos, Ngal):
        return simps(HMFs * Ngal, Mhalos)

    def astro_parameters(self, data, z):
        alphastar_central = data.mcmc_parameters["alphastar_central"]["current"]
        betastar_central = data.mcmc_parameters["betastar_central"]["current"]
        epsilonstar_central = 10 ** (data.mcmc_parameters["log10epsilonstar_central_slope"]["current"] * np.log10((1 + z)/(1 + 6.)) + data.mcmc_parameters["log10epsilonstar_central_icept"]["current"]
        )
        Mc_central = 10 ** (data.mcmc_parameters["log10Mc_central_slope"]["current"] * np.log10((1 + z)/(1 + 6.)) + data.mcmc_parameters["log10Mc_central_icept"]["current"])
        sigma_MUV_central = data.mcmc_parameters["sigma_MUV_central"]["current"]
        Mmin_central = 10 ** (data.mcmc_parameters["log10Mmin_central"]["current"])

        alphastar_satellite = data.mcmc_parameters["alphastar_satellite"]["current"]
        betastar_satellite = data.mcmc_parameters["betastar_satellite"]["current"]
        epsilonstar_satellite = 10 ** (data.mcmc_parameters["log10epsilonstar_satellite_slope"]["current"] * np.log10((1 + z)/(1 + 6.)) + 
            data.mcmc_parameters["log10epsilonstar_satellite_icept"]["current"]
        )
        Mc_satellite = 10 ** (data.mcmc_parameters["log10Mc_satellite_slope"]["current"] * np.log10((1 + z)/(1 + 6.)) + 
            data.mcmc_parameters["log10Mc_satellite_icept"]["current"]
        )
        sigma_MUV_satellite = data.mcmc_parameters["sigma_MUV_satellite"]["current"]
        Mmin_satellite = 10 ** (data.mcmc_parameters["log10Mmin_satellite"]["current"])

        return (
            alphastar_central,
            betastar_central,
            epsilonstar_central,
            Mc_central,
            sigma_MUV_central,
            Mmin_central,
            alphastar_satellite,
            betastar_satellite,
            epsilonstar_satellite,
            Mc_satellite,
            sigma_MUV_satellite,
            Mmin_satellite,
        )

    def MUV_from_Mh(
        self, cosmo, data, interpolator, z, Mh, alphastar, betastar, epsilonstar, Mc
    ):
        Q = data.mcmc_parameters["Q"]["current"]
        sigma1 = self.sigma_interp(interpolator, z, Mh / Q)
        sigma2 = self.sigma_interp(interpolator, z, Mh)
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
            / cosmo.scale_independent_growth_factor(z)
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

    def P_1_halo(self, HMFs, Mhalos, N_central, N_satellite, uNFW, ngals_of_z):
        return (
            simps(
                (2.0 * N_central * N_satellite * uNFW + N_satellite**2 * uNFW**2)
                * HMFs,
                Mhalos,
                axis=1,
            )
            / ngals_of_z**2
        )

    def P_2_halo(
        self,
        cosmo,
        k_array,
        z,
        HMFs,
        biases,
        Mhalos,
        N_central,
        N_satellite,
        uNFW,
        ngals_of_z,
    ):
        pk_lin = np.vectorize(lambda x: cosmo.pk_lin(x, z))(k_array)
        return (
            pk_lin
            * simps(HMFs * (N_central + N_satellite * uNFW) * biases, Mhalos) ** 2.0
            / ngals_of_z**2
        )

    def scatter_probability(self, MUV, width, MUV_av, sigma_MUV):
        return 0.5 * (
            erf((MUV_av - MUV + width / 2.0) / (sigma_MUV * np.sqrt(2)))
            - erf((MUV_av - MUV - width / 2.0) / (sigma_MUV * np.sqrt(2)))
        )

    def omega_theta(self, theta, yy, G, rcomoving, drcomovingdz):
        PP = (
            PchipInterpolator(np.log(yy / theta), G, extrapolate=False)(
                np.log(rcomoving)
            )
            / 2.0
            / np.pi
            / drcomovingdz
        )
        return PP

    def loglkl(self, cosmo, data):
        sigma_interpolator = self.sigma_table(cosmo, data)

        Omega_m = data.mcmc_parameters["Omega_m"]["current"]
        hubble = data.mcmc_parameters["h"]["current"]

        chisq = 0.0
        for numz, z in enumerate(self.MTNG_redshifts):
            MTNG_data_at_z = self.MTNG_data["{}".format(z)]
            acf_theory = np.zeros(
                (len(self.magnitudes), len(np.unique(MTNG_data_at_z[0])))
            )

            (
                alphastar_central,
                betastar_central,
                epsilonstar_central,
                Mc_central,
                sigma_MUV_central,
                Mmin_central,
                alphastar_satellite,
                betastar_satellite,
                epsilonstar_satellite,
                Mc_satellite,
                sigma_MUV_satellite,
                Mmin_satellite,
            ) = self.astro_parameters(data, z)

            rcomoving = self.rcomoving(z, Omega_m, hubble)
            drcomovingdz = self.drcomovingdz(z, Omega_m, hubble)

            HMFs = self.HMF(data, sigma_interpolator, z, self.Mhalos)
            SHMFs = self.SHMF(z, self.Mhalos[:, np.newaxis], self.Msubhalos)

            MUV_avs_central = self.MUV_from_Mh(
                cosmo,
                data,
                sigma_interpolator,
                z,
                self.Mhalos,
                alphastar_central,
                betastar_central,
                epsilonstar_central,
                Mc_central,
            )
            MUV_avs_satellite = self.MUV_from_Mh(
                cosmo,
                data,
                sigma_interpolator,
                z,
                self.Msubhalos,
                alphastar_satellite,
                betastar_satellite,
                epsilonstar_satellite,
                Mc_satellite,
            )

            halo_biases = self.halo_bias(sigma_interpolator, z, self.Mhalos)

            for num_mags, mags in enumerate(self.magnitudes):
                MUV_min, MUV_max = mags

                P_central = self.scatter_probability(
                    0.5 * (MUV_min + MUV_max),
                    MUV_max - MUV_min,
                    MUV_avs_central,
                    sigma_MUV_central,
                )
                P_satellite = self.scatter_probability(
                    0.5 * (MUV_min + MUV_max),
                    MUV_max - MUV_min,
                    MUV_avs_satellite,
                    sigma_MUV_satellite,
                )

                N_c = self.N_central(P_central, self.Mhalos, Mmin_central)
                N_s = self.N_satellite(
                    SHMFs, P_satellite, self.Msubhalos, Mmin_satellite
                )

                Ngals = N_c + N_s
                ngals_of_z = self.ngal_of_z(HMFs, self.Mhalos, Ngals)

                uNFW = self.fourier_of_NFW(
                    data, z, self.Mhalos, self.k_array[:, np.newaxis]
                )
                Pgal = self.P_2_halo(
                    cosmo,
                    self.k_array,
                    z,
                    HMFs,
                    halo_biases,
                    self.Mhalos,
                    N_c,
                    N_s,
                    uNFW,
                    ngals_of_z,
                )
                yy, G = self.Hankel_transform(Pgal, extrap=False)
                thetasdata, acf_data, errorsq = MTNG_data_at_z
                acf_theory[num_mags, :] = np.array(
                    [
                        self.omega_theta(theta, yy, G, rcomoving, drcomovingdz)
                        for theta in np.unique(thetasdata)
                    ]
                )

                # thetas = np.geomspace(1., 1000., 50) * np.pi / 3600. / 180.
                # acf_theory = [self.omega_theta(theta, yy, G, rcomoving, drcomovingdz) for theta in thetas]

                # plt.subplot(3, 5, (num_mags + numz * 5 + 1))
                # plt.loglog(thetas * 180 * 3600 / np.pi, acf_theory, color="blue", ls="dashed")
                # if z == 7:
                #     plt.xlabel(r"$\theta$ [arcsec]")
                # if num_mags == 0:
                #    plt.ylabel(r"$\omega(\theta)$")

            chisq += np.sum((acf_theory.ravel() - acf_data) ** 2 / errorsq)
            # chisq += np.linalg.multi_dot([(acf_theory.ravel() - acf_data), inv_cov, (acf_theory.acf_theory() - acf_data)])

        chisq += (
            (
                data.mcmc_parameters["omega_b"]["current"]
                * data.mcmc_parameters["omega_b"]["scale"]
                - self.omegab_BBN
            )
            / self.omegab_BBN_error
        ) ** 2
        # plt.show()

        return -0.5 * chisq

