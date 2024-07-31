from __future__ import division
from likelihood_class import Likelihood

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from scipy.integrate import simps
from scipy.special import erf, sici
from mcfit import Hankel
import time


class MTNG_Clustering(Likelihood):
    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

        # Define order of Gaussian quadrature integration
        self.points, self.weights = np.polynomial.legendre.leggauss(25)

        self.MTNG_redshifts = [4]  # , 5, 7]
        self.magnitudes = [
            [-23.0, -22.0],
            # [-22.0, -21.0],
            # [-21.0, -20.0],
            # [-20.0, -19.0],
            # [-19.0, -18.0],
        ]
        # arcsecs_to_rad = np.pi / 3600. / 180.

        # Import Millenium-TNG galaxy clustering data and covariances
        data_MTNG_theta_z4 = (
            np.load(data.path["data"] + "/Clustering/rth_full_80_mtng.npy")[:18] * 60.0
        )
        data_MTNG_omega_z4 = np.load(
            data.path["data"] + "/Clustering/wth_full_80_mtng.npy"
        )[:18]
        data_MTNG_cov_z4 = np.load(
            data.path["data"] + "/Clustering/wth_cov_80_mtng.npy"
        )[:18, :18]

        data_MTNG_theta_z5 = (
            np.load(data.path["data"] + "/Clustering/rth_full_69_mtng.npy")[:18] * 60.0
        )
        data_MTNG_omega_z5 = np.load(
            data.path["data"] + "/Clustering/wth_full_69_mtng.npy"
        )[:18]
        data_MTNG_cov_z5 = np.load(
            data.path["data"] + "/Clustering/wth_cov_69_mtng.npy"
        )[:18, :18]

        data_MTNG_theta_z7 = (
            np.load(data.path["data"] + "/Clustering/rth_full_51_mtng.npy") * 60.0
        )[:18]
        data_MTNG_omega_z7 = np.load(
            data.path["data"] + "/Clustering/wth_full_51_mtng.npy"
        )[:18]
        data_MTNG_cov_z7 = np.load(
            data.path["data"] + "/Clustering/wth_cov_51_mtng.npy"
        )[:18, :18]

        # Cut the data in the same places as in Goldrush IV
        data_z4 = self.cut_data(
            data_MTNG_theta_z4, data_MTNG_omega_z4, data_MTNG_cov_z4, 4
        )
        # data_z5 = self.cut_data(
        #    data_MTNG_theta_z5, data_MTNG_omega_z5, data_MTNG_cov_z5, 5
        # )
        # data_z7 = self.cut_data(
        #    data_MTNG_theta_z7, data_MTNG_omega_z7, data_MTNG_cov_z7, 7
        # )

        # data_z4[0] *= arcsecs_to_rad
        # data_z5[0] *= arcsecs_to_rad
        # data_z7[0] *= arcsecs_to_rad

        # data_z4[2] = np.linalg.inv(data_z4[2])
        # data_z5[2] = np.linalg.inv(data_z5[2])
        # data_z7[2] = np.linalg.inv(data_z7[2])

        self.MTNG_data = {"4": data_z4}  # , "5": data_z5, "7": data_z7}

        # Halo masses which we integrate over
        self.Mhalos = np.geomspace(1e6, 1e16, 600)
        self.Msubhalos = np.geomspace(1e6, 1e16, 601)

        # Number of elements in k-array and the array itself
        self.k_length = 100
        self.k_array = np.geomspace(1e-3, 200.0, self.k_length)

        # Initialise Hankel transform function from mcfit
        self.Hankel_transform = Hankel(self.k_array, lowring=True)

        return

    def cut_data(self, thetas, omegas, covs, z):
        # Filter thetas based on unique values and specific conditions
        thetas_unique = np.unique(thetas)
        pos_single = np.where((thetas_unique <= 10) | (thetas_unique > 100))[0]

        # Calculate segment length for position cuts
        segment_length = len(thetas) // len(self.magnitudes)

        # Apply initial data cuts based on magnitudes and conditions
        theta_new, omega_new, cov_new = [], [], []
        for i in range(len(self.magnitudes)):
            pos = i * segment_length + pos_single
            if len(pos) > len(thetas):  # Ensure pos does not exceed thetas' length
                pos = pos[: len(thetas)]
            theta_new.append(thetas[pos])
            omega_new.append(omegas[pos])
            cov_new.append(np.diag(covs)[pos])

        # Further filtering where omega > 0
        for i in range(len(self.magnitudes)):
            pos_cuts2 = np.where(omega_new[i] > 0)[0]
            theta_new[i] = theta_new[i][pos_cuts2]
            omega_new[i] = omega_new[i][pos_cuts2]
            cov_new[i] = cov_new[i][pos_cuts2]

        # Additional filtering for a specific case
        if z == 7:
            mask = np.ones(len(theta_new[1]), dtype=bool)
            mask[1] = False
            theta_new[1] = theta_new[1][mask]
            omega_new[1] = omega_new[1][mask]
            cov_new[1] = cov_new[1][mask]

        # Convert theta from arcsec to rad
        for i in range(len(theta_new)):
            theta_new[i] *= np.pi / (3600.0 * 180.0)

        # Compute sigma squared, ensuring it's at least as large as (minError * omega) ** 2
        min_error = 0.1
        sigmasq = []
        for i in range(len(self.magnitudes)):
            sigmasq.append(np.maximum((min_error * omega_new[i]) ** 2, cov_new[i]))

        return theta_new, omega_new, sigmasq

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
        # # Tinker fit
        # _Deltahalo = 200
        # _yhalo = np.log10(_Deltahalo)
        # _Abias = 1.0 + 0.24 * _yhalo * np.exp(-((4.0 / _yhalo) ** 4.0))
        # _abias = 0.44 * _yhalo - 0.88
        # _Bbias = 0.183
        # _bbias = 1.5
        # _Cbias = 0.019 + 0.107 * _yhalo + 0.19 * np.exp(-((4.0 / _yhalo) ** 4.0))
        # _cbias = 2.4

        # return (
        #     1.0
        #     - _Abias * (nu**_abias / (nu**_abias + delta_c**_abias))
        #     + _Bbias * nu**_bbias
        #     + _Cbias * nu**_cbias
        # )

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
        # c = data.mcmc_parameters["A_c"]["current"] * (
        #    10.14
        #    * ((Mh / (2 * 1e12 / data.mcmc_parameters["h"]["current"])) ** (-0.081))
        #    * ((1 + z) ** (-1.01))
        # )
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
        epsilonstar_central = 10 ** (
            data.mcmc_parameters["log10epsilonstar_central_slope"]["current"]
            * np.log10((1 + z) / (1 + 6))
            + data.mcmc_parameters["log10epsilonstar_central_icept"]["current"]
        )
        Mc_central = 10 ** (
            data.mcmc_parameters["log10Mc_central_slope"]["current"]
            * np.log10((1 + z) / (1 + 6))
            + data.mcmc_parameters["log10Mc_central_icept"]["current"]
        )
        sigma_MUV_central = data.mcmc_parameters["sigma_MUV_central"]["current"]
        Mmin_central = 10 ** (data.mcmc_parameters["log10Mmin_central"]["current"])

        alphastar_satellite = data.mcmc_parameters["alphastar_satellite"]["current"]
        betastar_satellite = data.mcmc_parameters["betastar_satellite"]["current"]
        epsilonstar_satellite = 10 ** (
            data.mcmc_parameters["log10epsilonstar_satellite_slope"]["current"]
            * np.log10((1 + z) / (1 + 6))
            + data.mcmc_parameters["log10epsilonstar_satellite_icept"]["current"]
        )
        Mc_satellite = 10 ** (
            data.mcmc_parameters["log10Mc_satellite_slope"]["current"]
            * np.log10((1 + z) / (1 + 6))
            + data.mcmc_parameters["log10Mc_satellite_icept"]["current"]
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
            thetasdata, acf_data, errorsq = MTNG_data_at_z
            # acf_theory = np.zeros(
            #     (len(self.magnitudes), len(np.unique(MTNG_data_at_z[0])))
            # )

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
                # print(z, mags)
                # plt.figure()
                # plt.loglog(self.Mhalos, N_c, color="red", label="centrals")
                # plt.loglog(self.Mhalos, N_s, color="blue", label="satellites")
                # plt.title("z = {} , mags = {} to {}".format(z, mags[0], mags[1]))
                # plt.legend()
                # plt.xlabel("Mhalo (main halo)")
                # plt.ylabel("HOD")
                # plt.axis(ymin=1e-2, ymax=50)
                # plt.show()

                Ngals = N_c + N_s
                ngals_of_z = self.ngal_of_z(HMFs, self.Mhalos, Ngals)

                uNFW = self.fourier_of_NFW(
                    data, z, self.Mhalos, self.k_array[:, np.newaxis]
                )
                Pgal = self.P_1_halo(
                    HMFs, self.Mhalos, N_c, N_s, uNFW, ngals_of_z
                ) + self.P_2_halo(
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
                acf_theory = np.array(
                    [
                        self.omega_theta(theta, yy, G, rcomoving, drcomovingdz)
                        for theta in thetasdata[num_mags]
                    ]
                )

                chisq += np.sum(
                    (acf_theory - acf_data[num_mags]) ** 2 / errorsq[num_mags]
                )

                # thetas = np.geomspace(1., 1000., 50) * np.pi / 3600. / 180.
                # acf_theory = [self.omega_theta(theta, yy, G, rcomoving, drcomovingdz) for theta in thetas]

                # plt.subplot(3, 5, (num_mags + numz * 5 + 1))
                # plt.loglog(thetas * 180 * 3600 / np.pi, acf_theory, color="blue", ls="dashed")
                # if z == 7:
                #     plt.xlabel(r"$\theta$ [arcsec]")
                # if num_mags == 0:
                #    plt.ylabel(r"$\omega(\theta)$")
                # length = len(np.unique(thetasdata))

                # plt.errorbar(
                #     np.unique(thetasdata) / np.pi * 180 * 3600,
                #     acf_data[length * num_mags : length * num_mags + length],
                #     yerr=np.sqrt(errorsq[length * num_mags : length * num_mags + length]),
                #     ls="None",
                #     marker=".",
                #     markersize=5,
                #     markeredgewidth=1.5,
                #     elinewidth=1.5,
                #     color="black",
                #     zorder=0,
                # )

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
