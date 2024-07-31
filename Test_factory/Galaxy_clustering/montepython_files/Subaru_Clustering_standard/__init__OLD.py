from likelihood_class import Likelihood

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simps
from scipy.special import erf, sici
from mcfit import Hankel
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Subaru_Clustering_standard(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Define order of Gaussian quadrature integration
        self.points, self.weights = np.polynomial.legendre.leggauss(25)

        # Import Subaru galaxy clustering data and covariances
        data_z4_bin1 = np.loadtxt(data.path["data"]+"/Subaru_clustering/acf_z=4_23.00_24.00.data", unpack=True)
        data_z4_bin2 = np.loadtxt(data.path["data"]+"/Subaru_clustering/acf_z=4_24.00_25.00.data", unpack=True)
        data_z4_bin3 = np.loadtxt(data.path["data"]+"/Subaru_clustering/acf_z=4_25.00_26.00.data", unpack=True)
        data_z5_bin1 = np.loadtxt(data.path["data"]+"/Subaru_clustering/acf_z=5_23.00_24.00.data", unpack=True)
        data_z5_bin2 = np.loadtxt(data.path["data"]+"/Subaru_clustering/acf_z=5_24.00_25.00.data", unpack=True)
        data_z5_bin3 = np.loadtxt(data.path["data"]+"/Subaru_clustering/acf_z=5_25.00_26.00.data", unpack=True)
        data_z6_bin1 = np.loadtxt(data.path["data"]+"/Subaru_clustering/acf_z=6_24.00_25.00.data", unpack=True)

        cov_z4_bin1 = np.loadtxt(data.path["data"]+"/Subaru_clustering/cov_matrix_z=4_23.00_24.00.data")
        cov_z4_bin2 = np.loadtxt(data.path["data"]+"/Subaru_clustering/cov_matrix_z=4_24.00_25.00.data")
        cov_z4_bin3 = np.loadtxt(data.path["data"]+"/Subaru_clustering/cov_matrix_z=4_25.00_26.00.data")
        cov_z5_bin1 = np.loadtxt(data.path["data"]+"/Subaru_clustering/cov_matrix_z=5_23.00_24.00.data")
        cov_z5_bin2 = np.loadtxt(data.path["data"]+"/Subaru_clustering/cov_matrix_z=5_24.00_25.00.data")
        cov_z5_bin3 = np.loadtxt(data.path["data"]+"/Subaru_clustering/cov_matrix_z=5_25.00_26.00.data")
        cov_z6_bin1 = np.loadtxt(data.path["data"]+"/Subaru_clustering/cov_matrix_z=6_24.00_25.00.data")

        # Cut the data in the same places as in Goldrush IV for the time being
        pos_z4_bin1 = np.where((data_z4_bin1[0] > 2) & (data_z4_bin1[0] <= 10) | (data_z4_bin1[0] > 120) & (data_z4_bin1[0] <= 300))[0]
        pos_z4_bin2 = np.where((data_z4_bin2[0] > 2) & (data_z4_bin2[0] <= 10) | (data_z4_bin2[0] > 90))[0]
        pos_z4_bin3 = np.where((data_z4_bin3[0] > 2) & (data_z4_bin3[0] <= 10) | (data_z4_bin3[0] > 90))[0]
        pos_z5_bin1 = np.where((data_z5_bin1[0] > 2) & (data_z5_bin1[0] <= 10) | (data_z5_bin1[0] > 120) & (data_z5_bin1[0] <= 300))[0]
        pos_z5_bin2 = np.where((data_z5_bin2[0] > 2) & (data_z5_bin2[0] <= 10) | (data_z5_bin2[0] > 90))[0]
        pos_z5_bin3 = np.where((data_z5_bin3[0] > 2) & (data_z5_bin3[0] <= 10) | (data_z5_bin3[0] > 90))[0]
        pos_z6_bin1 = np.where((data_z6_bin1[0] > 2) & (data_z6_bin1[0] <= 10) | (data_z6_bin1[0] > 90))[0]

        data_z4_bin1 = data_z4_bin1[:2,pos_z4_bin1]
        data_z4_bin2 = data_z4_bin2[:2,pos_z4_bin2]
        data_z4_bin3 = data_z4_bin3[:2,pos_z4_bin3]
        data_z5_bin1 = data_z5_bin1[:2,pos_z5_bin1]
        data_z5_bin2 = data_z5_bin2[:2,pos_z5_bin2]
        data_z5_bin3 = data_z5_bin3[:2,pos_z5_bin3]
        data_z6_bin1 = data_z6_bin1[:2,pos_z6_bin1]

        self.cov_z4_bin1 = cov_z4_bin1[pos_z4_bin1,:][:,pos_z4_bin1]
        self.cov_z4_bin2 = cov_z4_bin2[pos_z4_bin2,:][:,pos_z4_bin2]
        self.cov_z4_bin3 = cov_z4_bin3[pos_z4_bin3,:][:,pos_z4_bin3]
        self.cov_z5_bin1 = cov_z5_bin1[pos_z5_bin1,:][:,pos_z5_bin1]
        self.cov_z5_bin2 = cov_z5_bin2[pos_z5_bin2,:][:,pos_z5_bin2]
        self.cov_z5_bin3 = cov_z5_bin3[pos_z5_bin3,:][:,pos_z5_bin3]
        self.cov_z6_bin1 = cov_z6_bin1[pos_z6_bin1,:][:,pos_z6_bin1]

        inv_cov_z4_bin1 = np.linalg.inv(self.cov_z4_bin1)
        inv_cov_z4_bin2 = np.linalg.inv(self.cov_z4_bin2)
        inv_cov_z4_bin3 = np.linalg.inv(self.cov_z4_bin3)
        inv_cov_z5_bin1 = np.linalg.inv(self.cov_z5_bin1)
        inv_cov_z5_bin2 = np.linalg.inv(self.cov_z5_bin2)
        inv_cov_z5_bin3 = np.linalg.inv(self.cov_z5_bin3)
        inv_cov_z6_bin1 = np.linalg.inv(self.cov_z6_bin1)

        degrees_to_rad = np.pi / 3600. / 180.
        self.Subaru_redshifts = [4, 5, 6]
        self.Subaru_data = {"4": [[23., 24., data_z4_bin1[0]*degrees_to_rad, data_z4_bin1[1], inv_cov_z4_bin1], [24., 25., data_z4_bin2[0]*degrees_to_rad, data_z4_bin2[1], inv_cov_z4_bin2], [25., 26., data_z4_bin3[0]*degrees_to_rad, data_z4_bin3[1], inv_cov_z4_bin3]],
                            "5": [[23., 24., data_z5_bin1[0]*degrees_to_rad, data_z5_bin1[1], inv_cov_z5_bin1], [24., 25., data_z5_bin2[0]*degrees_to_rad, data_z5_bin2[1], inv_cov_z5_bin2], [25., 26., data_z5_bin3[0]*degrees_to_rad, data_z5_bin3[1], inv_cov_z5_bin3]],
                            "6": [[24., 25., data_z6_bin1[0]*degrees_to_rad, data_z6_bin1[1], inv_cov_z6_bin1]]}

        # Import and normalise redshift window functions
        self.Subaru_window_z4_data = np.loadtxt(data.path["data"]+"/Subaru_clustering/Subaru_window_z=4.txt", unpack=True)
        self.Subaru_window_z5_data = np.loadtxt(data.path["data"]+"/Subaru_clustering/Subaru_window_z=5.txt", unpack=True)
        self.Subaru_window_z6_data = np.loadtxt(data.path["data"]+"/Subaru_clustering/Subaru_window_z=6.txt", unpack=True)
        self.Subaru_window_z4_data[1] /= simps(self.Subaru_window_z4_data[1], self.Subaru_window_z4_data[0])
        self.Subaru_window_z5_data[1] /= simps(self.Subaru_window_z5_data[1], self.Subaru_window_z5_data[0])
        self.Subaru_window_z6_data[1] /= simps(self.Subaru_window_z6_data[1], self.Subaru_window_z6_data[0])
        self.redshift_windows = {"4": self.Subaru_window_z4_data,
                                 "5": self.Subaru_window_z5_data,
                                 "6": self.Subaru_window_z6_data
                                 }

        # Beta functions for the dust correction
        betadata = np.loadtxt(data.path["data"]+"/Subaru_clustering/Beta_params_Bouwens2014.txt", unpack=True)
        self.betainterp = PchipInterpolator(betadata[0], betadata[1], extrapolate=False)
        self.dbetadMUVinterp = PchipInterpolator(betadata[0], betadata[2], extrapolate=False)

        # Halo masses which we integrate over
        self.Mhalos = np.geomspace(1e6, 1e16, 600)
        self.Msubhalos = np.geomspace(1e6, 1e16, 600)

        # Number of redshifts points within a redshift window which we use to interpolate
        self.len_sub_redshifts = 10

        # Number of elements in k-array and the array itself
        self.k_length = 100
        self.k_array = np.geomspace(1e-3, 200., self.k_length)

        # Initialise Hankel transform function from mcfit
        self.Hankel_transform = Hankel(self.k_array, lowring=True)

        return

    def betaAverage(self, z, MUV):
        if MUV < -19.5:
            return self.dbetadMUVinterp(z) * (MUV + 19.5) + self.betainterp(z)
        return (self.betainterp(z) + 2.33) * np.exp((self.dbetadMUVinterp(z) * (MUV + 19.5)) / (self.betainterp(z) + 2.33)) - 2.33

    def AUV(self, z, MUV):
        if z < 2.5 or z > 8:
            return 0.

        sigmabeta = 0.34
        # return max(0., 4.54 + 0.2 * np.log(10) * (2.07**2) * (sigmabeta**2) + 2.07 * self.betaAverage(z, MUV)) # Overzier 2011
        # return max(0., 4.43 + 0.2 * np.log(10) * (1.99**2) * (sigmabeta**2) + 1.99 * self.betaAverage(z, MUV)) # Meurer 1999
        # return max(0., 3.36 + 0.2 * np.log(10) * (2.04**2) * (sigmabeta**2) + 2.04 * self.betaAverage(z, MUV)) # Casey 2014
        return max(0., 2.45 + 0.2 * np.log(10) * (1.1**2) * (sigmabeta**2) + 1.1 * self.betaAverage(z, MUV)) # Bouwens 2016

    def rcomoving(self, z, Omega_m, h):
        return self.c * self.integrator(lambda x: 1/np.sqrt(Omega_m * np.power(1 + x,3) + 1. - Omega_m), 0., z) / (100. * h)

    def drcomovingdz(self, z, Omega_m, h):
        return self.c / np.sqrt(Omega_m * np.power(1 + z,3) + 1. - Omega_m) / (100. * h)

    def integrator(self, f, a, b):
        sub = (b - a) / 2.
        add = (b + a) / 2.

        if sub == 0:
            return 0.

        return sub * np.dot(f(sub * self.points + add), self.weights)

    def mUV_to_MUV(self, data, z, mUV_min, mUV_max):
        Omega_m = data.mcmc_parameters['Omega_m']['current']
        hubble = data.mcmc_parameters['h']['current']
        conversion = -5. * (np.log10((1. + z) * self.rcomoving(z, Omega_m, hubble) * 1e6) - 1.) + 2.5 * np.log10(1. + z)
        return mUV_min + conversion, mUV_max + conversion

    def HMF(self, cosmo, data, z, Mh):
        rhoM = np.power(data.mcmc_parameters['h']['current'], 2) * data.mcmc_parameters['Omega_m']['current'] * self.rho_crit
        deltaM = 100
        sigma =  cosmo.sigma_cb(np.power(3. * Mh / (4. * np.pi * rhoM), 1./3), z)
        dsigmadM = (cosmo.sigma_cb(np.power(3. * (Mh+deltaM) / (4. * np.pi * rhoM), 1./3), z) - cosmo.sigma_cb(np.power(3. * (Mh-deltaM) / (4. * np.pi * rhoM), 1./3), z)) / (2. * deltaM)
        return -self.AST * np.sqrt(2. * self.aST / np.pi) * (1. + np.power(np.power(sigma,2) / (self.aST * np.power(self.deltaST, 2)), self.pST)) * (self.deltaST / sigma) * np.exp(-self.aST * np.power(self.deltaST, 2) / (2. * np.power(sigma, 2))) * (rhoM / (Mh * sigma)) * dsigmadM

    def SHMF(self, z, Mh, Ms):
        a = 6e-4 * Mh * (2. - 0.08 * np.log10(Mh))**2 * (np.log10(1e5 * Mh))**(-0.1) * (1 + z)
        b = 8e-5 * Mh * (2. - 0.08 * np.log10(Mh)) * (np.log10(1e5 * Mh))**(-0.08 * z) * (np.log10(1e8 * Mh))**(-1.) * (np.log10(1e-18 * Mh))**(2.)
        Mt = 0.05 * (1. + z) * Mh
        alpha = 0.2 + 0.02 * z
        beta = 3.
        return (a + b * Ms**alpha) * np.exp(-(Ms/Mt)**beta) / Ms**2

    def N_central(self, P_central, Mhalos, Mmin_central):
        return P_central * np.exp(-Mmin_central/Mhalos)

    def N_satellite(self, SHMFs, P_satellite, Msubhalos, Mmin_satellite):
        return simps(np.multiply(SHMFs, P_satellite * np.exp(-Mmin_satellite/Msubhalos)), Msubhalos, axis=1)

    def ngal_of_z(self, HMFs, Mhalos, Ngal):
        return simps(HMFs * Ngal, Mhalos)

    def fourier_of_NFW(self, data, z, Mh, k):
        Delta_vir = 200.
        c = data.mcmc_parameters['A_c']['current'] * 10**(1.3081 - 0.1078 * (1 + z) + 0.00398 * (1 + z)**2 + (0.0223 - 0.0944 * (1 + z)**(-0.3907)) * np.log10(Mh))
        rhoC = np.power(data.mcmc_parameters['h']['current'], 2) * self.rho_crit
        Rs = (3. * Mh / (4 * np.pi * Delta_vir * rhoC))**(1./3) / c
        rhoS = Mh / (4. * np.pi * Rs**3 * (np.log(1 + c) - c/(1. + c)))
        return 4 * np.pi * rhoS * Rs**3 / Mh * (np.sin(k * Rs) * (sici((1 + c) * k * Rs)[0] - sici(k * Rs)[0]) - np.sin(c * k * Rs) / ((1 + c) * k * Rs) + np.cos(k * Rs) * (sici((1 + c) * k * Rs)[1] - sici(k * Rs)[1]))

    def MUV_from_Mh(self, cosmo, data, z, Mh, alphastar, betastar, epsilonstar, Mc):
        Q = data.mcmc_parameters['Q']['current']
        rhoM = np.power(data.mcmc_parameters['h']['current'], 2) * data.mcmc_parameters['Omega_m']['current'] * self.rho_crit
        sigma1 =  cosmo.sigma_cb(np.power(3. * Mh / Q / (4. * np.pi * rhoM), 1./3), z)
        sigma2 =  cosmo.sigma_cb(np.power(3. * Mh / (4. * np.pi * rhoM), 1./3), z)
        functionf = 1./np.sqrt(sigma1**2 - sigma2**2)
        dgrowthdz = -cosmo.scale_independent_growth_factor_f(z) * cosmo.scale_independent_growth_factor(z) / (1.+z)
        Mhdot = -(1+z) * cosmo.Hubble(z) * self.invMpctoinvYear * 1.686 * np.sqrt(2./np.pi) * Mh * functionf * dgrowthdz / cosmo.scale_independent_growth_factor(z)**2
        return -2.5 * np.log10(epsilonstar * Mhdot / ((Mh/Mc)**alphastar + (Mh/Mc)**betastar) / self.kappaUV) + 51.63

    def halo_bias(self, cosmo, data, z, Mh):
        delta_c = 1.686
        rhoM = np.power(data.mcmc_parameters['h']['current'], 2) * data.mcmc_parameters['Omega_m']['current'] * self.rho_crit
        nu = delta_c / cosmo.sigma_cb(np.power(3. * Mh / (4. * np.pi * rhoM), 1./3), z)
        return 1. + 1./(np.sqrt(0.707) * delta_c) * (np.sqrt(0.707) * (0.707 * nu**2) + np.sqrt(0.707) * 0.5 * (0.707 * nu**2)**(0.4) - (0.707 * nu**2)**0.6 / ((0.707 * nu**2)**0.6 + 0.14))

    def P_1_halo(self, HMFs, Mhalos, N_central, N_satellite, uNFW, ngals_of_z):
        return simps((2. * N_central * N_satellite * uNFW + N_satellite**2 * uNFW**2) * HMFs, Mhalos) / ngals_of_z**2

    def P_2_halo(self, cosmo, k, z, HMFs, biases, Mhalos, N_central, N_satellite, uNFW, ngals_of_z):
        return cosmo.pk_cb_lin(k, z) * simps(HMFs * (N_central + N_satellite * uNFW) * biases, Mhalos)**2. / ngals_of_z**2

    def scatter_probability(self, MUV, width, MUV_av, sigma_MUV):
        return 0.5 * (erf((MUV_av-MUV+width/2.) / (sigma_MUV*np.sqrt(2))) - erf((MUV_av-MUV-width/2.) / (sigma_MUV*np.sqrt(2))))

    def omega_theta(self, theta, yy, G, z_sub_array, rcomovings, drcomovingsdz, redshifts_window, window):
        PP  = np.array([PchipInterpolator(np.log(yy / theta), G[numzsub], extrapolate=False)(np.log(rcomovings[numzsub])) / 2. / np.pi for numzsub in range(len(z_sub_array))])
        integrand = PchipInterpolator(z_sub_array, PP/drcomovingsdz, extrapolate=False)(redshifts_window) * window**2
        return simps(integrand, redshifts_window)

    def loglkl(self, cosmo, data):

        Omega_m = data.mcmc_parameters['Omega_m']['current']
        hubble = data.mcmc_parameters['h']['current']

        chisq = 0.
        # start = time.time()
        for numz, z in enumerate(self.Subaru_redshifts):
            redshifts_window, window = self.redshift_windows["{}".format(z)]
            sub_redshifts = np.linspace(min(redshifts_window), max(redshifts_window), self.len_sub_redshifts)

            Subaru_data_at_z = self.Subaru_data["{}".format(z)]
            rcomovings = []
            drcomovingsdz = []
            Pgal = np.zeros((len(Subaru_data_at_z), len(sub_redshifts), self.k_length))

            for numzsub, zsub in enumerate(sub_redshifts):
                alphastar_central = data.mcmc_parameters['alphastar_central']['current']
                betastar_central = data.mcmc_parameters['betastar_central']['current']
                epsilonstar_central = 10**(data.mcmc_parameters['log10epsilonstar_central_slope']['current'] * np.log10((1 + zsub)/(1 + 5)) + data.mcmc_parameters['log10epsilonstar_central_icept']['current'])
                Mc_central = 10**(data.mcmc_parameters['log10Mc_central_slope']['current'] * np.log10((1 + zsub)/(1 + 5)) + data.mcmc_parameters['log10Mc_central_icept']['current'])
                sigma_MUV_central = data.mcmc_parameters['sigma_MUV_central']['current']
                Mmin_central = 10**(data.mcmc_parameters['log10Mmin_central']['current'])

                alphastar_satellite = data.mcmc_parameters['alphastar_satellite']['current']
                betastar_satellite = data.mcmc_parameters['betastar_satellite']['current']
                epsilonstar_satellite = 10**(data.mcmc_parameters['log10epsilonstar_satellite_slope']['current'] * np.log10((1 + zsub)/(1 + 5)) + data.mcmc_parameters['log10epsilonstar_satellite_icept']['current'])
                Mc_satellite = 10**(data.mcmc_parameters['log10Mc_satellite_slope']['current'] * np.log10((1 + zsub)/(1 + 5)) + data.mcmc_parameters['log10Mc_satellite_icept']['current'])
                sigma_MUV_satellite = data.mcmc_parameters['sigma_MUV_satellite']['current']
                Mmin_satellite = 10**(data.mcmc_parameters['log10Mmin_satellite']['current'])

                rcomovings.append(self.rcomoving(zsub, Omega_m, hubble))
                drcomovingsdz.append(self.drcomovingdz(zsub, Omega_m, hubble))

                HMFs = np.array([self.HMF(cosmo, data, zsub, mass) for mass in self.Mhalos])
                SHMFs = np.array([self.SHMF(zsub, mass, self.Msubhalos) for mass in self.Mhalos])

                MUV_avs_central = np.array([self.MUV_from_Mh(cosmo, data, zsub, mass, alphastar_central, betastar_central, epsilonstar_central, Mc_central) for mass in self.Mhalos])
                MUV_avs_satellite = np.array([self.MUV_from_Mh(cosmo, data, zsub, mass, alphastar_satellite, betastar_satellite, epsilonstar_satellite, Mc_satellite) for mass in self.Msubhalos])

                halo_biases = np.array([self.halo_bias(cosmo, data, zsub, mass) for mass in self.Mhalos])

                for num_MUV_bins in range(len(Subaru_data_at_z)):
                    mUV_min, mUV_max = Subaru_data_at_z[num_MUV_bins][:2]
                    MUV_min, MUV_max = self.mUV_to_MUV(data, zsub, mUV_min, mUV_max)
                    MUV_min -= self.AUV(zsub, MUV_min)
                    MUV_max -= self.AUV(zsub, MUV_max)

                    P_central = self.scatter_probability(0.5*(MUV_min + MUV_max), MUV_max - MUV_min, MUV_avs_central, sigma_MUV_central)
                    P_satellite = self.scatter_probability(0.5*(MUV_min + MUV_max), MUV_max - MUV_min, MUV_avs_satellite, sigma_MUV_satellite)

                    N_c = self.N_central(P_central, self.Mhalos, Mmin_central)
                    N_s = self.N_satellite(SHMFs, P_satellite, self.Msubhalos, Mmin_satellite)

                    Ngals = N_c + N_s
                    ngals_of_z = self.ngal_of_z(HMFs, self.Mhalos, Ngals)

                    for numk, k in enumerate(self.k_array):
                        uNFW = self.fourier_of_NFW(data, zsub, self.Mhalos, k)
                        Pgal[num_MUV_bins, numzsub, numk] = self.P_1_halo(HMFs, self.Mhalos, N_c, N_s, uNFW, ngals_of_z) + self.P_2_halo(cosmo, k, zsub, HMFs, halo_biases, self.Mhalos, N_c, N_s, uNFW, ngals_of_z)

            for num_MUV_bins in range(len(Subaru_data_at_z)):
                yy, G = self.Hankel_transform(Pgal[num_MUV_bins, :, :], extrap=False)
                thetasdata, acf_data, inv_cov = Subaru_data_at_z[num_MUV_bins][2:]
                thetas = np.geomspace(1., 1000., 50) * np.pi / 3600. / 180.
                acf_theory = [self.omega_theta(theta, yy, G, sub_redshifts, rcomovings, drcomovingsdz, redshifts_window, window) for theta in thetas]
                acf_theory2 = [self.omega_theta(theta, yy, G, sub_redshifts, rcomovings, drcomovingsdz, redshifts_window, window) for theta in thetasdata]
                chisq += np.linalg.multi_dot([(acf_theory2 - acf_data), inv_cov, (acf_theory2 - acf_data)])
                plt.subplot(len(self.Subaru_redshifts), 3, 1*(3) + (num_MUV_bins+1))
                if z == 4:
                    plt.loglog(thetas * 180 * 3600 / np.pi, acf_theory, color="blue", ls="dashed")
                    plt.xlabel(r"$\theta$ [arcsec]")
                if num_MUV_bins == 0:
                   plt.ylabel(r"$\omega(\theta)$")

                if z == 4:
                    plt.errorbar(thetasdata * 180 * 3600 / np.pi, acf_data, yerr=np.sqrt(np.diagonal([self.cov_z4_bin1,self.cov_z4_bin2,self.cov_z4_bin3][num_MUV_bins])), ls="None", marker=".", markersize=5, markeredgewidth=1.5, elinewidth=1.5, color="black", zorder=0)
                    plt.title(r"$m_\mathrm{UV} = $" + "{}".format(["23 - 24", "24 - 25", "25 - 26"][num_MUV_bins]))
                # if z == 5:
                #     plt.errorbar(thetasdata * 180 * 3600 / np.pi, acf_data, yerr=np.sqrt(np.diagonal([self.cov_z5_bin1,self.cov_z5_bin2,self.cov_z5_bin3][num_MUV_bins])), ls="None", marker=".", markersize=5, markeredgewidth=1.5, elinewidth=1.5, color="black", zorder=0)
                #     plt.title(r"$m_\mathrm{UV} = $" + "{}".format(["23 - 24", "24 - 25", "25 - 26"][num_MUV_bins]))
                # if z == 6:
                #     plt.errorbar(thetasdata * 180 * 3600 / np.pi, acf_data, yerr=np.sqrt(np.diagonal([self.cov_z6_bin1][num_MUV_bins])), ls="None", marker=".", markersize=5, markeredgewidth=1.5, elinewidth=1.5, color="black", zorder=0)
                    # plt.title(r"$m_\mathrm{UV} = 24 - 25$")
                # plt.fill_between(thetas * 180 * 3600 / np.pi, data_lower_upper[num_MUV_bins][0], data_lower_upper[num_MUV_bins][1], facecolor="none", color="blue", linewidth=0.0, alpha=0.3, zorder=3)

                # print("z = ", z, "\t", "MUV bin: ", num_MUV_bins + 1)
       #  end = time.time()
       #  print(end - start)
        plt.show()

        chisq += ((data.mcmc_parameters['omega_b']['current'] * data.mcmc_parameters['omega_b']['scale'] - self.omegab_BBN) / self.omegab_BBN_error)**2

        return -0.5 * chisq