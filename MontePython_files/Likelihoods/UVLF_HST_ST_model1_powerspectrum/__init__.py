########################################################################
# Written by Nashwan Sabti - 2021
# Please cite 2110.XXXX and 2110.XXXX when using this likelihood
########################################################################

from likelihood_class import Likelihood

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simps
from scipy.special import erf

class UVLF_HST_ST_model1_powerspectrum(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Define order of Gaussian quadrature integration
        self.points, self.weights = np.polynomial.legendre.leggauss(25)
        self.points_highres, self.weights_highres = np.polynomial.legendre.leggauss(1000)
        
        # Import HST UV LF data and impose 20% minimal error
        minError = 0.2
        self.UVLF_HST = np.loadtxt(data.path["data"]+"/UVLF/UVLF_HST.txt")
        self.UVLF_HST[:,4] = np.array(list(map(max,zip(minError*self.UVLF_HST[:,3], self.UVLF_HST[:,4]))))
        self.zs = np.unique(self.UVLF_HST[:,0])

        # Beta functions for the dust correction
        betadata = np.loadtxt(data.path["data"]+"/UVLF/Beta_parameters.txt", unpack=True)
        self.betainterp = PchipInterpolator(betadata[0], betadata[1])
        self.dbetadMUVinterp = PchipInterpolator(betadata[0], betadata[2])

        # Undusting the data
        dust_corr = []
        bin_corr = []
        LF_corr = []
        for item in self.UVLF_HST:
            z, MUV, bin_width = item[0], item[1], item[2]
            new_bin_width = bin_width - self.AUV(z, MUV + bin_width/2) + self.AUV(z, MUV - bin_width/2)
            dust_corr.append(self.AUV(z, MUV))
            bin_corr.append(new_bin_width)
            LF_corr.append(bin_width / new_bin_width)

        self.UVLF_HST[:,1] -= np.array(dust_corr)
        self.UVLF_HST[:,2]  = np.array(bin_corr)
        self.UVLF_HST[:,3] *= np.array(LF_corr)
        self.UVLF_HST[:,4] *= np.array(LF_corr)

        # Halo masses which we integrate over
        self.Mhalos = np.geomspace(1e8, 1e14, 1000)

        return

    # Beta function for the dust extinction
    def betaAverage(self, z, MUV):
        if MUV < -19.5:
            return self.dbetadMUVinterp(z) * (MUV + 19.5) + self.betainterp(z)
        return (self.betainterp(z) + 2.33) * np.exp((self.dbetadMUVinterp(z) * (MUV + 19.5)) / (self.betainterp(z) + 2.33)) - 2.33

    # Dust extinction parameter (only applied at redshifts for which the beta function is measured)
    def AUV(self, z, MUV):
        if z < 2.5 or z > 8:
            return 0.

        sigmabeta = 0.34
        return max(0., 4.54 + 0.2 * np.log(10) * (2.07**2) * (sigmabeta**2) + 2.07 * self.betaAverage(z, MUV)) # Overzier 2011
        # return max(0., 4.43 + 0.2 * np.log(10) * (1.99**2) * (sigmabeta**2) + 1.99 * self.betaAverage(z, MUV)) # Meurer 1999
        # return max(0., 3.36 + 0.2 * np.log(10) * (2.04**2) * (sigmabeta**2) + 2.04 * self.betaAverage(z, MUV)) # Casey 2014
        # return max(0., 2.45 + 0.2 * np.log(10) * (1.1**2) * (sigmabeta**2) + 1.1 * self.betaAverage(z, MUV)) # Bouwens 2016

    # Comoving radial distance
    def rcomoving(self, z, Omega_m, h):
        return self.c * self.integrator(lambda x: 1/np.sqrt(Omega_m * np.power(1 + x,3) + 1. - Omega_m), 0., z) / (100. * h)

    # Gaussian integrator
    def integrator(self, f, a, b, highres=False):
        sub = (b - a) / 2.
        add = (b + a) / 2.
        if sub == 0:
            return 0.
        if not highres:
           return sub * np.dot(f(sub * self.points + add), self.weights)
        return sub * np.dot(f(sub * self.points_highres + add), self.weights_highres)

    # Alcock-Paczynski corrections
    def AP_effect(self, data):
        deltaz = 0.5 # 1/2 width of redshift slice
        Omega_m = data.mcmc_parameters['Omega_m']['current']
        hubble = data.mcmc_parameters['h']['current']

        self.UVLF_data = self.UVLF_HST.copy()
        for z in self.zs:
            # Vratio is the correction to the UV LF
            Vratio =  (np.power(self.rcomoving(z+deltaz, self.Omega_m_HST, self.h_HST),3) - np.power(self.rcomoving(z-deltaz, self.Omega_m_HST, self.h_HST),3)) / (np.power(self.rcomoving(z+deltaz, Omega_m, hubble),3) - np.power(self.rcomoving(z-deltaz, Omega_m, hubble),3))

            # Apply correction to UV LF and error
            self.UVLF_data[self.UVLF_data[:,0]==z,3:] *= Vratio

            # Apply correction to magntiudes, since luminosity distances are affected too
            self.UVLF_data[self.UVLF_data[:,0]==z,1] = self.UVLF_data[self.UVLF_data[:,0]==z,1] - 5. * np.log10(self.rcomoving(z, Omega_m, hubble) / self.rcomoving(z, self.Omega_m_HST, self.h_HST))

    # The multiplication factor of the power spectrum in each bin
    def scaling(self, k, as2, as3, as4):
        k1 = 0.5
        k2 = 2.25
        k3 = 10.
        if k < k1:
            return 1.
        if k >= k1 and k <= k2:
            return as2
        if k > k2 and k <= k3:
            return as3
        return as4

    # Calculate the mass variance manually intead from CLASS
    def calculate_sigma(self, R, kmin, kmax, pk, z, as2, as3, as4):
        mps = np.vectorize(lambda y: pk(y, z) * self.scaling(y, as2, as3, as4))
        window = lambda x: np.heaviside(1. - x * R, 1.)
        return np.sqrt(self.integrator(lambda x: window(x)**2 * mps(x) * x**2, kmin, kmax, True) / 2. / np.pi**2)

    # Compute the HMF and average <MUV> given halo mass Mh. This is done in one function in order not to compute "sigma" again, which takes a long time here
    def HMF_and_MUV_from_Mh(self, cosmo, data, z, Mh, ks):
        rhoM = np.power(data.mcmc_parameters['h']['current'], 2) * data.mcmc_parameters['Omega_m']['current'] * self.rho_crit
        aST = data.mcmc_parameters['aST']['current']
        As2 = 10**data.mcmc_parameters['logas2']['current']
        As3 = 10**data.mcmc_parameters['logas3']['current']
        As4 = 10**data.mcmc_parameters['logas4']['current']

        R = np.power(3. * Mh / (4. * np.pi * rhoM), 1./3) / self.cM
        sigma = self.calculate_sigma(R, min(ks), max(ks), cosmo.pk_cb_lin, z, As2, As3, As4)
        dsigmadM = -cosmo.pk_cb_lin(1/R, z) * self.scaling(1/R, As2, As3, As4) / 12. / np.pi**2 / R**3 / Mh / sigma

        #################

        alphastar = data.mcmc_parameters['alphastar']['current']
        betastar = data.mcmc_parameters['betastar']['current']
        epsilonstar = 10**(data.mcmc_parameters['epsilonstar_slope']['current'] * np.log10((1 + z)/(1 + 6)) + data.mcmc_parameters['epsilonstar_icept']['current'])
        Mc = 10**(data.mcmc_parameters['Mc_slope']['current'] * np.log10((1 + z)/(1 + 6)) + data.mcmc_parameters['Mc_icept']['current'])

        Q = data.mcmc_parameters['Q']['current']
        sigma1 = self.calculate_sigma(np.power(3. * Mh / Q / (4. * np.pi * rhoM), 1./3) / self.cM, min(ks), max(ks), cosmo.pk_cb_lin, z, As2, As3, As4)
        functionf = 1./np.sqrt(sigma1**2 - sigma**2)
        dgrowthdz = -cosmo.scale_independent_growth_factor_f(z) * cosmo.scale_independent_growth_factor(z) / (1.+z)
        Mhdot = -(1+z) * cosmo.Hubble(z) * self.invMpctoinvYear * 1.686 * np.sqrt(2./np.pi) * Mh * functionf * dgrowthdz / cosmo.scale_independent_growth_factor(z)

        return -self.AST * np.sqrt(2. * aST / np.pi) * (1. + np.power(np.power(sigma,2) / (aST * np.power(self.deltaST, 2)), self.pST)) * (self.deltaST / sigma) * np.exp(-aST * np.power(self.deltaST, 2) / (2. * np.power(sigma, 2))) * (rhoM / (Mh * sigma)) * dsigmadM, -2.5 * np.log10(epsilonstar * Mhdot / ((Mh/Mc)**alphastar + (Mh/Mc)**betastar) / self.kappaUV) + 51.63

    # Integrated Gaussian distribution for given average magnitude
    def first_integrand(self, MUV, width, MUV_av, sigma_MUV):
        return 0.5 * (erf((MUV_av-MUV+width/2.) / (sigma_MUV*np.sqrt(2))) - erf((MUV_av-MUV-width/2.) / (sigma_MUV*np.sqrt(2))))

    # Log-likelihood
    def loglkl(self, cosmo, data):
        
        # Compute AP effect
        self.AP_effect(data)

        chisq = 0
        # Iterate over redshifts of data
        for z in self.zs:

            # Array of wavenumbers 
            ks = cosmo.get_transfer(z)["k (h/Mpc)"] * data.mcmc_parameters['h']['current']

            # HMFs and average MUVs only need to be computed at each redshift slice
            HMF_and_MUV_from_Mh = np.array([self.HMF_and_MUV_from_Mh(cosmo, data, z, mass, ks) for mass in self.Mhalos])
            HMFs = HMF_and_MUV_from_Mh[:, 0]
            MUV_avs = HMF_and_MUV_from_Mh[:, 1]

            # Iterate over data at redshift z
            for item in self.UVLF_data[self.UVLF_data[:,0]==z,1:]:
                MUV, width, UVLF_data, UVLF_error = item
                second_integrand = HMFs * self.first_integrand(MUV, width, MUV_avs, data.mcmc_parameters['sigma_MUV']['current']) / width
                chisq += ((simps(second_integrand, self.Mhalos) - UVLF_data) / UVLF_error)**2

        return -0.5 * chisq
