import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simps
from scipy.special import erf
from classy import Class
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from contextlib import closing

cosmo = Class()
args = {'P_k_max_1/Mpc': 40., 'T_ncdm': 0.71611, 'N_ur': 2.0308, 'N_ncdm': 1, 'tau_reio': 0.0544, 'n_s': 0.9649 , 'k_pivot': 0.05, 'omega_b': 0.02236, 'm_ncdm': 0.06, 'h': 0.6727, 'z_max_pk': 13.0, 'output': 'mPk, dTk', 'omega_cdm': 0.1202, 'ln10^{10}A_s': 3.045}
cosmo.set(args)
cosmo.compute(['lensing'])

# JWST_Mstar = np.loadtxt("Labbe_Mstar.txt")

# Define order of Gaussian quadrature integration
points, weights = np.polynomial.legendre.leggauss(25)
points_highres, weights_highres = np.polynomial.legendre.leggauss(150)

# Halo masses which we integrate over
Mhalos = np.geomspace(1e9, 1e14, 600)

c = 299792.458
rho_crit = 2.77463e11
kappaUV = 1.15e-28
invMpctoinvYear = 3.066067275784999e-07
AST = 0.3222
aST = 0.85
pST = 0.3
deltaST = 1.686

# Comoving radial distance
def rcomoving(z, Omega_m, h):
    c * integrator(lambda x: 1/np.sqrt(Omega_m * np.power(1 + x,3) + 1. - Omega_m), 0., z) / (100. * h)

# Gaussian integrator
def integrator(f, a, b, highres=False):
    sub = (b - a) / 2.
    add = (b + a) / 2.
    if sub == 0:
        return 0.
    if not highres:
        return sub * np.dot(f(sub * points + add), weights)
    return sub * np.dot(f(sub * points_highres + add), weights_highres)

# The Gaussian bump added to the power spectrum
def Gaussian_bump(k, amp, mean, sigma):
    k1 = 0.5
    if k < k1:
        return 0.
    return amp * np.exp(-(np.log(k) - mean)**2 / (2. * sigma**2))

# Calculate the mass variance manually intead from CLASS
def calculate_sigma(R, kmin, kmax, pk, z, amp, mean, sigma):
    mps = np.vectorize(lambda y: pk(y, z) * (1. + Gaussian_bump(y, amp, mean, sigma)))
    window = lambda x: 3. * (np.sin(x * R) - (x * R) * np.cos(x * R)) / (x * R)**3
    return np.sqrt(integrator(lambda lnx: window(np.exp(lnx))**2 * mps(np.exp(lnx)) * np.exp(lnx)**3, np.log(kmin), np.log(kmax), True) / 2. / np.pi**2)

# Compute the HMF and average <MUV> given halo mass Mh
def HMF_and_MUV_from_Mh(cosmo, z, Mh, ks, amp, mean, sigma, alphastar, betastar, eps1, eps2, mc1, mc2, Q):
    rhoM = np.power(cosmo.h(), 2) * cosmo.Omega_m() * rho_crit
    deltaM = 100
    Gaussian_amp = amp
    Gaussian_mean = mean
    Gaussian_sigma = sigma

    sigma = calculate_sigma(np.power(3. * Mh / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z, Gaussian_amp, Gaussian_mean, Gaussian_sigma)
    sigma_plus_deltaM = calculate_sigma(np.power(3. * (Mh+deltaM) / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z, Gaussian_amp, Gaussian_mean, Gaussian_sigma)
    sigma_minus_deltaM = calculate_sigma(np.power(3. * (Mh-deltaM) / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z, Gaussian_amp, Gaussian_mean, Gaussian_sigma)
    dsigmadM = (sigma_plus_deltaM - sigma_minus_deltaM) / (2. * deltaM)

    #################

    epsilonstar = 10**(eps1 * np.log10((1 + z)/(1 + 6)) + eps2)
    Mc = 10**(mc1 * np.log10((1 + z)/(1 + 6)) + mc2)

    sigma1 = calculate_sigma(np.power(3. * Mh / Q / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z, Gaussian_amp, Gaussian_mean, Gaussian_sigma)
    functionf = 1./np.sqrt(sigma1**2 - sigma**2)
    dgrowthdz = -cosmo.scale_independent_growth_factor_f(z) * cosmo.scale_independent_growth_factor(z) / (1.+z)
    Mhdot = -(1+z) * cosmo.Hubble(z) * invMpctoinvYear * 1.686 * np.sqrt(2./np.pi) * Mh * functionf * dgrowthdz / cosmo.scale_independent_growth_factor(z)**2

    return -AST * np.sqrt(2. * aST / np.pi) * (1. + np.power(np.power(sigma,2) / (aST * np.power(deltaST, 2)), pST)) * (deltaST / sigma) * np.exp(-aST * np.power(deltaST, 2) / (2. * np.power(sigma, 2))) * (rhoM / (Mh * sigma)) * dsigmadM, -2.5 * np.log10(epsilonstar * Mhdot / ((Mh/Mc)**alphastar + (Mh/Mc)**betastar) / kappaUV) + 51.63

# Integrated Gaussian distribution for given average magnitude
def first_integrand(MUV, width, MUV_av, sigma_MUV):
    return 0.5 * (erf((MUV_av-MUV+width/2.) / (sigma_MUV*np.sqrt(2))) - erf((MUV_av-MUV-width/2.) / (sigma_MUV*np.sqrt(2))))


# # Compute the HMF and average <MUV> given halo mass Mh
# def HMF(cosmo, z, Mh, ks, amp, mean, sigma):
#     rhoM = np.power(cosmo.h(), 2) * cosmo.Omega_m() * rho_crit
#     deltaM = 100
#     Gaussian_amp = amp
#     Gaussian_mean = mean
#     Gaussian_sigma = sigma

#     sigma = calculate_sigma(np.power(3. * Mh / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z, Gaussian_amp, Gaussian_mean, Gaussian_sigma)
#     sigma_plus_deltaM = calculate_sigma(np.power(3. * (Mh+deltaM) / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z, Gaussian_amp, Gaussian_mean, Gaussian_sigma)
#     sigma_minus_deltaM = calculate_sigma(np.power(3. * (Mh-deltaM) / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z, Gaussian_amp, Gaussian_mean, Gaussian_sigma)
#     dsigmadM = (sigma_plus_deltaM - sigma_minus_deltaM) / (2. * deltaM)

#     return -AST * np.sqrt(2. * aST / np.pi) * (1. + np.power(np.power(sigma,2) / (aST * np.power(deltaST, 2)), pST)) * (deltaST / sigma) * np.exp(-aST * np.power(deltaST, 2) / (2. * np.power(sigma, 2))) * (rhoM / (Mh * sigma)) * dsigmadM

def compute_UVLF(args):
    
    weight, lkl, omegab, alpha, beta, eps1, eps2, mc1, mc2, s1, Q, amp, mean, sigma8, Omegam = args


    z = 12.
    MUVs = np.linspace(-23.,-16.,40)
    width = (np.abs(np.diff(MUVs))/2.)[0]
    phis = np.zeros(1 +  len(MUVs))

    # Array of wavenumbers 
    ks = cosmo.get_transfer(z)["k (h/Mpc)"] * 0.6727

    # HMFs and average MUVs only need to be computed at each redshift slice
    HMF_and_MUV_from_Mhs = np.array([HMF_and_MUV_from_Mh(cosmo, z, mass, ks, amp, mean, 0.1, alpha, beta, eps1, eps2, mc1, mc2, Q) for mass in Mhalos])
    HMFs = HMF_and_MUV_from_Mhs[:, 0]
    MUV_avs = HMF_and_MUV_from_Mhs[:, 1]

    for numMUV, MUV in enumerate(MUVs):
        second_integrand = HMFs * first_integrand(MUV, width, MUV_avs, s1) / width
        phis[0] = weight
        phis[numMUV + 1] = simps(second_integrand, Mhalos)
    return z, MUVs, phis

# def Ngals(args):

#     weight, lkl, omegab, alpha, beta, eps1, eps2, mc1, mc2, s1, QQ, amp, mean, sigma8, Omegam, sigma, Mcut = args

#     z = 7.

#     # Array of wavenumbers 
#     ks = cosmo.get_transfer(z)["k (h/Mpc)"] * 0.6727

#     # HMFs and average MUVs only need to be computed at each redshift slice
#     HMFs = np.array([HMF(cosmo, z, mass, ks, amp, mean, sigma) for mass in Mhalos])
#     HMF_interp = PchipInterpolator(Mhalos, HMFs)

#     M_lower = Mcut
#     Mh_array_integrate = np.geomspace(M_lower, 1e14, 500)
#     Ngal = simps(HMF_interp(Mh_array_integrate), Mh_array_integrate) * 1e5

#     return weight, Ngal

#######################

if __name__ == '__main__':

    data_HST = []
    for filepath in glob.iglob('../../Chains/Gaussian_2/*__*.txt'):
        data = np.loadtxt(filepath)
        data_HST.append(data)
    data_HST = np.vstack(np.array(data_HST))

    #######################

    # UVLFs_full = []
    # with closing(Pool(processes=None)) as pool:
    #     for numm, UVLF in enumerate(pool.imap(compute_UVLF, data_HST)):
    #         z = UVLF[0]
    #         MUVs = UVLF[1]
    #         UVLFs_full.append(UVLF[2])
    #         if numm % 2 == 0:
    #             first_row = np.hstack(([z], MUVs))
    #             np.savetxt("UVLFs_HST_z12.txt", np.vstack((first_row, np.array(UVLFs_full))))
    #     pool.terminate()

    # np.savetxt("UVLFs_HST_z12.txt", np.vstack((first_row, np.array(UVLFs_full))))



    pos_bestfit = np.argmin(data_HST[:,1])
    UVLF_bestfit = compute_UVLF(data_HST[pos_bestfit])
    first_row = np.hstack(([UVLF_bestfit[0]], UVLF_bestfit[1]))
    np.savetxt("UVLFs_HST_z12_bestfit.txt", np.vstack((first_row, UVLF_bestfit[2])))
