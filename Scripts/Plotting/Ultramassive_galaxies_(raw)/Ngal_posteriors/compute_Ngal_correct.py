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
args = {'P_k_max_1/Mpc': 40., 'T_ncdm': 0.71611, 'N_ur': 2.0308, 'N_ncdm': 1, 'tau_reio': 0.0544, 'n_s': 0.9649 , 'k_pivot': 0.05, 'omega_b': 0.02236, 'm_ncdm': 0.06, 'h': 0.6727, 'z_max_pk': 10.0, 'output': 'mPk, dTk', 'omega_cdm': 0.1202, 'ln10^{10}A_s': 3.045}
cosmo.set(args)
cosmo.compute(['lensing'])

JWST_Mstar = np.loadtxt("Labbe_Mstar.txt")

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
    return c * integrator(lambda x: 1/np.sqrt(Omega_m * np.power(1 + x,3) + 1. - Omega_m), 0., z) / (100. * h)

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
def HMF(cosmo, z, Mh, ks, amp, mean, sigma):
    rhoM = np.power(cosmo.h(), 2) * cosmo.Omega_m() * rho_crit
    deltaM = 100
    Gaussian_amp = amp
    Gaussian_mean = mean
    Gaussian_sigma = sigma

    sigma = calculate_sigma(np.power(3. * Mh / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z, Gaussian_amp, Gaussian_mean, Gaussian_sigma)
    sigma_plus_deltaM = calculate_sigma(np.power(3. * (Mh+deltaM) / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z, Gaussian_amp, Gaussian_mean, Gaussian_sigma)
    sigma_minus_deltaM = calculate_sigma(np.power(3. * (Mh-deltaM) / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z, Gaussian_amp, Gaussian_mean, Gaussian_sigma)
    dsigmadM = (sigma_plus_deltaM - sigma_minus_deltaM) / (2. * deltaM)

    return -AST * np.sqrt(2. * aST / np.pi) * (1. + np.power(np.power(sigma,2) / (aST * np.power(deltaST, 2)), pST)) * (deltaST / sigma) * np.exp(-aST * np.power(deltaST, 2) / (2. * np.power(sigma, 2))) * (rhoM / (Mh * sigma)) * dsigmadM

def Ngals(args):

    weight, lkl, omegab, amp, mean, Omegam, sigma, epsilon = args

    # Iterate over redshifts of data
    Ngal = 0
    zrange1 = [7., 8.5]
    zrange2 = [8.5, 10.]
    for z, log10Mstar, Vol in JWST_Mstar:

        # Array of wavenumbers 
        ks = cosmo.get_transfer(z)["k (h/Mpc)"] * 0.6727
        M_lower = 10**(log10Mstar) / 0.157 / epsilon # epsilon = 0.1

        # HMFs and average MUVs only need to be computed at each redshift slice

        # Ngal += simps(HMF_interp(Mh_array_integrate), Mh_array_integrate) * Vol

        if z == 7.48:
            zz = np.linspace(7., 8.5, 10)
        else:
            zz = np.linspace(8.5, 10., 10)
        dNgaldz = []
        for znew in zz:
            HMFs = np.array([HMF(cosmo, znew, mass, ks, amp, mean, sigma) for mass in Mhalos])
            HMF_interp = PchipInterpolator(Mhalos, HMFs)
            Mh_array_integrate = np.geomspace(M_lower, 1e14, 500)
            dngaldz = simps(HMF_interp(Mh_array_integrate), Mh_array_integrate)
            dVol = (rcomoving(znew, cosmo.Omega_m(), cosmo.h()))**2 * c / np.sqrt(cosmo.Omega_m() * np.power(1 + znew,3) + 1. - cosmo.Omega_m()) / (100. * cosmo.h())  * (38. / 60**2 * (np.pi / 180.)**2)
            dNgaldz.append(dngaldz * dVol)
        Ngal += simps(dNgaldz, zz)

    return weight, Ngal


def Ngals2(args):

    weight, lkl, omegab, amp, mean, Omegam, sigma, epsilon = args

    # Iterate over redshifts of data
    Ngal = 0
    for z, log10Mstar, Vol in JWST_Mstar:

        # Array of wavenumbers 
        ks = cosmo.get_transfer(z)["k (h/Mpc)"] * 0.6727

        # HMFs and average MUVs only need to be computed at each redshift slice
        HMFs = np.array([HMF(cosmo, z, mass, ks, amp, mean, sigma) for mass in Mhalos])
        HMF_interp = PchipInterpolator(Mhalos, HMFs)

        M_lower = 10**(log10Mstar) / 0.157 / epsilon # epsilon = 0.1
        Mh_array_integrate = np.geomspace(M_lower, 1e14, 500)
        Ngal += simps(HMF_interp(Mh_array_integrate), Mh_array_integrate) * Vol
    return weight, Ngal


#######################

if __name__ == '__main__':

    for epsi in [0.1, 0.5]:
        for ampp in [0., 0.5, 1., 2., 5., 10.]:
            for meann in [-0.69, -0.3, 0., 0.5, 1., 2., 3.]:
                correct = Ngals([0., 0., 0., ampp, meann, 0., 0.1, epsi])[1]
                not_so_correct = Ngals2([0., 0., 0., ampp, meann, 0., 0.1, epsi])[1]
                print(epsi, ampp, meann, correct, not_so_correct, 100*(correct/not_so_correct-1.))

    # Poisson_epsilon1 = []
    # for filepath in glob.iglob('../Final_Poisson_2_brightest2_epsilon0p1/*__*.txt'):
    #     data = np.loadtxt(filepath)
    #     Poisson_epsilon1.append(data)
    # Poisson_epsilon1 = np.vstack(np.array(Poisson_epsilon1))

    # #######################

    # Ngals_epsilon1 = []
    # with closing(Pool(processes=None)) as pool:
    #     for numm, Ngal_epsilon1 in enumerate(pool.imap(Ngals, list(np.column_stack((Poisson_epsilon1, np.repeat(0.1, len(Poisson_epsilon1[:,0])), np.repeat(0.1, len(Poisson_epsilon1[:,0]))))))):
    #         Ngals_epsilon1.append(Ngal_epsilon1)
    #         if numm % 100 == 0:
    #             np.savetxt("Final_Ngals_epsilon0p1.txt", np.vstack(np.array(Ngals_epsilon1)))
    #     pool.terminate()

    # # for amp, mean in Poisson_epsilon1:
    # #     Ngal_epsilon1.append(Ngals(cosmo, amp, mean, 0.1, 1.))

    # np.savetxt("Final_Ngals_epsilon0p1.txt", Ngals_epsilon1)







# import numpy as np
# from scipy.interpolate import PchipInterpolator
# from scipy.integrate import simps
# from scipy.special import erf
# from classy import Class
# import glob
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from multiprocessing import Pool
# from contextlib import closing

# cosmo = Class()
# args = {'P_k_max_1/Mpc': 40., 'T_ncdm': 0.71611, 'N_ur': 2.0308, 'N_ncdm': 1, 'tau_reio': 0.0544, 'n_s': 0.9649 , 'k_pivot': 0.05, 'omega_b': 0.02236, 'm_ncdm': 0.06, 'h': 0.6727, 'z_max_pk': 10.0, 'output': 'mPk, dTk', 'omega_cdm': 0.1202, 'ln10^{10}A_s': 3.045}
# cosmo.set(args)
# cosmo.compute(['lensing'])

# JWST_Mstar = np.loadtxt("Labbe_Mstar.txt")

# # Define order of Gaussian quadrature integration
# points, weights = np.polynomial.legendre.leggauss(25)
# points_highres, weights_highres = np.polynomial.legendre.leggauss(150)

# # Halo masses which we integrate over
# Mhalos = np.geomspace(1e9, 1e14, 600)

# c = 299792.458
# rho_crit = 2.77463e11
# kappaUV = 1.15e-28
# invMpctoinvYear = 3.066067275784999e-07
# AST = 0.3222
# aST = 0.85
# pST = 0.3
# deltaST = 1.686

# # Comoving radial distance
# def rcomoving(z, Omega_m, h):
#     c * integrator(lambda x: 1/np.sqrt(Omega_m * np.power(1 + x,3) + 1. - Omega_m), 0., z) / (100. * h)

# # Gaussian integrator
# def integrator(f, a, b, highres=False):
#     sub = (b - a) / 2.
#     add = (b + a) / 2.
#     if sub == 0:
#         return 0.
#     if not highres:
#         return sub * np.dot(f(sub * points + add), weights)
#     return sub * np.dot(f(sub * points_highres + add), weights_highres)

# # The Gaussian bump added to the power spectrum
# def Gaussian_bump(k, amp, mean, sigma):
#     k1 = 0.5
#     if k < k1:
#         return 0.
#     return amp * np.exp(-(np.log(k) - mean)**2 / (2. * sigma**2))

# # Calculate the mass variance manually intead from CLASS
# def calculate_sigma(R, kmin, kmax, pk, z, amp, mean, sigma):
#     mps = np.vectorize(lambda y: pk(y, z) * (1. + Gaussian_bump(y, amp, mean, sigma)))
#     window = lambda x: 3. * (np.sin(x * R) - (x * R) * np.cos(x * R)) / (x * R)**3
#     return np.sqrt(integrator(lambda lnx: window(np.exp(lnx))**2 * mps(np.exp(lnx)) * np.exp(lnx)**3, np.log(kmin), np.log(kmax), True) / 2. / np.pi**2)

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

# def Ngals(args):

#     weight, lkl, omegab, amp, mean, Omegam, sigma, epsilon = args

#     # Iterate over redshifts of data
#     Ngal = 0
#     for z, log10Mstar in JWST_Mstar:

#         # Array of wavenumbers 
#         ks = cosmo.get_transfer(z)["k (h/Mpc)"] * 0.6727

#         # HMFs and average MUVs only need to be computed at each redshift slice
#         HMFs = np.array([HMF(cosmo, z, mass, ks, amp, mean, sigma) for mass in Mhalos])
#         HMF_interp = PchipInterpolator(Mhalos, HMFs)

#         M_lower = 10**(log10Mstar) / 0.156 / epsilon # epsilon = 0.1
#         Mh_array_integrate = np.geomspace(M_lower, 1e14, 500)
#         Ngal += simps(HMF_interp(Mh_array_integrate), Mh_array_integrate) * 1e5
#     print("[", lkl, amp, mean, "]", Ngal)
#     return weight, Ngal

# #######################

# if __name__ == '__main__':

#     Poisson_epsilon1 = []
#     for filepath in glob.iglob('../../Chains/Poisson_11_brightest2_epsilon1/*__*.txt'):
#         data = np.loadtxt(filepath)
#         Poisson_epsilon1.append(data)
#     Poisson_epsilon1 = np.vstack(np.array(Poisson_epsilon1))

#     #######################

#     Ngals_epsilon1 = []
#     with closing(Pool(processes=None)) as pool:
#         for numm, Ngal_epsilon1 in enumerate(pool.imap(Ngals, list(np.column_stack((Poisson_epsilon1, np.repeat(0.1, len(Poisson_epsilon1[:,0])), np.repeat(1., len(Poisson_epsilon1[:,0]))))))):
#             Ngals_epsilon1.append(Ngal_epsilon1)
#             # if numm % 100 == 0:
#             #     np.savetxt("Ngals_epsilon1.txt", np.vstack(np.array(Ngals_epsilon1)))
#         pool.terminate()

#     # for amp, mean in Poisson_epsilon1:
#     #     Ngal_epsilon1.append(Ngals(cosmo, amp, mean, 0.1, 1.))

#     # np.savetxt("Ngals_epsilon1.txt", Ngals_epsilon1)






# # def Ngals(amp, mean, sigma, epsilon):

# #     # Iterate over redshifts of data
# #     Ngal = 0
# #     for z, log10Mstar in JWST_Mstar:

# #         # Array of wavenumbers 
# #         ks = cosmo.get_transfer(z)["k (h/Mpc)"] * 0.6727

# #         # HMFs and average MUVs only need to be computed at each redshift slice
# #         HMFs = np.array([HMF(cosmo, z, mass, ks, amp, mean, sigma) for mass in Mhalos])
# #         HMF_interp = PchipInterpolator(Mhalos, HMFs)

# #         M_lower = 10**(log10Mstar) / 0.156 / epsilon # epsilon = 0.1
# #         Mh_array_integrate = np.geomspace(M_lower, 1e14, 500)
# #         Ngal += simps(HMF_interp(Mh_array_integrate), Mh_array_integrate) * 1e5
# #     return Ngal

# # #######################

# # amps = [5.450024721690235]
# # means = [-0.31519765293829105]

# # if __name__ == '__main__':

# #     for amp in amps:
# #         for mean in means:

# #             print(Ngals(amp, mean, 0.1, 1.))
