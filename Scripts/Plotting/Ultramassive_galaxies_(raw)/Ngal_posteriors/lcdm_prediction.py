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

# Calculate the mass variance manually intead from CLASS
def calculate_sigma(R, kmin, kmax, pk, z):
    mps = np.vectorize(lambda y: pk(y, z))
    window = lambda x: 3. * (np.sin(x * R) - (x * R) * np.cos(x * R)) / (x * R)**3
    return np.sqrt(integrator(lambda lnx: window(np.exp(lnx))**2 * mps(np.exp(lnx)) * np.exp(lnx)**3, np.log(kmin), np.log(kmax), True) / 2. / np.pi**2)

# Compute the HMF and average <MUV> given halo mass Mh
def HMF(cosmo, z, Mh, ks):
    rhoM = np.power(cosmo.h(), 2) * cosmo.Omega_m() * rho_crit
    deltaM = 100

    sigma = calculate_sigma(np.power(3. * Mh / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z)
    sigma_plus_deltaM = calculate_sigma(np.power(3. * (Mh+deltaM) / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z)
    sigma_minus_deltaM = calculate_sigma(np.power(3. * (Mh-deltaM) / (4. * np.pi * rhoM), 1./3), min(ks), max(ks), cosmo.pk_cb_lin, z)
    dsigmadM = (sigma_plus_deltaM - sigma_minus_deltaM) / (2. * deltaM)

    return -AST * np.sqrt(2. * aST / np.pi) * (1. + np.power(np.power(sigma,2) / (aST * np.power(deltaST, 2)), pST)) * (deltaST / sigma) * np.exp(-aST * np.power(deltaST, 2) / (2. * np.power(sigma, 2))) * (rhoM / (Mh * sigma)) * dsigmadM

def Ngals(Mcut):

    z = 7.

    # Array of wavenumbers 
    ks = cosmo.get_transfer(z)["k (h/Mpc)"] * 0.6727

    # HMFs and average MUVs only need to be computed at each redshift slice
    HMFs = np.array([HMF(cosmo, z, mass, ks) for mass in Mhalos])
    HMF_interp = PchipInterpolator(Mhalos, HMFs)

    M_lower = Mcut
    Mh_array_integrate = np.geomspace(M_lower, 1e14, 500)
    Ngal = simps(HMF_interp(Mh_array_integrate), Mh_array_integrate) * 1e5

    print(Ngal)

    return Mcut, Ngal

#######################

if __name__ == '__main__':

    Mcut_array = np.geomspace(1e9, 1e14, 200)

    Ngals_lcdm = []
    with closing(Pool(processes=None)) as pool:
        for Ngal_lcdm in pool.imap(Ngals, Mcut_array):
            Ngals_lcdm.append(Ngal_lcdm)
        pool.terminate()

    np.savetxt("Ngals_lcdm.txt", np.vstack(np.array(Ngals_lcdm)))