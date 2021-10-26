import numpy as np
import h5py
import galcv

data = {}
models = ["A", "B", "C"]
simulations = ["TNG100-1", "TNG300-1", "TNG50-1", "combined"]
redshifts = {33:2, 25:3, 21:4, 17:5, 13:6, 11:7, 8:8, 6:9, 4:10}
quantities = ["bincenters", "luminosity_function", "number_count", "lf_combined"]

## Extract data and store in dictionary "data"
with h5py.File("UVLF_TNG_MV2019.hdf5", "r") as f:
    for model in list(f.keys()):
        data[model] = {}
        for simu in f[model]:
            data[model][simu] = {}
            for redshift in f[model][simu]:
                redshift_translated = redshifts[float(redshift)]
                data[model][simu][redshift_translated] = {}
                for quantity in f[model][simu][redshift]:
                    data[model][simu][redshift_translated][quantity] = f[model][simu][redshift][quantity][()]

## This function calls the data
def sim(model, simulation, redshift, quantity):
    return data[model][simulation][redshift][quantity]

## Total magnitude range
magnitudes = sim("A", "TNG300-1", 4, "bincenters")
bin_width = np.diff(magnitudes)[-1]

## Compute volume (Mpc^3 x mag_bin_size) of each simulation
vol_300 = (sim("A", "TNG300-1", 4, "number_count") / 10**sim("A", "TNG300-1", 4, "luminosity_function"))[-1]
vol_100 = (sim("A", "TNG100-1", 4, "number_count") / 10**sim("A", "TNG100-1", 4, "luminosity_function"))[-1]
vol_50 = (sim("A", "TNG50-1", 4, "number_count") / 10**sim("A", "TNG50-1", 4, "luminosity_function"))[-1]

## Extract number counts or LF of each simulation
def data_slices(model, redshift, number=True, individual=True):
    if individual:
        if number:
            return sim(model, "TNG300-1", redshift, "number_count"), sim(model, "TNG100-1", redshift, "number_count"), sim(model, "TNG50-1", redshift, "number_count")
        return sim(model, "TNG300-1", redshift, "number_count") / vol_300, sim(model, "TNG100-1", redshift, "number_count") / vol_100, sim(model, "TNG50-1", redshift, "number_count") / vol_50
    if number:
        return np.concatenate((
            sim(model, "TNG300-1", redshift, "number_count"),
            sim(model, "TNG100-1", redshift, "number_count"),
            sim(model, "TNG50-1", redshift, "number_count")
            ))
    return np.concatenate((
            sim(model, "TNG300-1", redshift, "number_count") / vol_300,
            sim(model, "TNG100-1", redshift, "number_count") / vol_100,
            sim(model, "TNG50-1", redshift, "number_count") / vol_50
            ))

## Integrating tools
# Define order of Gaussian quadrature integration
points, weights = np.polynomial.legendre.leggauss(50)

# Gaussian quadrature integrator
def integrator(f, a, b):

    sub = (b - a) / 2.
    add = (b + a) / 2.

    if sub == 0:
        return 0.

    return sub * np.dot(f(sub * points + add), weights)

## Comoving Angular diameter distance
def D_A(z, Omega_m=0.3089, h=0.6774):
    return integrator(lambda x: 1/np.sqrt(Omega_m * np.power(1 + x,3) + 1. - Omega_m), 0., z) * 299792.458 / (100. * h)

## Redshift bin width of each simulation box
def delta_z(z, Lbox, Omega_m=0.3089, h=0.6774):
    return Lbox * 100. * h * np.sqrt(Omega_m * np.power(1 + z,3) + 1. - Omega_m) / 299792.458

## Compute cosmic variance using the galcv code
def cosmic_variance(model, redshift):
    
    # These are the LFs from each simulation
    lf300, lf100, lf50 = data_slices(model,redshift,number=False)
    
    # Compute the effective areas in arcmin^2
    areas = [(Lbox/D_A(redshift))**2 * (180*60/np.pi)**2 for Lbox in [302.627694125,110.71744907,51.6681428993]]

    # galcv can't compute the CV at z = 4, so we use the CV at z = 5 for it (which is a conservative approach)
    redshift = max(5, redshift)

    # Compute cosmic variance with galcv
    cv_300 = np.array(galcv.getcv(mag=magnitudes, area=areas[0], z=redshift, zW=max(0.1, delta_z(redshift, 302.627694125)), appOrAbs="absolute", interpWarning=0))
    cv_100 = np.array(galcv.getcv(mag=magnitudes, area=areas[1], z=redshift, zW=max(0.1, delta_z(redshift, 110.71744907)), appOrAbs="absolute", interpWarning=0))
    cv_50 = np.array(galcv.getcv(mag=magnitudes, area=areas[2], z=redshift, zW=max(0.1, delta_z(redshift, 51.6681428993)), appOrAbs="absolute", interpWarning=0))

    # In some cases galcv gives nan, e.g., if magnitudes are too bright. Therefore we use the largest error there (which is dominated by poisson error anyway)
    cv_300[np.isnan(cv_300)] = max(cv_300[np.isfinite(cv_300)])
    cv_100[np.isnan(cv_100)] = max(cv_100[np.isfinite(cv_100)])
    cv_50[np.isnan(cv_50)] = max(cv_50[np.isfinite(cv_50)])

    # Minimal error in cosmic variance
    minimal = 0.05

    err_300 = np.array(list(map(max,zip(np.repeat(minimal, len(cv_300)), cv_300))))
    err_100 = np.array(list(map(max,zip(np.repeat(minimal, len(cv_100)), cv_100))))
    err_50 = np.array(list(map(max,zip(np.repeat(minimal, len(cv_50)), cv_50))))

    return err_300 * lf300, err_100 * lf100, err_50 * lf50

## Compute Poisson error
def Poisson_error(model, redshift):

    # These are the number of galaxies from each simulation
    num300, num100, num50 = data_slices(model,redshift)

    return np.sqrt(num300)/vol_300, np.sqrt(num100)/vol_100, np.sqrt(num50)/vol_50

## Return combined LF
def data_combined(model, redshift, original=False):

    # Return the original, combined UVLF from the raw data 
    if original:
        return 10**sim(model, "combined", redshift, "lf_combined")
    
    # Positions of where we want to transition from one simulation to the other - pos300 or pos100 corresponds...
    # ...to the magnitude M_i where the number count of galaxies in the simulation peaks at M_i+2
    pos300 = np.argmax(sim(model, "TNG300-1", redshift, "number_count")) - 1
    pos100 = np.argmax(sim(model, "TNG100-1", redshift, "number_count")) - 1

    # These are the LFs from each simulation
    lf_300, lf_100, lf_50 = data_slices(model, redshift, number=False, individual=True)

    # set LFs equal to 0 where simulation statistics become relevant (at peak of number of galaxies)
    lf_300[pos300:] = 0.
    lf_100[pos100:] = 0.

    # These are the Poisson errors
    poisson_error_300, poisson_error_100, poisson_error_50 = Poisson_error(model, redshift)

    # These are the cosmic variances
    cv_error_300, cv_error_100, cv_error_50 = cosmic_variance(model,redshift)

    # Combine Poisson error and cosmic variance
    error_300 = np.sqrt(poisson_error_300**2 + cv_error_300**2)
    error_100 = np.sqrt(poisson_error_100**2 + cv_error_100**2)
    error_50 = np.sqrt(poisson_error_50**2 + cv_error_50**2)

    # Add a minimal error of 20% in each simulation data to account for simulation statistics
    min_error = 0.2
    error_300 = np.array(list(map(max,zip(min_error * lf_300, error_300))))
    error_100 = np.array(list(map(max,zip(min_error * lf_100, error_100))))
    error_50 = np.array(list(map(max,zip(min_error * lf_50, error_50))))

    error_300[pos300:] = 0.
    error_100[pos100:] = 0.

    # Compute inverse errors
    inv_error_300 = 1/error_300
    inv_error_100 = 1/error_100
    inv_error_50 = 1/error_50

    # Set inverse error equal to 0 where it's infinite 
    inv_error_300[np.isinf(inv_error_300)] = 0.
    inv_error_100[np.isinf(inv_error_100)] = 0.
    inv_error_50[np.isinf(inv_error_50)] = 0.

    error_tot = 1/np.sqrt(inv_error_300**2 + inv_error_100**2 + inv_error_50**2)
    error_tot[np.isinf(error_tot)] = 0.

    # inverse variance method
    lf_tot = error_tot * np.sqrt((lf_300 * inv_error_300)**2 + (lf_100 * inv_error_100)**2 + (lf_50 * inv_error_50)**2)

    return lf_tot, error_tot

## Construct array for output
for_output = []
for z in [4,5,6,7,8,9,10]:
    if z < 6.5:
        MUV_cutoff = -16. 
    elif z < 8.5:
        MUV_cutoff = -16.5
    else:
        MUV_cutoff = -16.75
    lfs, errors = data_combined("A",z)
    for num, LF in enumerate(zip(lfs, errors)):
        if np.isfinite(LF[0]) and magnitudes[num] <= MUV_cutoff and LF[0] != 0.:
            for_output.append((z, magnitudes[num], bin_width, LF[0], LF[1]))

np.savetxt("UVLF_IllustrisTNG.txt", np.array(for_output))
