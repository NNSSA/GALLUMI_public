import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator
import galcv

#########################################################################################

# Gaussian integrator
points, weights = np.polynomial.legendre.leggauss(1000)
def integrator(f, a, b):
    sub = (b - a) / 2.
    add = (b + a) / 2.
    if sub == 0:
        return 0.
    return sub * np.dot(np.exp(f(sub * points + add)), weights)

# Import best-fit UV LF curves (using HST data with model1)
bestfit_data = np.loadtxt("model1_bestfit.txt")
bestfit_data = bestfit_data[bestfit_data[:,2] > 0.]
bestfit_z4 = bestfit_data[bestfit_data[:,0]==4.]
bestfit_z5 = bestfit_data[bestfit_data[:,0]==5.]
bestfit_z6 = bestfit_data[bestfit_data[:,0]==6.]
bestfit_z7 = bestfit_data[bestfit_data[:,0]==7.]
bestfit_z8 = bestfit_data[bestfit_data[:,0]==8.]
bestfit_z9 = bestfit_data[bestfit_data[:,0]==9.]
bestfit_z10 = bestfit_data[bestfit_data[:,0]==10.]

# Interpolate the best-fit curves
phi_z4_interp = PchipInterpolator(bestfit_z4[:,1], np.log(bestfit_z4[:,2]))
phi_z5_interp = PchipInterpolator(bestfit_z5[:,1], np.log(bestfit_z5[:,2]))
phi_z6_interp = PchipInterpolator(bestfit_z6[:,1], np.log(bestfit_z6[:,2]))
phi_z7_interp = PchipInterpolator(bestfit_z7[:,1], np.log(bestfit_z7[:,2]))
phi_z8_interp = PchipInterpolator(bestfit_z8[:,1], np.log(bestfit_z8[:,2]))
phi_z9_interp = PchipInterpolator(bestfit_z9[:,1], np.log(bestfit_z9[:,2]))
phi_z10_interp = PchipInterpolator(bestfit_z10[:,1], np.log(bestfit_z10[:,2]))
interps = [phi_z4_interp, phi_z5_interp, phi_z6_interp, phi_z7_interp, phi_z8_interp, phi_z9_interp, phi_z10_interp]

# Redshifts we are interested in
redshifts = [4., 5., 6., 7., 8., 9., 10.]
# Area covered by survey and corresponding fraction of sky
area = 7200000
f_sky = area / 148510660.498
# Luminosity and comoving radial distances
DLs = {4.:36828.7, 5.:47906., 6.:59264.3, 7.:70843.1, 8.:82600.8, 9.:94507.7, 10.:106541.8}
r_com = {4.:[6983.0, 7695.9], 5.:[7695.9, 8239.1], 6.:[8239.1, 8670.5], 7.:[8670.5, 9023.7], 8.:[9023.7, 9319.7], 9.:[9319.7, 9572.4]}
# Faintest apparent magnitude covered by survey
mUV_max = 26.5
# Brightest absolute magnitude covered by survey
MUV_min = -24.5
# Bin width
DeltaMUV = 0.5
# Array in which we store the mock data
for_output = []

# Iterate over redshifts
for index, z in enumerate(redshifts):

    # Volume at redshift z
    Volume = 4 * np.pi * f_sky * (r_com[z][1]**3 - r_com[z][0]**3) / 3.

    # Faintest absolute magnitude corresponding to mUV_max
    MUV_max = mUV_max - 5. * (np.log10(DLs[z] * 1e6) - 1.)

    # Define magnitude bin
    MUV_bins = np.linspace(MUV_min, MUV_max, 1+int(np.around((MUV_max-MUV_min)/DeltaMUV)))
    bin_size = np.diff(MUV_bins)[0]
    MUV_centers = MUV_bins[:-1] + bin_size/2.

    # Calculate number of galaxies
    Num_gals_int = np.array([integrator(interps[index], mag, mag+bin_size) for mag in MUV_bins[:-1]]) * Volume
    # Sample from a Poisson distribution using the previous computation
    Num_gals = np.random.poisson(Num_gals_int)
    # Compute the UV LF mock data
    Phi = Num_gals / Volume / bin_size 
    
    # Compute Poisson error
    Poisson_error = np.sqrt(Num_gals) / Volume / bin_size
    
    # Compute cosmic variance using galcv code
    cosmic_variance = np.array(galcv.getcv(mag=MUV_centers, area=31600, z=max(5., z), zW=1., appOrAbs="absolute", interpWarning=0)) * Phi
    if any(Num_gals):
        cosmic_variance[np.argwhere(np.isnan(cosmic_variance))] = cosmic_variance[np.argwhere(np.isfinite(cosmic_variance))][0]

    # Add two errors together and impose 5% minimal error
    Phi_error = np.sqrt(Poisson_error**2 + cosmic_variance**2)
    Phi_error = np.array(list(map(max,zip(0.05 * Phi, Phi_error))))

    # Store UV LF mock data in for_output array
    for index2 in range(len(MUV_centers)):
        if Phi[index2] > 0.:
            for_output.append([z, MUV_centers[index2], bin_size, Phi[index2], Phi_error[index2]])

# Save UV LF mock data
for_output = np.array(for_output)
np.savetxt("UVLF_FutureMock_WideField.txt", for_output)