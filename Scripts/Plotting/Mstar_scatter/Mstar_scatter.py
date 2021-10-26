import glob
import numpy as np
from scipy.interpolate import PchipInterpolator
from matplotlib import pyplot as plt
plt.style.use("../template.mplstyle")

# purple - green - darkgoldenrod - blue - red
colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145']
linestyles = [(0, (1,1.05)), (0, (3, 1, 1, 1)), (0, (1,3)), (0, (3,3.65)), (0, (3,2.772)), (0, (3, 1, 1, 1, 1, 1))]

#########################################################################################

def get_hist(data, num_bins=40, weights=[None]):
    if not any(weights):
        weights = np.ones(len(data))
    hist, bin_edges = np.histogram(data, bins=num_bins, weights=weights)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    return hist, bin_edges, bin_centres

###

TNG_z4 = np.loadtxt("IllustrisTNG_z4.txt", unpack=True)
TNG_z5 = np.loadtxt("IllustrisTNG_z5.txt", unpack=True)
TNG_z6 = np.loadtxt("IllustrisTNG_z6.txt", unpack=True)

chains = []

for filepath in glob.iglob('../../Data/UVLF_HST_ST_model2/*__*.txt'):
    data = np.loadtxt(filepath)
    chains.append(data)

chains = np.vstack(np.array(chains))

data_for_lims = chains[:,12]
hist, bin_edges, bin_centres = get_hist(data_for_lims, num_bins=20, weights=chains[:,0])
xarray = np.linspace(min(bin_centres), max(bin_centres), 1000)
interpolator = PchipInterpolator(bin_centres, hist)(xarray)

A = np.cumsum(interpolator)/np.sum(interpolator)
bound68 = xarray[np.argmin(np.abs(A-0.68))] * 0.4
bound95 = xarray[np.argmin(np.abs(A-0.95))] * 0.4

plt.figure(figsize=(8.,6.5))
ax = plt.subplot(111)
ax.tick_params(axis='x', which='major', pad=6)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.tick_params(axis='both', which='minor', labelsize=25)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.2)

ax.axhline(bound68, color="black", lw=2.5, alpha=0.7)
ax.axhline(bound95, color="black", lw=2.5, alpha=0.7)
ax.plot(TNG_z4[0], TNG_z4[1], color=colors[1], ls=(0,(3,2,1.3,2)), lw=2.5, label=r"$z = 4$")
ax.plot(TNG_z5[0], TNG_z5[1], color=colors[-1], ls=(0,(3,1)), lw=2.5, label=r"$z = 5$")
ax.plot(TNG_z6[0], TNG_z6[1], color=colors[3], ls=(0,(1,1)), lw=2.5, label=r"$z = 6$")

ax.text(-17.95, 0.307, r'$\mathrm{HST\ 68\%\ CL}$', weight='bold', fontsize=20, color="black", alpha=0.75)
ax.text(-19.45, 0.391, r'$\mathrm{HST\ 95\%\ CL\ (This\ Work)}$', weight='bold', fontsize=20, color="black", alpha=0.75)

plt.xlabel(r"$M_\mathrm{UV}\ [\mathrm{mag}]$", labelpad=9, fontsize=27)
plt.ylabel(r"$\sigma_{\log_{10}(M_*)}$", labelpad=12, fontsize=29)

plt.axis(xmin=-22.8, xmax=-16.2, ymin=0., ymax=0.5)

leg = plt.legend(loc="lower left", frameon=False, markerfirst=True, prop={'size': 22}, handlelength=1.58, handletextpad=0.5, numpoints=1)

plt.savefig("Mstar_scatter.pdf")