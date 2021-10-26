import numpy as np
from matplotlib import pyplot as plt
import glob
from scipy.interpolate import PchipInterpolator
plt.style.use("../template.mplstyle")

# purple - green - darkgoldenrod - blue - red
colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145']

#################################################################

def ctr_level(hist, lvl, infinite=False):
    hist.sort()
    cum_hist = np.cumsum(hist[::-1])
    cum_hist = cum_hist / cum_hist[-1]

    alvl = np.searchsorted(cum_hist, lvl)
    clist = [0]+[hist[-i] for i in alvl]
    if not infinite:
        return clist[1:]
    return clist

def get_hist(data, num_bins=40, weights=[None]):
    if not any(weights):
        weights = np.ones(len(data))
    hist, bin_edges = np.histogram(data, bins=num_bins, weights=weights)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    return hist, bin_edges, bin_centres

def plot_hist(data, ax, num_bins=30, weights=[None], color=None):
    if not any(weights):
        weights = np.ones(len(data))
    if color == None:
        color="darkblue"

    hist, bin_edges, bin_centres = get_hist(data, num_bins=num_bins, weights=weights)
    ax.plot(bin_centres, hist/max(hist), color=color, lw=2)
    ax.step(bin_centres, hist/max(hist), where='mid', color=color)
    pkarray = np.linspace(min(bin_centres), max(bin_centres), 1000)
    interpolator = PchipInterpolator(bin_centres, hist)(pkarray)
    ax.plot(pkarray, interpolator/max(interpolator), color="red", lw=2)

#################################################################

UVLF_MPS = []

for filepath in glob.iglob('../../Data/UVLF_HST_ST_model1_powerspectrum/*__*.txt'):
    data = np.loadtxt(filepath)
    UVLF_MPS.append(data)

UVLF_MPS = np.vstack(np.array(UVLF_MPS))

names = [r'$P(k=1.06\,\mathrm{Mpc}^{-1})$', r'$P(k=1.5\,\mathrm{Mpc}^{-1})$', r'$P(k=2\,\mathrm{Mpc}^{-1})$', r'$P(k=4.7\,\mathrm{Mpc}^{-1})$', r'$P(k=6\,\mathrm{Mpc}^{-1})$', r'$P(k=8\,\mathrm{Mpc}^{-1})$']

fig = plt.figure(figsize=(20,12))
ax1 = plt.subplot(231)
ax2 = plt.subplot(232)
ax3 = plt.subplot(233)
ax4 = plt.subplot(234)
ax5 = plt.subplot(235)
ax6 = plt.subplot(236)
plot_hist(UVLF_MPS[:,-1]*10**UVLF_MPS[:,17], ax1, 30, UVLF_MPS[:,0], "black")
plot_hist(UVLF_MPS[:,-2]*10**UVLF_MPS[:,17], ax2, 30, UVLF_MPS[:,0], "black")
plot_hist(UVLF_MPS[:,-3]*10**UVLF_MPS[:,17], ax3, 30, UVLF_MPS[:,0], "black")
plot_hist(UVLF_MPS[:,-4]*10**UVLF_MPS[:,16], ax4, 30, UVLF_MPS[:,0], "black")
plot_hist(UVLF_MPS[:,-5]*10**UVLF_MPS[:,16], ax5, 30, UVLF_MPS[:,0], "black")
plot_hist(UVLF_MPS[:,-6]*10**UVLF_MPS[:,16], ax6, 30, UVLF_MPS[:,0], "black")

ax1.set_xlabel(names[-1])
ax2.set_xlabel(names[-2])
ax3.set_xlabel(names[-3])
ax4.set_xlabel(names[-4])
ax5.set_xlabel(names[-5])
ax6.set_xlabel(names[-6])

data_for_lims = UVLF_MPS[:,-6] * 10**UVLF_MPS[:,16]
hist, bin_edges, bin_centres = get_hist(data_for_lims, num_bins=30, weights=UVLF_MPS[:,0])
pkarray = np.linspace(min(bin_centres), max(bin_centres), 1000)
interpolator = PchipInterpolator(bin_centres, hist)(pkarray)
levels = ctr_level(interpolator.copy(), [0.68])
pos = [np.searchsorted(interpolator[:np.argmax(interpolator)], levels)[0]-1, -np.searchsorted((interpolator[::-1])[:np.argmax(interpolator[::-1])], levels)[0]]
ax6.axvline(pkarray[pos[0]], linestyle="dashed", color="black", alpha=0.6)
ax6.axvline(pkarray[pos[1]], linestyle="dashed", color="black", alpha=0.6)

print("bin1 frequent: ", pkarray[np.argmax(interpolator)])
print("bin1 mean - lower (68%): ", pkarray[np.argmax(interpolator)] - pkarray[pos[0]])
print("bin1 upper - mean (68%): ", pkarray[pos[1]] - pkarray[np.argmax(interpolator)])

data_for_lims = UVLF_MPS[:,-3] * 10**UVLF_MPS[:,17]
hist, bin_edges, bin_centres = get_hist(data_for_lims, num_bins=30, weights=UVLF_MPS[:,0])
pkarray = np.linspace(min(bin_centres), max(bin_centres), 1000)
interpolator = PchipInterpolator(bin_centres, hist)(pkarray)
levels = ctr_level(interpolator.copy(), [0.68])
pos = [np.searchsorted(interpolator[:np.argmax(interpolator)], levels)[0]-1, -np.searchsorted((interpolator[::-1])[:np.argmax(interpolator[::-1])], levels)[0]]
ax3.axvline(pkarray[pos[0]], linestyle="dashed", color="black", alpha=0.6)
ax3.axvline(pkarray[pos[1]], linestyle="dashed", color="black", alpha=0.6)

print("bin2 frequent: ", pkarray[np.argmax(interpolator)])
print("bin2 mean - lower (68%): ", pkarray[np.argmax(interpolator)] - pkarray[pos[0]])
print("bin2 upper - mean (68%): ", pkarray[pos[1]] - pkarray[np.argmax(interpolator)])

plt.savefig("Power_spectrum_values.pdf")
