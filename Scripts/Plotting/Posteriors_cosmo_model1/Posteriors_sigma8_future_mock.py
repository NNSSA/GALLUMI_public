import numpy as np
from matplotlib import pyplot as plt
import glob
from scipy.interpolate import PchipInterpolator
plt.style.use("../template.mplstyle")

# purple - green - darkgoldenrod - blue - red
colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145']

############################################################

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

############################################################

UVLF_HST = []
UVLF_mock_deep = []
UVLF_mock_wide = []

for filepath in glob.iglob('../../Data/UVLF_HST_ST_model1/*__*.txt'):
    data = np.loadtxt(filepath)
    UVLF_HST.append(data)
for filepath in glob.iglob('../../Data/UVLF_FutureMock_ST_model1_DeepField/*__*.txt'):
    data = np.loadtxt(filepath)
    UVLF_mock_deep.append(data)
for filepath in glob.iglob('../../Data/UVLF_FutureMock_ST_model1_WideField/*__*.txt'):
    data = np.loadtxt(filepath)
    UVLF_mock_wide.append(data)

UVLF_HST = np.vstack(np.array(UVLF_HST))
UVLF_mock_deep = np.vstack(np.array(UVLF_mock_deep))
UVLF_mock_wide = np.vstack(np.array(UVLF_mock_wide))

sigma8fid = 7.6139e-01
Omegamfid = 2.9848e-01
nsfid = 9.949533e-01

fig = plt.figure(figsize=(8.5,6.5))
ax1 = plt.subplot(111)
ax1.tick_params(axis='x', which='major', pad=6)
ax1.tick_params(axis='both', which='both', labelsize=24)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(1.9)

def func(data):
    data_for_lims = data[:,2]
    hist, bin_edges, bin_centres = get_hist(data_for_lims, num_bins=16, weights=data[:,0])
    xarray = np.linspace(min(bin_centres), max(bin_centres), 1000)
    interpolator = PchipInterpolator(bin_centres, hist)(xarray)
    levels = ctr_level(interpolator.copy(), [0.68])
    pos = [np.searchsorted(interpolator[:np.argmax(interpolator)], levels)[0]-1, -np.searchsorted((interpolator[::-1])[:np.argmax(interpolator[::-1])], levels)[0]]    

    print("mean: ", np.mean(data_for_lims))
    print("mean - lower (68%): ", np.mean(data_for_lims) - xarray[pos[0]])
    print("upper - mean (68%): ", xarray[pos[1]] - np.mean(data_for_lims))

    return xarray, interpolator, pos

xarray, interpolator, pos = func(UVLF_mock_wide)
xarray = np.concatenate((np.array([0.5]), xarray))
interpolator = np.concatenate((interpolator[:1]/10, interpolator))
ax1.plot(xarray, interpolator/max(interpolator), color=colors[2], lw=2.3, ls=(0, (1,1)), zorder=2, label=r"$\mathrm{Mock\ Wide\ Field}$")
ax1.fill_between(xarray[pos[0]:pos[1]+1],np.repeat(0., len(interpolator[pos[0]:pos[1]+1])), interpolator[pos[0]:pos[1]+1]/max(interpolator[pos[0]:pos[1]+1]), facecolor="none", color=colors[2], linewidth=0.0, zorder=2, alpha=0.2)

xarray, interpolator, pos = func(UVLF_mock_deep)
xarray = np.concatenate((np.array([0.429]), xarray))
interpolator = np.concatenate((interpolator[:1]/1000, interpolator))
ax1.plot(xarray, interpolator/max(interpolator), color=colors[1], lw=2.3, ls=(0, (3,2)), zorder=1, label=r"$\mathrm{Mock\ Deep\ Field}$")
ax1.fill_between(xarray[pos[0]:pos[1]+1],np.repeat(0., len(interpolator[pos[0]:pos[1]+1])), interpolator[pos[0]:pos[1]+1]/max(interpolator[pos[0]:pos[1]+1]), facecolor="none", color=colors[1], linewidth=0.0, zorder=1, alpha=0.2)

xarray, interpolator, pos = func(UVLF_HST)
xarray = np.concatenate((np.array([0.435, 0.452]), xarray))
interpolator = np.concatenate((np.array([interpolator[0]/1000, interpolator[1]/2.]), interpolator))
ax1.plot(xarray, interpolator/max(interpolator), color=colors[3], lw=2.3, ls="solid", zorder=0, label=r"$\mathrm{HST}$")
ax1.fill_between(xarray[pos[0]:pos[1]+1],np.repeat(0., len(interpolator[pos[0]:pos[1]+1])), interpolator[pos[0]:pos[1]+1]/max(interpolator[pos[0]:pos[1]+1]), facecolor="none", color=colors[3], linewidth=0.0, zorder=0, alpha=0.2)

ax1.set_xlabel(r'$\sigma_8$', labelpad=5, fontsize=26)
ax1.set_ylabel(r'$\mathrm{Posterior\ Probability}$', labelpad=12, fontsize=26)
ax1.set_xlim(0.2, 1.2)
ax1.axes.yaxis.set_ticklabels([])
ax1.axes.yaxis.set_ticks([])

ax1.legend(loc="upper left", frameon=False, markerfirst=True, prop={'size': 17}, handlelength=1.7, handletextpad=0.8, numpoints=1)

plt.savefig("Posteriors_sigma8_future_mock.pdf")