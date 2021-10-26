import numpy as np
from matplotlib import pyplot as plt
import glob
from matplotlib import patches as mpatches
import scipy.ndimage
from scipy.interpolate import PchipInterpolator

plt.style.use("../template.mplstyle")
# purple - green - darkgoldenrod - blue - red
colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145']
linestyles = [(0, (1,1.05)), (0, (3, 1, 1, 1)), (0, (1,3)), (0, (3,3.65)), (0, (3,2.772)), (0, (3, 1, 1, 1, 1, 1))]

#########################################################################################

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

def ctr_level2d(histogram2d, lvl, infinite=False):
    hist = histogram2d.flatten()*1.
    hist.sort()
    cum_hist = np.cumsum(hist[::-1])
    cum_hist /= cum_hist[-1]

    alvl = np.searchsorted(cum_hist, lvl)[::-1]
    clist = [0]+[hist[-i] for i in alvl]+[hist.max()]
    if not infinite:
        return clist[1:]
    return clist

def get_hist2d(datax, datay, num_bins=40, weights=[None]):
    if not any(weights):
        weights = np.ones(len(datax))
    hist, bin_edgesx, bin_edgesy = np.histogram2d(datax, datay, bins=num_bins, weights=weights)
    bin_centresx = 0.5*(bin_edgesx[1:]+bin_edgesx[:-1])
    bin_centresy = 0.5*(bin_edgesy[1:]+bin_edgesy[:-1])
    return hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def plot_hist2d(datax, datay, ax, num_bins=30, weights=[None], color=None, zorder=0):
    if not any(weights):
        weights = np.ones(len(datax))
    if color == None:
        color="black"

    hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy = get_hist2d(datax, datay, num_bins=num_bins, weights=weights)

    interpolation_smoothing = 3.
    gaussian_smoothing = 0.5
    sigma = interpolation_smoothing * gaussian_smoothing

    interp_y_centers = scipy.ndimage.zoom(bin_centresy, interpolation_smoothing, mode='reflect')
    interp_x_centers = scipy.ndimage.zoom(bin_centresx,interpolation_smoothing, mode='reflect')
    interp_hist = scipy.ndimage.zoom(hist, interpolation_smoothing, mode='reflect')
    interp_smoothed_hist = scipy.ndimage.filters.gaussian_filter(interp_hist, [sigma,sigma], mode='reflect')

    ax.contourf(interp_x_centers, interp_y_centers, np.transpose(interp_smoothed_hist), colors=[adjust_lightness(color,1.4), adjust_lightness(color,0.8)], levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]), zorder=zorder, alpha=0.55)
    ax.contour(interp_x_centers, interp_y_centers, np.transpose(interp_smoothed_hist), colors=[color, adjust_lightness(color,0.8)], linewidths=2., levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]), zorder=zorder)


##################################################################################################

UVLF_fiducial = []
UVLF_conservative = []

for filepath in glob.iglob('../../Data/UVLF_HST_ST_model1/*__*.txt'):
    data = np.loadtxt(filepath)
    UVLF_fiducial.append(data)
for filepath in glob.iglob('../../Data/UVLF_HST_ST_model1_conservative/*__*.txt'):
    data = np.loadtxt(filepath)
    UVLF_conservative.append(data)

UVLF_fiducial = np.vstack(np.array(UVLF_fiducial))
UVLF_conservative = np.vstack(np.array(UVLF_conservative))

plt.figure(figsize=(17.,6.5))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.tick_params(axis='x', which='major', pad=6)
ax2.tick_params(axis='x', which='major', pad=6)
ax1.tick_params(axis='both', which='major', labelsize=25)
ax1.tick_params(axis='both', which='minor', labelsize=25)
ax2.tick_params(axis='both', which='major', labelsize=25)
ax2.tick_params(axis='both', which='minor', labelsize=25)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(2.2)
    ax2.spines[axis].set_linewidth(2.2)

###############

plot_hist2d(datax=UVLF_fiducial[:,-7], datay=UVLF_fiducial[:,2], ax=ax1, num_bins=20, weights=UVLF_fiducial[:,0], color=colors[3], zorder=2)
plot_hist2d(datax=UVLF_conservative[:,-7], datay=UVLF_conservative[:,2], ax=ax1, num_bins=20, weights=UVLF_conservative[:,0], color=colors[-1], zorder=1)
ax1.set_xlabel(r'$\Omega_\mathrm{m}$', labelpad=10, fontsize=28)
ax1.set_ylabel(r'$\sigma_8$', labelpad=12, fontsize=28)
ax1.set_xlim(0.2, 0.4)
ax1.set_ylim(0.3, 1.3)

patch_blue = mpatches.Patch(color=colors[3], lw=1.5, label=r"$\mathrm{Fiducial}$")
patch_red = mpatches.Patch(color=colors[-1], lw=1.5, label=r"$\mathrm{Conservative}$")
leg = ax1.legend(handles=[patch_red, patch_blue], loc="upper left", frameon=False, markerfirst=True, prop={'size': 18}, handlelength=1.9, handletextpad=0.5, numpoints=1)

###############

plot_hist2d(datax=UVLF_fiducial[:,5], datay=UVLF_fiducial[:,2], ax=ax2, num_bins=20, weights=UVLF_fiducial[:,0], color=colors[3], zorder=2)
plot_hist2d(datax=UVLF_conservative[:,5], datay=UVLF_conservative[:,2], ax=ax2, num_bins=20, weights=UVLF_conservative[:,0], color=colors[-1], zorder=1)
ax2.set_ylabel(r'$\sigma_8$', labelpad=12, fontsize=28)
ax2.set_xlabel(r'$n_\mathrm{s}$', labelpad=10, fontsize=28)
ax2.set_xlim(0.7, 1.3)
ax2.set_ylim(0.3, 1.3)

plt.savefig("Posteriors_cosmo_model1_conservative.pdf")

data_for_lims = UVLF_conservative[:,2]
hist, bin_edges, bin_centres = get_hist(data_for_lims, num_bins=20, weights=UVLF_conservative[:,0])
xarray = np.linspace(min(bin_centres), max(bin_centres), 1000)
interpolator = PchipInterpolator(bin_centres, hist)(xarray)
levels = ctr_level(interpolator.copy(), [0.68])
pos = [np.searchsorted(interpolator[:np.argmax(interpolator)], levels)[0]-1, -np.searchsorted((interpolator[::-1])[:np.argmax(interpolator[::-1])], levels)[0]]

print("mean: ", np.mean(data_for_lims))
print("mean - lower (68%): ", np.mean(data_for_lims) - xarray[pos[0]])
print("upper - mean (68%): ", xarray[pos[1]] - np.mean(data_for_lims))
