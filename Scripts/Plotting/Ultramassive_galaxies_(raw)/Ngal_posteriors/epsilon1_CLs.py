import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import glob
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patches as mpatches
from scipy.interpolate import PchipInterpolator
import scipy.ndimage
from scipy.integrate import simps
plt.style.use("scientific")
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
# purple - green - darkgoldenrod - blue - red
colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145', "#cf630a"]
linestyles = [(0, (1,1.05)), (0, (3, 1, 1, 1)), (0, (1,3)), (0, (3,3.65)), (0, (3,2.772)), (0, (3, 1, 1, 1, 1, 1))]

############################## Functions ##############################

def ctr_level(hist, lvl, infinite=False):
    hist.sort()
    cum_hist = np.cumsum(hist[::-1])
    cum_hist = cum_hist / cum_hist[-1]

    alvl = np.searchsorted(cum_hist, lvl)
    clist = [0]+[hist[-i] for i in alvl]
    if not infinite:
        return clist[1:]
    return clist

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

###

def get_hist(data, num_bins=40, weights=[None]):
    if not any(weights):
        weights = np.ones(len(data))
    hist, bin_edges = np.histogram(data, bins=num_bins, weights=weights)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    return hist, bin_edges, bin_centres

def get_hist2d(datax, datay, num_bins=40, weights=[None]):
    if not any(weights):
        weights = np.ones(len(datax))
    hist, bin_edgesx, bin_edgesy = np.histogram2d(datax, datay, bins=num_bins, weights=weights)
    bin_centresx = 0.5*(bin_edgesx[1:]+bin_edgesx[:-1])
    bin_centresy = 0.5*(bin_edgesy[1:]+bin_edgesy[:-1])

    return hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy

###

def confidence(data, num_bins=30, weights=[None]):
    hist, bin_edges, bin_centres = get_hist(data, num_bins=num_bins, weights=weights)
    levels = ctr_level(hist.copy(), [0.68]) #[0.68, 0.95])
    pos = [np.searchsorted(hist, levels)[0], -np.searchsorted(hist[::-1], levels)[0]]
    return bin_centres[pos]

###

# def plot_hist(data, ax, num_bins=30, weights=[None], color=None, label=None, ls=None):
#     if not any(weights):
#         weights = np.ones(len(data))
#     if color == None:
#         color="darkblue"
#     if ls == None:
#         ls == "solid"

#     hist, bin_edges, bin_centres = get_hist(np.log(data), num_bins=num_bins, weights=weights)
#     ax.plot(np.exp(bin_centres), hist/max(hist), color=color, lw=2, label=label, ls=ls)
#     # ax.step(np.exp(bin_centres), hist/max(hist), where='mid', color=color)

def plot_hist(data, ax, num_bins=30, weights=[None], color=None, label=None, ls=None):
    if not any(weights):
        weights = np.ones(len(data))
    if color == None:
        color="darkblue"
    if ls == None:
        ls == "solid"

    hist, bin_edges, bin_centres = get_hist(data, num_bins=num_bins, weights=weights)
    ax.plot(bin_centres, hist/max(hist), color=color, lw=2, label=label, ls=ls)
    # ax.step(np.exp(bin_centres), hist/max(hist), where='mid', color=color)

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

    # hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy = get_hist2d(datax, datay, num_bins=num_bins*2, weights=weights)
    # plt.hist2d(datax, datay, bins=[bin_edgesx, bin_edgesy], weights=weights, cmap="Greys")

    hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy = get_hist2d(datax, datay, num_bins=num_bins, weights=weights)

    interpolation_smoothing = 3.
    gaussian_smoothing = 0.5
    sigma = interpolation_smoothing * gaussian_smoothing

    interp_y_centers = scipy.ndimage.zoom(bin_centresy, interpolation_smoothing, mode='reflect')
    interp_x_centers = scipy.ndimage.zoom(bin_centresx,interpolation_smoothing, mode='reflect')
    interp_hist = scipy.ndimage.zoom(hist, interpolation_smoothing, mode='reflect')
    interp_smoothed_hist = scipy.ndimage.filters.gaussian_filter(interp_hist, [sigma,sigma], mode='reflect')

    # density_cmap = LinearSegmentedColormap.from_list("density_cmap", [(0, color), (0.4, color),(1,(1, 1, 1, 0))]).reversed()
    # ax.imshow(np.transpose(interp_smoothed_hist)[::-1], extent=[bin_edgesx.min(), bin_edgesx.max(), bin_edgesy.min(), bin_edgesy.max()], interpolation="nearest", cmap=density_cmap, aspect="auto", zorder=zorder, alpha=0.4)
    # ax.contour(interp_x_centers, interp_y_centers, np.transpose(interp_smoothed_hist), colors=color, linewidths=3., levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]), zorder=zorder)

    ax.contourf(interp_x_centers, interp_y_centers, np.transpose(interp_smoothed_hist), colors=[adjust_lightness(color,1.4), adjust_lightness(color,0.8)], levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]), zorder=zorder, alpha=0.45)
    ax.contour(interp_x_centers, interp_y_centers, np.transpose(interp_smoothed_hist), colors=[color, adjust_lightness(color,0.8)], linewidths=2., levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]), zorder=zorder)

########################################################################################################

data = np.loadtxt("Final_Ngals_HST_z6to10_epsilon1.txt")
Ngals = data[:,1]
weights = data[:,0].astype(int)


Ngals_weighted = np.sort(np.repeat(Ngals, weights))
print(np.searchsorted(Ngals_weighted, 2.), len(Ngals_weighted), 100*np.searchsorted(Ngals_weighted, 2.)/len(Ngals_weighted))


bins = 30
data_for_lims = Ngals
hist, bin_edges, bin_centres = get_hist(np.log(data_for_lims), num_bins=bins, weights=weights)
# hist /= max(hist)
bin_centres = np.exp(bin_centres)
x_array = np.linspace(min(bin_centres), max(bin_centres), 1000)
interpolator = PchipInterpolator(bin_centres, hist)(x_array)
area = simps(interpolator, np.log(x_array))
plt.plot(x_array, interpolator/area, color="black", lw=2)
# plt.plot(x_array, 10**(gaussian_filter1d(np.log10(interpolator/area), 2, mode="nearest")), color=colors[0], alpha=1, ls=(0.,(1,1)), lw=2, label=labels[num])
# plot_hist(data_for_lims, ax, bins, weights, colors[num], ls="dashed")

plt.axvline(2, ls="dashed")
plt.axvline(Ngals_weighted[np.searchsorted(Ngals_weighted, 2.)], color="red", ls="dotted")

plt.semilogx()
plt.show()