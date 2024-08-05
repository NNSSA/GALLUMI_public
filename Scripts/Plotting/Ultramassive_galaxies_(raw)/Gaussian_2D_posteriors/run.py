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
from scipy.ndimage import gaussian_filter
# purple - green - darkgoldenrod - blue - red
plt.style.use("scientific")
colors = ['darkgoldenrod', '#306B37', '#3F7BB6', 'purple', '#BF4145', "#cf630a"]
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

def plot_hist(data, ax, num_bins=30, weights=[None], color=None, label=None, ls=None):
    if not any(weights):
        weights = np.ones(len(data))
    if color == None:
        color="darkblue"
    if ls == None:
        ls == "solid"

    hist, bin_edges, bin_centres = get_hist(np.log(data), num_bins=num_bins, weights=weights)
    ax.plot(np.exp(bin_centres), hist/max(hist), color=color, lw=2, label=label, ls=ls)
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
    gaussian_smoothing = 1.
    sigma = interpolation_smoothing * gaussian_smoothing

    interp_y_centers = scipy.ndimage.zoom(bin_centresy, interpolation_smoothing, mode='reflect')
    interp_x_centers = scipy.ndimage.zoom(bin_centresx,interpolation_smoothing, mode='reflect')
    interp_hist = scipy.ndimage.zoom(hist, interpolation_smoothing, mode='reflect')
    interp_smoothed_hist = scipy.ndimage.filters.gaussian_filter(interp_hist, [sigma,sigma], mode='nearest')

    # density_cmap = LinearSegmentedColormap.from_list("density_cmap", [(0, color), (0.4, color),(1,(1, 1, 1, 0))]).reversed()
    # ax.imshow(np.transpose(interp_smoothed_hist)[::-1], extent=[bin_edgesx.min(), bin_edgesx.max(), bin_edgesy.min(), bin_edgesy.max()], interpolation="nearest", cmap=density_cmap, aspect="auto", zorder=zorder, alpha=0.4)
    # ax.contour(interp_x_centers, interp_y_centers, np.transpose(interp_smoothed_hist), colors=color, linewidths=3., levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]), zorder=zorder)

    ax.contourf(interp_x_centers, interp_y_centers, np.transpose(interp_smoothed_hist), colors=[adjust_lightness(color,1.4), adjust_lightness(color,0.8)], levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]), zorder=zorder, alpha=0.45)
    ax.contour(interp_x_centers, interp_y_centers, np.transpose(interp_smoothed_hist), colors=[color, adjust_lightness(color,0.8)], linewidths=2., levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]), zorder=zorder)

########################################################################################################

data_HST = []
for filepath in glob.iglob('../../Chains/Gaussian_2/*__*.txt'):
    data = np.loadtxt(filepath)
    data_HST.append(data)
data_HST = np.vstack(np.array(data_HST))
weights_HST = data_HST[:,0].astype(int)
amp_HST = data_HST[:,11]
mean_HST = data_HST[:,12]

data_epsilon0p1 = []
for filepath in glob.iglob('../../Chains/Final_Poisson_3_brightest2_epsilon0p1/*__*.txt'):
    data = np.loadtxt(filepath)
    data_epsilon0p1.append(data)
data_epsilon0p1 = np.vstack(np.array(data_epsilon0p1))
weights_epsilon0p1 = data_epsilon0p1[:,0].astype(int)
amp_epsilon0p1 = data_epsilon0p1[:,3]
mean_epsilon0p1 = data_epsilon0p1[:,4]

data_epsilon0p3 = []
for filepath in glob.iglob('../../Chains/Final_Poisson_2_brightest2_epsilon0p3/*__*.txt'):
    data = np.loadtxt(filepath)
    data_epsilon0p3.append(data)
data_epsilon0p3 = np.vstack(np.array(data_epsilon0p3))
weights_epsilon0p3 = data_epsilon0p3[:,0].astype(int)
amp_epsilon0p3 = data_epsilon0p3[:,3]
mean_epsilon0p3 = data_epsilon0p3[:,4]

# data_epsilon0p5 = []
# for filepath in glob.iglob('../../Chains/Final_Poisson_1_brightest2_epsilon0p5/*__*.txt'):
#     data = np.loadtxt(filepath)
#     data_epsilon0p5.append(data)
# data_epsilon0p5 = np.vstack(np.array(data_epsilon0p5))
# weights_epsilon0p5 = data_epsilon0p5[:,0].astype(int)
# amp_epsilon0p5 = data_epsilon0p5[:,3]
# mean_epsilon0p5 = data_epsilon0p5[:,4]

plt.figure(figsize=(8.7,7))
ax = plt.subplot(111)
ax.tick_params(axis='x', which='major', pad=6)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.tick_params(axis='both', which='minor', labelsize=25)

ax.fill_between([-1., 2.], [0., 0.], [0.27340214, 0.27340214], facecolor=adjust_lightness(colors[0],0.8), color=adjust_lightness(colors[0],0.8), linewidth=0.0, zorder=-9, alpha=0.45)
plot_hist2d(mean_HST, amp_HST, ax, num_bins=150, weights=weights_HST, color=colors[0], zorder=0)
plot_hist2d(mean_epsilon0p1, amp_epsilon0p1, ax, num_bins=50, weights=weights_epsilon0p1, color=colors[2], zorder=0)
plot_hist2d(mean_epsilon0p3[amp_epsilon0p3<19.2], amp_epsilon0p3[amp_epsilon0p3<19.2], ax, num_bins=107, weights=weights_epsilon0p3[amp_epsilon0p3<19.2], color=colors[1], zorder=0)

patch_0p1 = mpatches.Patch(color=colors[2], lw=1.5, label=r"$\mathrm{JWST}\ f_\star = 0.1$", alpha=0.5)
patch_0p3 = mpatches.Patch(color=colors[1], lw=1.5, label=r"$\mathrm{JWST}\ f_\star = 0.3$", alpha=0.5)
patch_HST = mpatches.Patch(color=colors[0], lw=1.5, label=r"$\mathrm{HST}$", alpha=0.5)
ax.legend(handles=[patch_0p1, patch_0p3, patch_HST], loc="upper right", frameon=False, markerfirst=False, prop={'size': 17.5}, handlelength=1.3, handletextpad=0.5, numpoints=1)


# plt.legend(loc="upper left", bbox_to_anchor=(0.13, 0.98), frameon=False, markerfirst=True, prop={'size': 17.5}, handlelength=1.3, handletextpad=0.5, numpoints=1)
plt.xlabel(r"$\ln(k_\star)$", fontsize=27)
plt.ylabel(r"$A_\star$", fontsize=27, labelpad=8)
plt.xlim(-0.6, 1)
plt.ylim(0, 35)
# plt.semilogy()
plt.savefig("Gaussian_posteriors_2D.pdf")

