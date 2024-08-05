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
from scipy.ndimage import gaussian_filter1d

# purple - green - darkgoldenrod - blue - red
plt.style.use("scientific")
colors = ["darkgoldenrod", "#306B37", "#3F7BB6", "purple", "#BF4145", "#cf630a"]
linestyles = [
    (0, (1, 1.05)),
    (0, (3, 1, 1, 1)),
    (0, (1, 3)),
    (0, (3, 3.65)),
    (0, (3, 2.772)),
    (0, (3, 1, 1, 1, 1, 1)),
]

############################## Functions ##############################


def ctr_level(hist, lvl, infinite=False):
    hist.sort()
    cum_hist = np.cumsum(hist[::-1])
    cum_hist = cum_hist / cum_hist[-1]

    alvl = np.searchsorted(cum_hist, lvl)
    clist = [0] + [hist[-i] for i in alvl]
    if not infinite:
        return clist[1:]
    return clist


def ctr_level2d(histogram2d, lvl, infinite=False):
    hist = histogram2d.flatten() * 1.0
    hist.sort()
    cum_hist = np.cumsum(hist[::-1])
    cum_hist /= cum_hist[-1]

    alvl = np.searchsorted(cum_hist, lvl)[::-1]
    clist = [0] + [hist[-i] for i in alvl] + [hist.max()]
    if not infinite:
        return clist[1:]
    return clist


###


def get_hist(data, num_bins=40, weights=[None]):
    if not any(weights):
        weights = np.ones(len(data))
    hist, bin_edges = np.histogram(data, bins=num_bins, weights=weights)
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return hist, bin_edges, bin_centres


def get_hist2d(datax, datay, num_bins=40, weights=[None]):
    if not any(weights):
        weights = np.ones(len(datax))
    hist, bin_edgesx, bin_edgesy = np.histogram2d(
        datax, datay, bins=num_bins, weights=weights
    )
    bin_centresx = 0.5 * (bin_edgesx[1:] + bin_edgesx[:-1])
    bin_centresy = 0.5 * (bin_edgesy[1:] + bin_edgesy[:-1])

    return hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy


###


def confidence(data, num_bins=30, weights=[None]):
    hist, bin_edges, bin_centres = get_hist(data, num_bins=num_bins, weights=weights)
    levels = ctr_level(hist.copy(), [0.68])  # [0.68, 0.95])
    pos = [np.searchsorted(hist, levels)[0], -np.searchsorted(hist[::-1], levels)[0]]
    return bin_centres[pos]


###


def plot_hist(data, ax, num_bins=30, weights=[None], color=None, label=None, ls=None):
    if not any(weights):
        weights = np.ones(len(data))
    if color == None:
        color = "darkblue"
    if ls == None:
        ls == "solid"

    hist, bin_edges, bin_centres = get_hist(
        np.log(data), num_bins=num_bins, weights=weights
    )
    ax.plot(
        np.exp(bin_centres), hist / max(hist), color=color, lw=2, label=label, ls=ls
    )
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
        color = "black"

    # hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy = get_hist2d(datax, datay, num_bins=num_bins*2, weights=weights)
    # plt.hist2d(datax, datay, bins=[bin_edgesx, bin_edgesy], weights=weights, cmap="Greys")

    hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy = get_hist2d(
        datax, datay, num_bins=num_bins, weights=weights
    )

    interpolation_smoothing = 3.0
    gaussian_smoothing = 0.5
    sigma = interpolation_smoothing * gaussian_smoothing

    interp_y_centers = scipy.ndimage.zoom(
        bin_centresy, interpolation_smoothing, mode="reflect"
    )
    interp_x_centers = scipy.ndimage.zoom(
        bin_centresx, interpolation_smoothing, mode="reflect"
    )
    interp_hist = scipy.ndimage.zoom(hist, interpolation_smoothing, mode="reflect")
    interp_smoothed_hist = scipy.ndimage.filters.gaussian_filter(
        interp_hist, [sigma, sigma], mode="reflect"
    )

    # density_cmap = LinearSegmentedColormap.from_list("density_cmap", [(0, color), (0.4, color),(1,(1, 1, 1, 0))]).reversed()
    # ax.imshow(np.transpose(interp_smoothed_hist)[::-1], extent=[bin_edgesx.min(), bin_edgesx.max(), bin_edgesy.min(), bin_edgesy.max()], interpolation="nearest", cmap=density_cmap, aspect="auto", zorder=zorder, alpha=0.4)
    # ax.contour(interp_x_centers, interp_y_centers, np.transpose(interp_smoothed_hist), colors=color, linewidths=3., levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]), zorder=zorder)

    ax.contourf(
        interp_x_centers,
        interp_y_centers,
        np.transpose(interp_smoothed_hist),
        colors=[adjust_lightness(color, 1.4), adjust_lightness(color, 0.8)],
        levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]),
        zorder=zorder,
        alpha=0.45,
    )
    ax.contour(
        interp_x_centers,
        interp_y_centers,
        np.transpose(interp_smoothed_hist),
        colors=[color, adjust_lightness(color, 0.8)],
        linewidths=2.0,
        levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]),
        zorder=zorder,
    )


########################################################################################################

plt.figure(figsize=(8.7, 7))
ax = plt.subplot(111)
ax.tick_params(axis="x", which="major", pad=6)
plt.tick_params(axis="both", which="major", labelsize=25)
plt.tick_params(axis="both", which="minor", labelsize=25)

names = [
    "Final_Ngals_epsilon0p5.txt",
    "Final_Ngals_epsilon0p3.txt",
    "Final_Ngals_epsilon0p1.txt",
]
labels = ["epsilon = 0.5", "epsilon = 0.3", "epsilon = 0.1"]
binss = [16, 16, 16]

for num, name in enumerate(names):

    data = np.loadtxt(name, unpack=True)
    weights = data[0]

    bins = binss[num]
    data_for_lims = data[1]
    hist, bin_edges, bin_centres = get_hist(
        np.log(data_for_lims), num_bins=bins, weights=weights
    )
    # hist /= max(hist)
    hist = hist[1:]
    bin_centres = np.exp(bin_centres[1:])
    x_array = np.linspace(min(bin_centres), max(bin_centres), 100)
    interpolator = PchipInterpolator(bin_centres, hist)(x_array)
    area = simps(interpolator, np.log(x_array))
    plt.plot(x_array, interpolator / area, color=colors[num], lw=2, ls=(0, (3, 1.5)))
    # plot_hist(data_for_lims, ax, bins, weights, colors[num], labels[num], "dashed")


names = [
    "Final_Ngals_HST_z6to10_epsilon0p5.txt",
    "Final_Ngals_HST_z6to10_epsilon0p3.txt",
    "Final_Ngals_HST_z6to10_epsilon0p1.txt",
]
labels = ["epsilon = 0.5", "epsilon = 0.3", "epsilon = 0.1"]
binss = [30, 30, 30]

for num, name in enumerate(names):

    data = np.loadtxt(name, unpack=True)
    weights = data[0]

    bins = binss[num]
    data_for_lims = data[1]
    hist, bin_edges, bin_centres = get_hist(
        np.log(data_for_lims), num_bins=bins, weights=weights
    )
    # hist /= max(hist)
    bin_edges = np.exp(bin_edges[:-1])
    x_array = np.linspace(min(bin_edges), max(bin_edges), 1000)
    interpolator = PchipInterpolator(bin_edges, hist)(x_array)
    area = simps(interpolator, np.log(x_array))
    plt.plot(
        x_array,
        10 ** (gaussian_filter1d(np.log10(interpolator / area), 2, mode="nearest")),
        color=colors[num],
        lw=2,
    )
    # plot_hist(data_for_lims, ax, bins, weights, colors[num], ls="dashed")

plt.axvline(2.0, color="black", alpha=0.7, ls=(0.0, (1.2, 1.5)), lw=2.0)

plt.plot([0.0, 0.0], [0.0, 0.0], color="black", label=r"$\mathrm{HST\ UV\ LF}$")
plt.plot(
    [0.0, 0.0], [0.0, 0.0], color="black", label=r"$\mathrm{JWST}$", ls=(0, (3, 1.5))
)

plt.text(
    4.2e-6,
    0.30,
    r"$f_\star = 0.1$",
    weight="bold",
    fontsize=22,
    color=colors[2],
    zorder=6,
    rotation=90,
)
plt.text(
    1.18e-3,
    0.30,
    r"$f_\star = 0.3$",
    weight="bold",
    fontsize=22,
    color=colors[1],
    zorder=6,
    rotation=90,
)
plt.text(
    9.9e-3,
    0.30,
    r"$f_\star = 0.5$",
    weight="bold",
    fontsize=22,
    color=colors[0],
    zorder=6,
    rotation=90,
)


plt.legend(
    loc="upper left",
    bbox_to_anchor=(0.13, 0.98),
    frameon=False,
    markerfirst=True,
    prop={"size": 17.5},
    handlelength=1.3,
    handletextpad=0.5,
    numpoints=1,
)
plt.semilogx()
plt.xlabel(r"$N_\mathrm{gal}$", fontsize=27)
plt.ylabel(r"$P(\ln N_\mathrm{gal})$", fontsize=27, labelpad=8)
plt.xlim(1e-6, 1e2)
plt.ylim(0.0, 0.7)
plt.savefig("Ngal_dist.pdf")
