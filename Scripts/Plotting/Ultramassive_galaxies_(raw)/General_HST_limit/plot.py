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

plt.style.use("scientific")
matplotlib.rc("text", usetex=True)
matplotlib.rc("text.latex", preamble=r"\usepackage{amsmath}")
# purple - green - darkgoldenrod - blue - red
colors = ["purple", "#306B37", "darkgoldenrod", "#3F7BB6", "#BF4145", "#cf630a"]
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
        color = "darkblue"
    if ls == None:
        ls == "solid"

    hist, bin_edges, bin_centres = get_hist(data, num_bins=num_bins, weights=weights)
    ax.plot(bin_centres, hist / max(hist), color=color, lw=2, label=label, ls=ls)
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

plt.figure(figsize=(8.7, 14.5))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

for ax in [ax1, ax2]:
    ax.tick_params(axis="x", which="major", pad=6)
    ax.tick_params(axis="both", which="major", labelsize=25)
    ax.tick_params(axis="both", which="minor", labelsize=25)

data_lcdm = np.loadtxt("ngals_lcdm.txt", unpack=True)
data_lcdm_interp = PchipInterpolator(data_lcdm[0], data_lcdm[1])

data1 = np.loadtxt("Ngals_HST_z6to10_z7p5forecast.txt")
data2 = np.loadtxt("Ngals_HST_z6to10_z7p5forecast_2.txt")
data = np.vstack((data1, data2[1:, :]))
mcuts = np.unique(data[:, 0])

weights = data[0, 1:].astype(int)
ngals = data[1:, 1:]

levels95 = []
levels68 = []

for num, mass in enumerate(mcuts):
    ngals_weighted = np.sort(np.repeat(ngals[num], weights))
    levels68.append(ngals_weighted[int(len(ngals_weighted) * 0.68)])
    levels95.append(ngals_weighted[int(len(ngals_weighted) * 0.95)])

lcdm = data_lcdm[1]
l68 = np.array(levels68)
l95 = np.array(levels95)

Mstar_cut_lcdm = data_lcdm[0] * 0.156 * 0.3
Mstar_cuts = mcuts * 0.156 * 0.3

ax1.plot(
    Mstar_cut_lcdm, lcdm, color="black", lw=2, zorder=2, label=r"$\Lambda\mathrm{CDM}$"
)
ax1.plot(
    Mstar_cuts,
    l95,
    color=colors[3],
    ls=(1.6, (2, 1, 2, 1)),
    lw=2.2,
    zorder=2,
    label=r"$\mathrm{HST}\ 95\%\ \mathrm{CL}$",
)
# ax1.plot(Mstar_cuts, l68, color=colors[4], ls=(0,(2,1,2,1)), lw=2, zorder=2, label=r"$\mathrm{HST}\ 68\%\ \mathrm{CL}$")
# ax1.fill(np.append(Mstar_cut_lcdm, Mstar_cuts[::-1]), np.append(lcdm, l68[::-1]), color=colors[4], alpha=0.5, lw=0, zorder=1)
ax1.fill(
    np.append(Mstar_cut_lcdm, Mstar_cuts[::-1]),
    np.append(lcdm, l95[::-1]),
    color=colors[3],
    alpha=0.5,
    lw=0,
    zorder=0,
)

ax1.scatter(
    10 ** (10.89),
    1e-5,
    marker="*",
    c=colors[4],
    alpha=1.0,
    s=200,
    linewidths=0.3,
    edgecolors="k",
    zorder=9,
    label=r"$\mathrm{Labb\acute{\mathrm{e}}\ et\ al.}$",
)

ax1.axhline(1 / 1e5, linestyle=(0, (1, 1.05)), color=colors[4], alpha=0.6, lw=2)
ax1.text(
    2e8,
    1 / 6.5e4,
    r"$\mathrm{CEERS}$",
    weight="bold",
    fontsize=18,
    color=colors[4],
    zorder=6,
)
ax1.annotate(
    "",
    xytext=(1.8e8, 1 / 1e5),
    xy=(1.8e8, 2e-4),
    arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": colors[4]},
    color=colors[4],
)

ax1.axhline(1 / 5e6, linestyle=(0, (3, 1, 1, 1)), color=colors[1], alpha=0.6, lw=2)
ax1.text(
    2e8,
    1 / 3.4e6,
    r"$\mathrm{COSMOS-Webb}$",
    weight="bold",
    fontsize=18,
    color=colors[1],
    zorder=6,
)
ax1.annotate(
    "",
    xytext=(1.8e8, 1 / 5e6),
    xy=(1.8e8, 4.1e-6),
    arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": colors[1]},
    color=colors[1],
)

ax1.axhline(1e-10, linestyle=(0, (3, 1)), color=colors[2], alpha=0.6, lw=2)
ax1.text(
    2e8,
    1 / 6.5e9,
    r"$\mathrm{Roman\ High-Latitude}$",
    weight="bold",
    fontsize=18,
    color=colors[2],
    zorder=6,
)
ax1.annotate(
    "",
    xytext=(1.8e8, 1 / 1e10),
    xy=(1.8e8, 2e-9),
    arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": colors[2]},
    color=colors[2],
)

ax1.text(
    2.867e11,
    5.1e-3,
    r"$z = 7.5$",
    weight="bold",
    fontsize=17.5,
    color="black",
    zorder=6,
)
ax1.text(
    2.55e11,
    8.5e-4,
    r"$f_\star = 0.3$",
    weight="bold",
    fontsize=17.5,
    color="black",
    zorder=6,
)

ax1.semilogx()
ax1.semilogy()
ax1.set_xlabel(r"$M_\star^\mathrm{obs}\ [M_\odot]$", fontsize=27)
ax1.set_ylabel(
    r"$n_\mathrm{gal}(M_\star \geq M_\star^\mathrm{obs})\ [\mathrm{Mpc}^{-3}]$",
    fontsize=27,
)
ax1.legend(
    loc="upper right",
    frameon=False,
    markerfirst=False,
    prop={"size": 17.5},
    handlelength=1.3,
    handletextpad=0.5,
    numpoints=1,
)
ax1.set_xlim(1e8, 1e12)
ax1.set_ylim(1e-11, 1e1)
ax1.set_yticks([1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1])
ax1.set_yticklabels(
    [
        r"$10^{-11}$",
        r"$10^{-9}$",
        r"$10^{-7}$",
        r"$10^{-5}$",
        r"$10^{-3}$",
        r"$10^{-1}$",
        r"$10^{1}$",
    ]
)
###########################################################################
###########################################################################
###########################################################################

Harikane_2022_data = np.loadtxt("Harikane_2022_data.txt", unpack=True)
Donnan_2022_data = np.loadtxt("Donnan_2022_data.txt", unpack=True)
Bouwens_2022_data = np.loadtxt("Bouwens_2022_data.txt", unpack=True)
Finkelstein_2022_data = [
    -20.4,
    0.000019259711460570002,
    0.000019259711460570002 - 0.000003416111534196904,
    0.00006716201168714377 - 0.000019259711460570002,
]
Naidu_2022_data = [
    -20.900000000000002,
    -20.900000000000002 + 21.401000000000003,
    -20.402000000000058 + 20.900000000000002,
    10 ** (-5.0484186795491155),
    10 ** (-5.0484186795491155) - 10 ** (-5.499304347826088),
    10 ** (-4.684489533011273) - 10 ** (-5.0484186795491155),
]

UVLF_bestfit = np.loadtxt("UVLFs_HST_z12_bestfit.txt")

UVLF_data = np.loadtxt("UVLFs_HST_z12.txt")
MUVs = UVLF_data[0, 1:]
weights = UVLF_data[1:, 0]
UVLFs = UVLF_data[1:, 1:]

UVLF_upper_68 = []
UVLF_lower_68 = []
UVLF_upper_95 = []
UVLF_lower_95 = []


for numMUV in range(len(MUVs)):
    hist, bin_edges, bin_centres = get_hist(
        UVLFs[:, numMUV], num_bins=80, weights=weights
    )
    xarray = np.linspace(min(bin_centres), max(bin_centres), 500)
    interpolator = PchipInterpolator(bin_centres, hist)(xarray)
    level_68 = ctr_level(interpolator.copy(), [0.68])
    level_95 = ctr_level(interpolator.copy(), [0.95])
    pos_68 = [
        np.searchsorted(interpolator[: np.argmax(interpolator)], level_68)[0] - 1,
        -np.searchsorted(
            (interpolator[::-1])[: np.argmax(interpolator[::-1])], level_68
        )[0],
    ]
    pos_95 = [
        np.searchsorted(interpolator[: np.argmax(interpolator)], level_95)[0] - 1,
        -np.searchsorted(
            (interpolator[::-1])[: np.argmax(interpolator[::-1])], level_95
        )[0],
    ]
    UVLF_lower_68.append(min(xarray[pos_68[0]], xarray[pos_68[1]]))
    UVLF_upper_68.append(max(xarray[pos_68[0]], xarray[pos_68[1]]))
    UVLF_lower_95.append(min(xarray[pos_95[0]], xarray[pos_95[1]]))
    UVLF_upper_95.append(max(xarray[pos_95[0]], xarray[pos_95[1]]))

    # plt.axvline(xarray[np.argmin(np.abs(interpolator-level_68[0]))], linestyle="dashed", color=colors[4], alpha=0.6)

    # plot_hist(UVLFs[:,numMUV], ax2, 50, weights, "black", None, "solid")

    # print(pos)


# ax2.semilogx()
l1 = ax2.errorbar(
    Naidu_2022_data[0],
    Naidu_2022_data[3],
    xerr=np.array([[Naidu_2022_data[1]], [Naidu_2022_data[2]]]),
    yerr=np.array([[Naidu_2022_data[4]], [Naidu_2022_data[5]]]),
    ls="None",
    marker="X",
    markersize=10,
    markeredgewidth=0.1,
    elinewidth=1.5,
    color=colors[3],
    zorder=2,
    label=r"$\mathrm{Naidu\ et\ al.}\ (z \approx 13)$",
)
l2 = ax2.errorbar(
    Donnan_2022_data[0],
    Donnan_2022_data[2],
    xerr=Donnan_2022_data[1],
    yerr=Donnan_2022_data[3:],
    ls="None",
    marker="8",
    markersize=10,
    markeredgewidth=0.1,
    elinewidth=1.5,
    color=colors[2],
    zorder=1,
    label=r"$\mathrm{Donnan\ et\ al.}\ (z \approx 13)$",
)
l3 = ax2.errorbar(
    Bouwens_2022_data[0],
    Bouwens_2022_data[1],
    yerr=Bouwens_2022_data[2],
    ls="None",
    marker="s",
    markersize=10,
    markeredgewidth=0.1,
    elinewidth=1.5,
    color=colors[4],
    zorder=1,
    label=r"$\mathrm{Bouwens\ et\ al.}\ (z \approx 13)$",
)
l4 = ax2.errorbar(
    Harikane_2022_data[0],
    Harikane_2022_data[1],
    yerr=Harikane_2022_data[2:],
    ls="None",
    marker="*",
    markersize=14,
    markeredgewidth=0.1,
    elinewidth=1.5,
    color=colors[1],
    zorder=1,
    label=r"$\mathrm{Harikane\ et\ al.}\ (z \approx 12)$",
)
l5 = ax2.errorbar(
    Finkelstein_2022_data[0],
    Finkelstein_2022_data[1],
    yerr=np.array([[Finkelstein_2022_data[2]], [Finkelstein_2022_data[3]]]),
    ls="None",
    marker="P",
    markersize=10,
    markeredgewidth=0.1,
    elinewidth=1.5,
    color=colors[0],
    zorder=1,
    label=r"$\mathrm{Finkelstein\ et\ al.}\ (z \approx 12)$",
)

ax2.fill_between(
    MUVs, UVLF_lower_68, UVLF_upper_68, color=colors[3], alpha=0.7, lw=0, zorder=0
)
ax2.fill_between(
    MUVs, UVLF_lower_95, UVLF_upper_95, color=colors[3], alpha=0.5, lw=0, zorder=0
)

ax2.text(
    -22.8,
    3.37e-4,
    r"$\boldsymbol{\mathrm{HST}\ (z = 12,\ \mathrm{extrapolated})}$",
    weight="bold",
    fontsize=17.5,
    color="black",
    zorder=6,
)
ax2.text(
    -18.755,
    2.6e-7,
    r"$\boldsymbol{\mathrm{JWST}\ (z = 12-13)}$",
    weight="bold",
    fontsize=17.5,
    color="black",
    zorder=6,
)
patch_68 = mpatches.Patch(
    color=colors[3], lw=1.5, label=r"$\mathrm{68\%\ \mathrm{CL}}$", alpha=0.8
)
patch_95 = mpatches.Patch(
    color=colors[3], lw=1.5, label=r"$\mathrm{95\%\ \mathrm{CL}}$", alpha=0.5
)

ax2.semilogy()
ax2.set_xlabel(r"$M_\mathrm{UV}\ [\mathrm{mag}]$", labelpad=9, fontsize=27)
ax2.set_ylabel(
    r"$\Phi_\mathrm{UV}\ [\mathrm{Mpc^{-3}\,mag^{-1}}]$", labelpad=8, fontsize=27
)

legend2 = ax2.legend(
    handles=[l1, l2, l3, l4, l5],
    loc="lower right",
    frameon=False,
    markerfirst=False,
    prop={"size": 17.5},
    handlelength=1.3,
    handletextpad=0.5,
    numpoints=1,
)
ax2.add_artist(legend2)
ax2.legend(
    handles=[patch_68, patch_95],
    loc="upper left",
    bbox_to_anchor=(0.0, 0.93),
    frameon=False,
    markerfirst=True,
    prop={"size": 17.5},
    handlelength=1.3,
    handletextpad=0.5,
    numpoints=1,
)

ax2.set_xlim(-23, -16)
ax2.set_ylim(1e-9, 1e-3)

plt.subplots_adjust(hspace=0.23)
plt.savefig("HST_forecasts.pdf")

# plt.show()


# data_lcdm = np.loadtxt("ngals_lcdm.txt", unpack=True)
# data_lcdm_interp = PchipInterpolator(data_lcdm[0], data_lcdm[1])

# data = np.loadtxt("ngals_HST_z6to10_z7forecast_new.txt")
# mcuts = np.unique(data[:,0])
# weights = data[0,1:]
# ngals = data[1:,1:]
# bins = 20

# levels95 = []
# levels68 = []

# for num, mass in enumerate(mcuts):

#     hist, bin_edges, bin_centres = get_hist(np.log(ngals[num]), num_bins=bins, weights=weights)
#     bin_centres = np.exp(bin_centres)
#     bin_edges = np.exp(bin_edges)
#     hist /= max(hist)

#     xarray = np.linspace(min(bin_centres), max(bin_centres), 100)
#     interpolator = PchipInterpolator(bin_centres, hist)(xarray)
#     level_95 = ctr_level(interpolator.copy(), [0.95])
#     level_68 = ctr_level(interpolator.copy(), [0.68])
#     levels95.append(xarray[np.argmin(np.abs(interpolator-level_95[0]))])
#     levels68.append(xarray[np.argmin(np.abs(interpolator-level_68[0]))])

#     ax.axvline(xarray[np.argmin(np.abs(interpolator-level_95[0]))], linestyle="dashed", color=colors[4], alpha=0.6)
#     ax.axvline(xarray[np.argmin(np.abs(interpolator-level_68[0]))], linestyle="dashed", color=colors[3], alpha=0.6)
#     ax.plot((xarray[np.argmin(np.abs(interpolator-level_68[0]))]/1.2, xarray[np.argmin(np.abs(interpolator-level_68[0]))]*1.2), (level_68, level_68), color="orange")
#     ax.plot((xarray[np.argmin(np.abs(interpolator-level_95[0]))]/1.2, xarray[np.argmin(np.abs(interpolator-level_95[0]))]*1.2), (level_95, level_95), color="purple")

#     plot_hist(ngals[num], ax, bins, weights, "black", None, "solid")
#     plt.plot(xarray, interpolator, color="gray")

# plt.semilogx()
# plt.semilogy()
# plt.show()

# plt.figure()
# plt.plot(mcuts, levels68/data_lcdm_interp(mcuts), color=colors[3], label="68% CL")
# plt.plot(mcuts, levels95/data_lcdm_interp(mcuts), color=colors[4], label="95% CL")
# # plt.plot(data_lcdm[0], data_lcdm[1], color="black", ls="dashed")

# plt.fill_between(mcuts, np.ones(len(mcuts)), levels68/data_lcdm_interp(mcuts), color=colors[3], alpha=0.5)
# plt.fill_between(mcuts, np.ones(len(mcuts)), levels95/data_lcdm_interp(mcuts), color=colors[4], alpha=0.5)

# plt.semilogx()
# # plt.semilogy()
# plt.xlabel("Mhcut in Msun")
# plt.ylabel("Ngal/Ngal(LCDM)")
# plt.legend()
# plt.show()
