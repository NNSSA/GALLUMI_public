import numpy as np
from matplotlib import pyplot as plt
import glob
from matplotlib import patches as mpatches
from scipy.interpolate import PchipInterpolator
import scipy.ndimage
plt.style.use("../template.mplstyle")

# purple - green - darkgoldenrod - blue - red
colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145', "#cf630a"]
linestyles = [(0, (1,1.05)), (0, (3, 1, 1, 1)), (0, (1,3)), (0, (3,3.65)), (0, (3,2.772)), (0, (3, 1, 1, 1, 1, 1))]

#############################################################

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

    ax.contourf(interp_x_centers, interp_y_centers, np.transpose(interp_smoothed_hist), colors=[adjust_lightness(color,1.4), adjust_lightness(color,0.8)], levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]), zorder=zorder, alpha=0.45)
    ax.contour(interp_x_centers, interp_y_centers, np.transpose(interp_smoothed_hist), colors=[color, adjust_lightness(color,0.8)], linewidths=2., levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]), zorder=zorder)

#############################################################

UVLF_HST_ST_model1 = []
UVLF_HST_ST_model2 = []
UVLF_HST_ST_model3 = []

for filepath in glob.iglob('../../UVLF_HST_ST_model1/*__*.txt'):
    data = np.loadtxt(filepath)
    UVLF_HST_ST_model1.append(data)
for filepath in glob.iglob('../../UVLF_HST_ST_model2/*__*.txt'):
    data = np.loadtxt(filepath)
    UVLF_HST_ST_model2.append(data)
for filepath in glob.iglob('../../UVLF_HST_ST_model3/*__*.txt'):
    data = np.loadtxt(filepath)
    UVLF_HST_ST_model3.append(data)

UVLF_HST_ST_model1 = np.vstack(np.array(UVLF_HST_ST_model1))
UVLF_HST_ST_model2 = np.vstack(np.array(UVLF_HST_ST_model2))
UVLF_HST_ST_model3 = np.vstack(np.array(UVLF_HST_ST_model3))

fig = plt.figure(figsize=(24.,6.))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
ax1.tick_params(axis='x', which='major', pad=6)
ax2.tick_params(axis='x', which='major', pad=6)
ax3.tick_params(axis='x', which='major', pad=6)
ax1.tick_params(axis='both', which='both', labelsize=26)
ax2.tick_params(axis='both', which='both', labelsize=26)
ax3.tick_params(axis='both', which='both', labelsize=26)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(2.2)
    ax2.spines[axis].set_linewidth(2.2)
    ax3.spines[axis].set_linewidth(2.2)

####################################

redshift = 6.
Mhalos = np.geomspace(1e9, 1e13, 50)

def Hubble(z, h, Omega_m):
    return 100. * h * np.sqrt(Omega_m * (1+z)**3 + (1.-Omega_m)) * 1000. * 86400. * 365.25 / 3.08567758e22

def Mh_MUV_v1(redshift, Mh):
    kappaUV = 1.15e-28
    alphastar = UVLF_HST_ST_model2[:,6]
    betastar = UVLF_HST_ST_model2[:,7]
    epsilonstar = 10**(UVLF_HST_ST_model2[:,8] * np.log10((1 + redshift)/(1 + 6)) + UVLF_HST_ST_model2[:,9])
    Mc = 10**(UVLF_HST_ST_model2[:,10] * np.log10((1 + redshift)/(1 + 6)) + UVLF_HST_ST_model2[:,11])

    return -2.5 * np.log10(epsilonstar * Mh * Hubble(redshift, UVLF_HST_ST_model2[:,14], UVLF_HST_ST_model2[:,15]) / ((Mh/Mc)**alphastar + (Mh/Mc)**betastar) / kappaUV) + 51.63

def Mh_MUV_v1_average(redshift, Mh, alphastar, betastar, epsilonstar, Mc, h, Omega_m):
    kappaUV = 1.15e-28
    return -2.5 * np.log10(epsilonstar * Mh * Hubble(redshift, h, Omega_m) / ((Mh/Mc)**alphastar + (Mh/Mc)**betastar) / kappaUV) + 51.63

Mstar_fit_a = {4:-0.54, 5:-0.5, 6:-0.57, 7:-0.49, 8:-0.49, 9:-0.46, 10:-0.41}
Mstar_fit_b = {4:9.7, 5:9.59, 6:9., 7:8.9, 8:8.8, 9:8.6, 10:8.5}
Mstar_fit_anchor = {4:21, 5:21, 6:20.5, 7:20.5, 8:20.5, 9:20.5, 10:20.5}
def Mh_MUV_v3(redshift, Mh):
    alphastar = UVLF_HST_ST_model3[:,6]
    betastar = UVLF_HST_ST_model3[:,7]
    epsilonstar = 10**(UVLF_HST_ST_model3[:,8] * np.log10((1 + redshift)/(1 + 6)) + UVLF_HST_ST_model3[:,9])
    Mc = 10**(UVLF_HST_ST_model3[:,10] * np.log10((1 + redshift)/(1 + 6)) + UVLF_HST_ST_model3[:,11])

    return (np.log10(epsilonstar * Mh / ((Mh/Mc)**alphastar + (Mh/Mc)**betastar)) - Mstar_fit_b[int(redshift)]) / Mstar_fit_a[int(redshift)] - Mstar_fit_anchor[int(redshift)]

def Mh_MUV_v3_average(redshift, Mh, alphastar, betastar, epsilonstar, Mc):

    return (np.log10(epsilonstar * Mh / ((Mh/Mc)**alphastar + (Mh/Mc)**betastar)) - Mstar_fit_b[int(redshift)]) / Mstar_fit_a[int(redshift)] - Mstar_fit_anchor[int(redshift)]

def limits(func, Mhalos, weights):
    lower = []
    upper = []

    for Mh in Mhalos:
        hist, bin_edges, bin_centres = get_hist(func(redshift, Mh), num_bins=26, weights=weights)
        xarray = np.linspace(min(bin_centres), max(bin_centres), 1000)
        interpolator = PchipInterpolator(bin_centres, hist)(xarray)
        levels = ctr_level(interpolator.copy(), [0.68])
        pos = [np.searchsorted(interpolator[:np.argmax(interpolator)], levels)[0]-1, -np.searchsorted((interpolator[::-1])[:np.argmax(interpolator[::-1])], levels)[0]]
        lower.append(min(xarray[pos[0]], xarray[pos[1]]))
        upper.append(max(xarray[pos[0]], xarray[pos[1]]))
    return np.array(lower), np.array(upper)


Mhalos_v2, data_v2_min, data_v2_max = np.loadtxt("Mh_MUV_lower_upper_model1.txt", unpack=True)
data_v1_min, data_v1_max = limits(Mh_MUV_v1, Mhalos, UVLF_HST_ST_model2[:,0])
data_v3_min, data_v3_max = limits(Mh_MUV_v3, Mhalos, UVLF_HST_ST_model3[:,0])
data_v1_average = Mh_MUV_v1_average(6., Mhalos, -6.8910e-01, 1.1028e+00, 10**(-6.2816e-01), 10**(1.1988e+01), 6.8436e-01, 2.9830e-01)
data_v2_average = np.loadtxt("Mh_MUV_average_model1.txt", unpack=True)
data_v3_average = Mh_MUV_v3_average(6., Mhalos, -1.1856e+00, 1.3528e+00, 10**(-1.5031e+00), 10**(1.2186e+01))


ax1.fill_between(np.append(data_v2_min, data_v2_max[::-1]), np.append(Mhalos_v2, Mhalos_v2[::-1]), facecolor="none", color=colors[3], linewidth=0.0, alpha=0.3, zorder=3)
ax1.fill_between(np.append(data_v1_min, data_v1_max[::-1]), np.append(Mhalos, Mhalos[::-1]), facecolor="none", color="#993437", linewidth=0.0, alpha=0.3, zorder=2)
ax1.fill_between(np.append(data_v3_min, data_v3_max[::-1]), np.append(Mhalos, Mhalos[::-1]), facecolor="none", color=colors[1], linewidth=0.0, alpha=0.3, zorder=1)

ax1.plot(data_v2_average[1], data_v2_average[0], color=colors[3], linewidth=3, alpha=1., zorder=3)
ax1.plot(data_v1_average, Mhalos, color="#993437", linewidth=2, alpha=1., zorder=2, linestyle=(0, (3,2.772)))
ax1.plot(data_v3_average, Mhalos, color=colors[1], linewidth=3., alpha=1., zorder=1, linestyle=(0, (1,1.05)))

patch_blue = mpatches.Patch(color=colors[3], lw=1.5, label=r"$\mathrm{model\, I}$", alpha=0.8)
patch_red = mpatches.Patch(color="#BF4145", lw=1.5, label=r"$\mathrm{model\, II}$", alpha=0.8)
patch_green = mpatches.Patch(color=colors[1], lw=1.5, label=r"$\mathrm{model\, III}$", alpha=0.7)

ax1.set_ylabel(r'$M_\mathrm{h}\ \mathrm{at}\ z=6\ [M_\odot]$', fontsize=30, labelpad=14)
ax1.set_xlabel(r'$M_\mathrm{UV}$', fontsize=30, labelpad=10)

ax1.legend(handles=[patch_blue, patch_red, patch_green], loc="upper right", frameon=False, markerfirst=False, prop={'size': 22}, handlelength=1.9, handletextpad=0.5, numpoints=1)

ax1.semilogy()
ax1.axis(xmin=-23, xmax=-16., ymin=8e9, ymax=5e12)


####################################


plot_hist2d(datax=UVLF_HST_ST_model1[:,-7], datay=UVLF_HST_ST_model1[:,2], ax=ax2, num_bins=20, weights=UVLF_HST_ST_model1[:,0], color=colors[3], zorder=3)
plot_hist2d(datax=UVLF_HST_ST_model2[:,-7], datay=UVLF_HST_ST_model2[:,2], ax=ax2, num_bins=20, weights=UVLF_HST_ST_model2[:,0], color="#993437", zorder=2)
plot_hist2d(datax=UVLF_HST_ST_model3[:,-7], datay=UVLF_HST_ST_model3[:,2], ax=ax2, num_bins=20, weights=UVLF_HST_ST_model3[:,0], color=colors[1], zorder=1)
ax2.set_xlabel(r'$\Omega_\mathrm{m}$', labelpad=11, fontsize=30)
ax2.set_ylabel(r'$\sigma_8$', labelpad=8, fontsize=30)
ax2.set_xlim(0.2, 0.4)
ax2.set_ylim(0.35, 1.3)

plot_hist2d(datax=UVLF_HST_ST_model1[:,5], datay=UVLF_HST_ST_model1[:,2], ax=ax3, num_bins=20, weights=UVLF_HST_ST_model1[:,0], color=colors[3], zorder=3)
plot_hist2d(datax=UVLF_HST_ST_model2[:,5], datay=UVLF_HST_ST_model2[:,2], ax=ax3, num_bins=20, weights=UVLF_HST_ST_model2[:,0], color="#993437", zorder=2)
plot_hist2d(datax=UVLF_HST_ST_model3[:,5], datay=UVLF_HST_ST_model3[:,2], ax=ax3, num_bins=20, weights=UVLF_HST_ST_model3[:,0], color=colors[1], zorder=1)
ax3.set_xlabel(r'$n_\mathrm{s}$', labelpad=10, fontsize=30)
ax3.set_ylabel(r'$\sigma_8$', labelpad=8, fontsize=30)
ax3.set_xlim(0.7, 1.3)
ax3.set_ylim(0.35, 1.3)

plt.savefig("Models_comparison.pdf")

