import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import glob
from matplotlib import patches as mpatches
import scipy.ndimage

plt.style.use("../template.mplstyle")
# purple - green - darkgoldenrod - blue - red
colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145']

#############################################################

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

###

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

#############################################################

UVLF_HST_model1 = []
UVLF_IllustrisTNG_model1 = []

for filepath in glob.iglob('../../Data/UVLF_HST_ST_model1/*__*.txt'):
    data = np.loadtxt(filepath)
    UVLF_HST_model1.append(data)
for filepath in glob.iglob('../../Data/UVLF_IllustrisTNG_ST_model1/*__*.txt'):
    data = np.loadtxt(filepath)
    UVLF_IllustrisTNG_model1.append(data)

UVLF_HST_model1 = np.vstack(np.array(UVLF_HST_model1))
UVLF_IllustrisTNG_model1 = np.vstack(np.array(UVLF_IllustrisTNG_model1))

sigma8fid = 0.8159
Omegamfid = 0.3089
nsfid = 0.9667
hfid = 0.6774

fig = plt.figure(figsize=(22.,6.))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
ax1.tick_params(axis='x', which='major', pad=6)
ax2.tick_params(axis='x', which='major', pad=6)
ax3.tick_params(axis='x', which='major', pad=6)
ax1.tick_params(axis='both', which='both', labelsize=24)
ax2.tick_params(axis='both', which='both', labelsize=24)
ax3.tick_params(axis='both', which='both', labelsize=24)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(2.2)
    ax2.spines[axis].set_linewidth(2.2)
    ax3.spines[axis].set_linewidth(2.2)

plot_hist2d(datax=UVLF_HST_model1[:,-7], datay=UVLF_HST_model1[:,2], ax=ax1, num_bins=20, weights=UVLF_HST_model1[:,0], color=colors[3], zorder=3)
plot_hist2d(datax=UVLF_IllustrisTNG_model1[:,-1], datay=UVLF_IllustrisTNG_model1[:,2], ax=ax1, num_bins=20, weights=UVLF_IllustrisTNG_model1[:,0], color=colors[-1], zorder=2)
ax1.axhline(sigma8fid, color="black", alpha=0.5, ls=(1,(5.2,5)))
ax1.axvline(Omegamfid, color="black", alpha=0.5, ls=(1,(5,5)))
ax1.set_xlabel(r'$\Omega_\mathrm{m}$', labelpad=10, fontsize=30)
ax1.set_ylabel(r'$\sigma_8$', labelpad=12, fontsize=30)
ax1.set_xlim(0.23, 0.37)
ax1.set_ylim(0.35, 1.2)
ax1.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2])
ax1.set_yticklabels([r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$', r'$1.2$'])

patch_purple = mpatches.Patch(color=colors[3], lw=1.5, label=r"$\mathrm{HST}$")
patch_red = mpatches.Patch(color=colors[-1], lw=1.5, label=r"$\mathrm{IllustrisTNG}$")
leg = ax1.legend(handles=[patch_purple, patch_red], loc="upper center", frameon=False, markerfirst=True, prop={'size': 19}, handlelength=1.9, handletextpad=0.5, numpoints=1, ncol=3, columnspacing=1.4)

plot_hist2d(datax=UVLF_HST_model1[:,5], datay=UVLF_HST_model1[:,2], ax=ax2, num_bins=20, weights=UVLF_HST_model1[:,0], color=colors[3], zorder=3)
plot_hist2d(datax=UVLF_IllustrisTNG_model1[:,5], datay=UVLF_IllustrisTNG_model1[:,2], ax=ax2, num_bins=20, weights=UVLF_IllustrisTNG_model1[:,0], color=colors[-1], zorder=2)
ax2.axhline(sigma8fid, color="black", alpha=0.5, ls=(1,(5.2,5)))
ax2.axvline(nsfid, color="black", alpha=0.5, ls=(1,(5,5)))
ax2.set_xlabel(r'$n_\mathrm{s}$', labelpad=10, fontsize=30)
ax2.set_xlim(0.7, 1.3)
ax2.set_ylim(0.35, 1.2)
ax2.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2])
ax2.set_yticklabels([r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$', r'$1.2$'])

plot_hist2d(datax=UVLF_HST_model1[:,-8], datay=UVLF_HST_model1[:,2], ax=ax3, num_bins=20, weights=UVLF_HST_model1[:,0], color=colors[3], zorder=3)
plot_hist2d(datax=UVLF_IllustrisTNG_model1[:,-2], datay=UVLF_IllustrisTNG_model1[:,2], ax=ax3, num_bins=20, weights=UVLF_IllustrisTNG_model1[:,0], color=colors[-1], zorder=2)
ax3.axhline(sigma8fid, color="black", alpha=0.5, ls=(1,(5.2,5)))
ax3.axvline(hfid, color="black", alpha=0.5, ls=(1,(5,5)))
ax3.set_xlabel(r'$h$', labelpad=10, fontsize=30)
ax3.set_xlim(0.64, 0.73)
ax3.set_ylim(0.35, 1.2)
ax3.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2])
ax3.set_yticklabels([r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$', r'$1.2$'])
ax3.set_xticks([0.64, 0.66, 0.68, 0.70, 0.72])
ax3.set_xticklabels([r'$0.64$', r'$0.66$', r'$0.68$', r'$0.70$', r'$0.72$'])
ax3.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))

fig.subplots_adjust(wspace=0.16)

plt.savefig("Posteriors_cosmo_model1.pdf")