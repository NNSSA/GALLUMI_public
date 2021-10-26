import numpy as np
from matplotlib import pyplot as plt
import glob
from matplotlib import patches as mpatches
import scipy.ndimage
plt.style.use("../template.mplstyle")

# purple - green - darkgoldenrod - blue - red
colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145']
linestyles = [(0, (1,1.05)), (0, (3, 1, 1, 1)), (0, (1,3)), (0, (3,3.65)), (0, (3,2.772)), (0, (3, 1, 1, 1, 1, 1))]

#########################################################################################

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

chains = []
chains_BOSS = []
samples_shear = []
samples_Planck = []

for filepath in glob.iglob('../../Data/UVLF_HST_ST_model1/*__*.txt'):
    data = np.loadtxt(filepath)
    chains.append(data)
for filepath in glob.iglob('../../Data/LCDM_boss_NOsne_kmax020_newW/*__*.txt'):
    data = np.loadtxt(filepath)
    chains_BOSS.append(data)
for filepath in glob.iglob('../../Data/DESKIDS/*kv450desy1_*.txt'):
    data = np.loadtxt(filepath)
    samples_shear.append(data)
for filepath in glob.iglob('../../Data/Planck2018/*lowE_*.txt'):
    data = np.loadtxt(filepath)
    samples_Planck.append(data)

chains = np.vstack(np.array(chains))
chains_BOSS = np.vstack(np.array(chains_BOSS))
samples_shear = np.vstack(np.array(samples_shear))
samples_Planck = np.vstack(np.array(samples_Planck))

plt.figure(figsize=(8.,6.5))
ax1 = plt.subplot(111)
ax1.tick_params(axis='x', which='major', pad=6)
ax1.tick_params(axis='both', which='major', labelsize=25)
ax1.tick_params(axis='both', which='minor', labelsize=25)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(2.2)

###############

plot_hist2d(datax=chains[:,16], datay=chains[:,2], ax=ax1, num_bins=20, weights=chains[:,0], color=colors[3], zorder=1)
plot_hist2d(datax=chains_BOSS[:,-3], datay=chains_BOSS[:,-1], ax=ax1, num_bins=20, weights=chains_BOSS[:,0], color=colors[-1], zorder=2)
plot_hist2d(datax=samples_shear[:,29], datay=samples_shear[:,33], ax=ax1, num_bins=20, weights=samples_shear[:,0], color=colors[1], zorder=0)
plot_hist2d(datax=samples_Planck[:,31], datay=samples_Planck[:,35], ax=ax1, num_bins=20, weights=samples_Planck[:,0], color=colors[2], zorder=3)
ax1.set_xlabel(r'$\Omega_\mathrm{m}$', labelpad=10, fontsize=27)
ax1.set_ylabel(r'$\sigma_8$', labelpad=12, fontsize=27)
ax1.set_xlim(0.2, 0.4)
ax1.set_ylim(0.4, 1.2)

patch_blue = mpatches.Patch(color=adjust_lightness(colors[3],1.3), lw=1.5, label=r"$\mathrm{UV\, LF\, +\, SNe\, +\, BBN}$")
patch_red = mpatches.Patch(color=adjust_lightness(colors[-1],1.1), lw=1.5, label=r"$\mathrm{FS\ Galaxy\ PS\ +\ BAO}$")
patch_green = mpatches.Patch(color=adjust_lightness(colors[1],1.8), lw=1.5, label=r"$\mathrm{Cosmic\, shear}$")
patch_yellow = mpatches.Patch(color=adjust_lightness(colors[2],1.4), lw=1.5, label=r"$\mathrm{CMB}$")
leg = ax1.legend(handles=[patch_blue, patch_red, patch_green, patch_yellow], loc="upper center", frameon=False, markerfirst=True, prop={'size': 16}, handlelength=1.9, handletextpad=0.5, numpoints=1, ncol=2)

plt.savefig("Posteriors_sigma8_Omegam.pdf")
