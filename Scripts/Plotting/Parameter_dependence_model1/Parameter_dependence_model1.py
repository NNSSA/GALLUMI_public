import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
plt.style.use("../template.mplstyle")

# purple - green - darkgoldenrod - blue - red
colors = ['#9467bd', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145']

########################################################################

Phialphastar = np.loadtxt("UVLF_dep_alphastar.txt")
Phibetastar = np.loadtxt("UVLF_dep_betastar.txt")
Phiepsilonstar = np.loadtxt("UVLF_dep_epsilonstar.txt")
PhiMc = np.loadtxt("UVLF_dep_Mc.txt")
Phisigma8 = np.loadtxt("UVLF_dep_sigma8.txt")
PhiOmegam = np.loadtxt("UVLF_dep_Omegam.txt")
Phins = np.loadtxt("UVLF_dep_ns.txt")

alphastar_max = max(Phialphastar[1:,0])
alphastar_min = min(Phialphastar[1:,0])
betastar_max = max(Phibetastar[1:,0])
betastar_min = min(Phibetastar[1:,0])
epsilonstar_max = max(Phiepsilonstar[1:,0])
epsilonstar_min = min(Phiepsilonstar[1:,0])
sigma8_max = max(Phisigma8[1:,0])
sigma8_min = min(Phisigma8[1:,0])
Omegam_max = max(PhiOmegam[1:,0])
Omegam_min = min(PhiOmegam[1:,0])

########################################################################

def hex_to_rgb(value):
    value = value.strip("#")
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

########################################################################

plt.figure(figsize=(18,8))
plt.gca().set_aspect('equal', adjustable='box')

ax1 = plt.subplot(241)
ax2 = plt.subplot(242)
ax3 = plt.subplot(243)
ax4 = plt.subplot(244)
ax5 = plt.subplot(245)
ax6 = plt.subplot(246)
ax7 = plt.subplot(247)
ax8 = plt.subplot(248)

ax1.tick_params(axis='both', which='both', labelsize=22)
ax2.tick_params(axis='both', which='both', labelsize=22)
ax3.tick_params(axis='both', which='both', labelsize=22)
ax4.tick_params(axis='both', which='both', labelsize=22)
ax5.tick_params(axis='both', which='both', labelsize=22)
ax6.tick_params(axis='both', which='both', labelsize=22)
ax7.tick_params(axis='both', which='both', labelsize=22)
ax8.tick_params(axis='both', which='both', labelsize=22)

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.8)

plt.subplots_adjust(wspace=0.07,hspace=0.08)

###################################################################################################

for i in range(len(Phisigma8[1:])):
    ax1.plot(Phisigma8[0,1:], Phisigma8[i+1,1:])

colormapax1 = get_continuous_cmap(['#FFFFFF', colors[-1]], float_list=[0, 1]) # plt.cm.Oranges
normalizeax1 = matplotlib.colors.TwoSlopeNorm(vcenter=0.4, vmin=-0.1, vmax=0.9)
smapax1 = matplotlib.cm.ScalarMappable(norm=normalizeax1, cmap=colormapax1)
colorssigma8 = [smapax1.to_rgba(i) for i in np.linspace(0, 1,len(ax1.lines))]
for i,j in enumerate(ax1.lines):
    j.set_color(colorssigma8[i])

ax1.semilogy()
ax1.axis(xmin=-24, xmax=-16, ymin=1e-12, ymax=1)
ax1.set_ylabel(r"$\Phi_\mathrm{UV}$", labelpad=12, fontsize=26)
ax1.text(0.655, 0.35, r'$\sigma_8$', fontsize=24, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
divider = make_axes_locatable(ax1)
cbar = plt.colorbar(smapax1, cax=ax1.inset_axes((0.48, 0.22, 0.35, 0.05)), orientation='horizontal')
cbar.set_ticks([-0.1, 0.9])
cbar.ax.tick_params(labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticklabels([r"$0.5$", r"$1.5$"])
cbar.outline.set_linewidth(1.5)

###################################################################################################

for i in range(len(PhiOmegam[1:])):
    ax2.plot(PhiOmegam[0,1:], PhiOmegam[i+1,1:])

colormapax2 = get_continuous_cmap(['#FFFFFF', "#D56808"], float_list=[0, 1])
normalizeax2 = matplotlib.colors.TwoSlopeNorm(vcenter=0.5, vmin=-0.1, vmax=0.9)
smapax2 = matplotlib.cm.ScalarMappable(norm=normalizeax2, cmap=colormapax2)
colorsOmegam = [smapax2.to_rgba(i) for i in np.linspace(0, 1,len(ax2.lines))]
for i,j in enumerate(ax2.lines):
    j.set_color(colorsOmegam[i])

ax2.semilogy()
ax2.axis(xmin=-24, xmax=-16, ymin=1e-12, ymax=1)
ax2.text(0.655, 0.35, r'$\Omega_\mathrm{m}$', fontsize=24, horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
divider = make_axes_locatable(ax2)
cbar = plt.colorbar(smapax2, cax=ax2.inset_axes((0.48, 0.22, 0.35, 0.05)), orientation='horizontal')
cbar.set_ticks([-0.1, 0.9])
cbar.ax.tick_params(labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticklabels([r"$0.06$", r"$1$"])

###################################################################################################

for i in range(len(Phins[1:])):
    ax3.plot(Phins[0,1:], Phins[i+1,1:])

colormapax3 = get_continuous_cmap(['#FFFFFF', "#dba82a"], float_list=[0, 1])
normalizeax3 = matplotlib.colors.TwoSlopeNorm(vcenter=0.4, vmin=-0.1, vmax=0.9)
smapax3 = matplotlib.cm.ScalarMappable(norm=normalizeax3, cmap=colormapax3)
colorsns = [smapax3.to_rgba(i) for i in np.linspace(0, 1,len(ax3.lines))]
for i,j in enumerate(ax3.lines):
    j.set_color(colorsns[i])

ax3.semilogy()
ax3.axis(xmin=-24, xmax=-16, ymin=1e-12, ymax=1)
ax3.text(0.655, 0.35, r'$n_\mathrm{s}$', fontsize=24, horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
divider = make_axes_locatable(ax3)
cbar = plt.colorbar(smapax3, cax=ax3.inset_axes((0.48, 0.22, 0.35, 0.05)), orientation='horizontal')
cbar.set_ticks([-0.1, 0.9])
cbar.ax.tick_params(labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticklabels([r"$0.5$", r"$1.5$"])

###################################################################################################

for i in range(len(Phialphastar[1:])):
    ax4.plot(Phialphastar[0,1:], Phialphastar[i+1,1:])

colormapax4 = get_continuous_cmap(['#FFFFFF', "#6e9773"], float_list=[0, 1])
normalizeax4 = matplotlib.colors.TwoSlopeNorm(vcenter=0.55, vmin=-0.1, vmax=0.9)
smapax4 = matplotlib.cm.ScalarMappable(norm=normalizeax4, cmap=colormapax4)
colorsalpha = [smapax4.to_rgba(i) for i in np.linspace(0, 1,len(ax4.lines))]
for i,j in enumerate(ax4.lines):
    j.set_color(colorsalpha[i])

ax4.semilogy()
ax4.axis(xmin=-24, xmax=-16, ymin=1e-12, ymax=1)
ax4.text(0.655, 0.35, r'$\alpha_*$', fontsize=24, horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
divider = make_axes_locatable(ax4)
cbar = plt.colorbar(smapax4, cax=ax4.inset_axes((0.48, 0.22, 0.35, 0.05)), orientation='horizontal')
cbar.set_ticks([-0.1, 0.9])
cbar.ax.tick_params(labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticklabels([r"$-3$", r"$0$"])

###################################################################################################

for i in range(len(Phibetastar[1:])):
    ax5.plot(Phibetastar[0,1:], Phibetastar[i+1,1:])

colormapax5 = get_continuous_cmap(['#FFFFFF', "#12a38c"], float_list=[0, 1])
normalizeax5 = matplotlib.colors.TwoSlopeNorm(vmin=-0.1, vcenter=0.4, vmax=0.9)
smapax5 = matplotlib.cm.ScalarMappable(norm=normalizeax5, cmap=colormapax5)
colorsbeta = [smapax5.to_rgba(i) for i in np.linspace(0, 1,len(ax5.lines))]
for i,j in enumerate(ax5.lines):
    j.set_color(colorsbeta[i])

ax5.semilogy()
ax5.axis(xmin=-24, xmax=-16, ymin=1e-12, ymax=1)
ax5.set_ylabel(r"$\Phi_\mathrm{UV}$", labelpad=12, fontsize=26)
ax5.set_xlabel(r"$M_\mathrm{UV}$", labelpad=14, fontsize=26)
ax5.text(0.655, 0.35, r'$\beta_*$', fontsize=24, horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
divider = make_axes_locatable(ax5)
cbar = plt.colorbar(smapax5, cax=ax5.inset_axes((0.48, 0.22, 0.35, 0.05)), orientation='horizontal')
cbar.set_ticks([-0.1, 0.9])
cbar.ax.tick_params(labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticklabels([r"$0$", r"$3$"])

###################################################################################################

for i in range(len(Phiepsilonstar[1:])):
    ax6.plot(Phiepsilonstar[0,1:], Phiepsilonstar[i+1,1:])

colormapax6 = get_continuous_cmap(['#FFFFFF', colors[-2]], float_list=[0, 1]) 
normalizeax6 = matplotlib.colors.TwoSlopeNorm(vcenter=0.6, vmin=-0.1, vmax=0.9)
smapax6 = matplotlib.cm.ScalarMappable(norm=normalizeax6, cmap=colormapax6)
colorsepsilon = [smapax6.to_rgba(i) for i in np.linspace(0, 1,len(ax6.lines))]
for i,j in enumerate(ax6.lines):
    j.set_color(colorsepsilon[i])

ax6.semilogy()
ax6.axis(xmin=-24, xmax=-16, ymin=1e-12, ymax=1)
ax6.set_xlabel(r"$M_\mathrm{UV}$", labelpad=14, fontsize=26)
ax6.text(0.655, 0.35, r'$\epsilon_*$', fontsize=24, horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)
divider = make_axes_locatable(ax6)
cbar = plt.colorbar(smapax6, cax=ax6.inset_axes((0.48, 0.22, 0.35, 0.05)), orientation='horizontal')
cbar.set_ticks([-0.1, 0.9])
cbar.ax.tick_params(labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticklabels([r"$10^{-3}$", r"$10^{-1}$"])

###################################################################################################

for i in range(len(PhiMc[1:])):
    ax7.plot(PhiMc[0,1:], PhiMc[i+1,1:])

colormapax7 = get_continuous_cmap(['#FFFFFF', colors[0]], float_list=[0, 1])
normalizeax7 = matplotlib.colors.TwoSlopeNorm(vcenter=0.6, vmin=-0.1, vmax=0.9)
smapax7 = matplotlib.cm.ScalarMappable(norm=normalizeax7, cmap=colormapax7)
colorsMc = [smapax7.to_rgba(i) for i in np.linspace(0, 1,len(ax7.lines))]
for i,j in enumerate(ax7.lines):
    j.set_color(colorsMc[i])

ax7.semilogy()
ax7.axis(xmin=-24, xmax=-16, ymin=1e-12, ymax=1)
ax7.set_xlabel(r"$M_\mathrm{UV}$", labelpad=14, fontsize=26)
ax7.text(0.655, 0.35, r'$M_\mathrm{c}\ [M_\odot]$', fontsize=24, horizontalalignment='center', verticalalignment='center', transform=ax7.transAxes)
divider = make_axes_locatable(ax7)
cbar = plt.colorbar(smapax7, cax=ax7.inset_axes((0.48, 0.22, 0.35, 0.05)), orientation='horizontal')
cbar.set_ticks([-0.1, 0.9])
cbar.ax.tick_params(labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticklabels([r"$10^{9}$", r"$10^{13}$"])

###################################################################################################

ax8.fill_between(np.append(Phisigma8[0,1:], Phisigma8[0,1:][::-1]), np.append(Phisigma8[0+1,1:], Phisigma8[-1,1:][::-1]), facecolor="none", color=colors[-1], linewidth=0.0, alpha=0.2)
ax8.fill_between(np.append(Phiepsilonstar[0,1:], Phiepsilonstar[0,1:][::-1]), np.append(Phiepsilonstar[0+1,1:], Phiepsilonstar[-1,1:][::-1]), facecolor="none", color=colors[-2], linewidth=0.0, alpha=0.2)

for i in [0, 25, 50, 75, 99]:
    ax8.plot(Phiepsilonstar[0,1:], Phiepsilonstar[i+1,1:], color=colors[-2], lw=1.5)
for i in [0, 8, 20, 45, 99]:
    ax8.plot(Phisigma8[0,1:], Phisigma8[i+1,1:], color=colors[-1], lw=1.5, ls=(1,(3,1)))

ax8.semilogy()
ax8.axis(xmin=-24, xmax=-16, ymin=1e-12, ymax=1)
ax8.set_xlabel(r"$M_\mathrm{UV}$", labelpad=14, fontsize=26)

###################################################################################################

ax1.xaxis.set_ticklabels([])
ax2.xaxis.set_ticklabels([])
ax3.xaxis.set_ticklabels([])
ax4.xaxis.set_ticklabels([])
ax2.yaxis.set_ticklabels([])
ax3.yaxis.set_ticklabels([])
ax4.yaxis.set_ticklabels([])
ax6.yaxis.set_ticklabels([])
ax7.yaxis.set_ticklabels([])
ax8.yaxis.set_ticklabels([])

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    ax.set_yticks([1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
    ax.set_xticks([-23, -21, -19, -17])

plt.savefig("Parameter_dependence_model1.pdf")
