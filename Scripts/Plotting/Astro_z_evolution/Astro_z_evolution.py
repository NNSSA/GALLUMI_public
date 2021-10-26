import numpy as np
from matplotlib import pyplot as plt
import matplotlib
plt.style.use("../template.mplstyle")
# purple - green - darkgoldenrod - blue - red
colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145']

##################################################################

data_Hubble = np.loadtxt("data.txt", unpack=True)

redshifts = np.array([4,5,6,7,8,9,10])

alphastar_Hubble_bestfit = data_Hubble[0,:7]
alphastar_Hubble_error = data_Hubble[1:,:7]
betastar_Hubble_bestfit = data_Hubble[0,7:14]
betastar_Hubble_error = data_Hubble[1:,7:14]
epsilonstar_Hubble_bestfit = data_Hubble[0,14:21]
epsilonstar_Hubble_error = data_Hubble[1:,14:21]
Mc_Hubble_bestfit = data_Hubble[0,21:28]
Mc_Hubble_error = data_Hubble[1:,21:28]

plt.figure(figsize=(14,11.5))
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)
for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(2)
    ax2.spines[axis].set_linewidth(2)
    ax3.spines[axis].set_linewidth(2)
    ax4.spines[axis].set_linewidth(2)

ax1.tick_params(axis='x', which='major', pad=6)
ax1.tick_params(axis='both', which='major', labelsize=25)
ax1.tick_params(axis='both', which='minor', labelsize=25)
ax2.tick_params(axis='x', which='major', pad=6)
ax2.tick_params(axis='both', which='major', labelsize=25)
ax2.tick_params(axis='both', which='minor', labelsize=25)
ax3.tick_params(axis='x', which='major', pad=6)
ax3.tick_params(axis='both', which='major', labelsize=25)
ax3.tick_params(axis='both', which='minor', labelsize=25)
ax4.tick_params(axis='x', which='major', pad=6)
ax4.tick_params(axis='both', which='major', labelsize=25)
ax4.tick_params(axis='both', which='minor', labelsize=25)

ax1.xaxis.set_ticklabels([])
ax2.xaxis.set_ticklabels([])

# 

alpha_upper = -0.48347039883049725
alpha_lower = -0.7687000545228562
beta_upper = 0.9420514115994328
beta_lower = 0.34895363153036374
eps_s_1 = -0.375
eps_s_2 = -3.
eps_i_1 = -1.75
eps_i_2 = -2.58
Mc_s_1 = 1.3
Mc_s_2 = 3.
Mc_i_1 = 11.66
Mc_i_2 = 12.45

redshifts2 = np.linspace(2,12,40)
ax1.scatter(redshifts, alphastar_Hubble_bestfit, color="black", s=10, marker="o", zorder=4)
ax1.errorbar(redshifts, alphastar_Hubble_bestfit, yerr=alphastar_Hubble_error, color="black", ls="none", zorder=4)
ax1.plot(redshifts2, np.repeat(-0.656134423535443, len(redshifts2)), color=colors[-1])
ax1.plot(redshifts2, np.repeat(alpha_lower, len(redshifts2)), color=colors[-1], alpha=0.5, lw=0.5)
ax1.plot(redshifts2, np.repeat(alpha_upper, len(redshifts2)), color=colors[-1], alpha=0.5, lw=0.5)
ax1.fill_between(redshifts2, alpha_lower, alpha_upper, color=colors[-1], alpha=0.3)
ax1.legend()

lolims = np.zeros(redshifts.shape)
lolims[:] = True
ax2.errorbar(redshifts, betastar_Hubble_bestfit, yerr=0.09, lolims=lolims, color="black", ls="none", markeredgewidth=2., elinewidth=1.5, zorder=4)

ax2.plot(redshifts2, np.repeat(0.9271467947570028, len(redshifts2)), color=colors[1])
ax2.plot(redshifts2, np.repeat(beta_lower, len(redshifts2)), color=colors[1], alpha=0.5, lw=0.5)
ax2.plot(redshifts2, np.repeat(beta_upper, len(redshifts2)), color=colors[1], alpha=0.5, lw=0.5)
ax2.fill_between(redshifts2, beta_lower, beta_upper, color=colors[1], alpha=0.3)

ax3.scatter(redshifts[:-1], epsilonstar_Hubble_bestfit[:-1], color="black", s=10, marker="o", zorder=4)
ax3.errorbar(redshifts[:-1], epsilonstar_Hubble_bestfit[:-1], yerr=epsilonstar_Hubble_error[:,:-1], color="black", ls="none", zorder=4)
ax3.errorbar(redshifts[-1], epsilonstar_Hubble_bestfit[-1], yerr=0.19, uplims=True, color="black", ls="none", markeredgewidth=2., elinewidth=1.5, zorder=4)
ax3.plot(redshifts2, -1.449079647880109 * (np.log10(redshifts2+1) - np.log10(6+1)) - 2.083036101416093, color=colors[3])
ax3.plot(redshifts2, eps_s_1 * (np.log10(redshifts2+1) - np.log10(6+1)) + eps_i_1, color=colors[3], alpha=0.5, lw=0.5)
ax3.plot(redshifts2, eps_s_2 * (np.log10(redshifts2+1) - np.log10(6+1)) + eps_i_2, color=colors[3], alpha=0.5, lw=0.5)
ax3.fill_between(redshifts2, eps_s_2 * (np.log10(redshifts2+1) - np.log10(6+1)) + eps_i_2, eps_s_1 * (np.log10(redshifts2+1) - np.log10(6+1)) + eps_i_1, color=colors[3], alpha=0.3)

ax4.scatter(redshifts[:-3], Mc_Hubble_bestfit[:-3], color="black", s=10, marker="o", zorder=4)
ax4.errorbar(redshifts[:-3], Mc_Hubble_bestfit[:-3], yerr=Mc_Hubble_error[:,:-3], color="black", ls="none", zorder=4)
ax4.errorbar(redshifts[-3:], Mc_Hubble_bestfit[-3:], yerr=0.22, lolims=True, color="black", ls="none", markeredgewidth=2., elinewidth=1.5, zorder=4)
ax4.plot(redshifts2, 2.005383762817394 * (np.log10(redshifts2+1) - np.log10(6+1)) + 11.899973904827537, color=colors[2])
ax4.plot(redshifts2, Mc_s_1 * (np.log10(redshifts2+1) - np.log10(6+1)) + Mc_i_1, color=colors[2], alpha=0.5, lw=0.5)
ax4.plot(redshifts2, Mc_s_2 * (np.log10(redshifts2+1) - np.log10(6+1)) + Mc_i_2, color=colors[2], alpha=0.5, lw=0.5)
ax4.fill_between(redshifts2, Mc_s_1 * (np.log10(redshifts2+1) - np.log10(6+1)) + Mc_i_1, Mc_s_2 * (np.log10(redshifts2+1) - np.log10(6+1)) + Mc_i_2, color=colors[2], alpha=0.3)

ax1.set_ylabel(r"$\alpha_*$", labelpad=8, fontsize=28)
ax1.axis(xmin=3.5, xmax=10.5, ymin=-1.8, ymax=0.)
ax2.set_ylabel(r"$\beta_*$", labelpad=8, fontsize=28)
ax2.axis(xmin=3.5, xmax=10.5, ymin=0., ymax=2.)
ax3.set_ylabel(r"$\mathrm{log}_{10}\epsilon_*$", labelpad=10, fontsize=28)
ax3.set_xlabel(r"$z$", labelpad=4, fontsize=28)
ax3.axis(xmin=3.5, xmax=10.5, ymin=-4, ymax=0)
ax4.set_ylabel(r"$\mathrm{log}_{10}\left(M_c/M_\odot\right)$", labelpad=8, fontsize=28)
ax4.set_xlabel(r"$z$", labelpad=4, fontsize=28)
ax4.axis(xmin=3.5, xmax=10.5, ymin=10., ymax=15)

ax1.set_yticks([-1.6, -1.2, -0.8, -0.4, 0.])
ax1.set_yticklabels([r'$-1.6$', r'$-1.2$', r'$-0.8$', r'$-0.4$', r'$0.0$'])
ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))

ax3.set_yticks([-4., -3., -2., -1., 0.])
ax3.set_yticklabels([r'$-4$', r'$-3$', r'$-2$', r'$-1$', r'$0$'])
ax3.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))

plt.subplots_adjust(wspace=0.25,hspace=0.1)

plt.savefig("Astro_z_evolution.pdf")
