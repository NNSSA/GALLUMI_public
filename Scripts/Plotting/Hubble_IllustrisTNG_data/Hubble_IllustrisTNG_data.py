import numpy as np
from matplotlib import pyplot as plt
plt.style.use("../template.mplstyle")

# purple - green - darkgoldenrod - blue - red
colors = ['purple', '#3F7BB6', "#12a38c", "#6e9773", 'darkgoldenrod', "#cf630a", '#BF4145']
# colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145', "#cf630a", "black"]
linestyles = [(0, (1,1.05)), (0, (3, 1, 1, 1)), (0, (1,3)), (0, (3,3.65)), (0, (3,2.772)), (0, (3, 1, 1, 1, 1, 1))]

#########################################################################################

fits_Hubble_model1 = np.loadtxt("model1_bestfit.txt")
data_Hubble_model1 = np.loadtxt("model1_bestfit_data.txt")
fits_Hubble_model2 = np.loadtxt("model2_bestfit.txt")
data_Hubble_model2 = np.loadtxt("model2_bestfit_data.txt")
fits_Hubble_model3 = np.loadtxt("model3_bestfit.txt")
data_Hubble_model3 = np.loadtxt("model3_bestfit_data.txt")
fits_TNG = np.loadtxt("model1_TNG_bestfit.txt")
data_TNG = np.loadtxt("model1_TNG_bestfit_data.txt")

plt.figure(figsize=(16.,13.))
ax_Hubble_model1 = plt.subplot(221)
ax_Hubble_model2 = plt.subplot(222)
ax_Hubble_model3 = plt.subplot(223)
ax_TNG_model1 = plt.subplot(224)
ax_Hubble_model1.tick_params(axis='x', which='major', pad=6)
ax_Hubble_model2.tick_params(axis='x', which='major', pad=6)
ax_Hubble_model3.tick_params(axis='x', which='major', pad=6)
ax_TNG_model1.tick_params(axis='x', which='major', pad=6)
ax_Hubble_model1.tick_params(axis='both', which='major', labelsize=25)
ax_Hubble_model1.tick_params(axis='both', which='minor', labelsize=25)
ax_Hubble_model2.tick_params(axis='both', which='major', labelsize=25)
ax_Hubble_model2.tick_params(axis='both', which='minor', labelsize=25)
ax_Hubble_model3.tick_params(axis='both', which='major', labelsize=25)
ax_Hubble_model3.tick_params(axis='both', which='minor', labelsize=25)
ax_TNG_model1.tick_params(axis='both', which='major', labelsize=25)
ax_TNG_model1.tick_params(axis='both', which='minor', labelsize=25)

for axis in ['top','bottom','left','right']:
    ax_Hubble_model1.spines[axis].set_linewidth(2.2)
    ax_Hubble_model2.spines[axis].set_linewidth(2.2)
    ax_Hubble_model3.spines[axis].set_linewidth(2.2)
    ax_TNG_model1.spines[axis].set_linewidth(2.2)

ax_Hubble_model1.plot(fits_Hubble_model1[fits_Hubble_model1[:,0]==4., 1], fits_Hubble_model1[fits_Hubble_model1[:,0]==4., 2], color=colors[0], lw=2., label=r"$z = 4$")
ax_Hubble_model1.errorbar(data_Hubble_model1[data_Hubble_model1[:,0]==4., 1], data_Hubble_model1[data_Hubble_model1[:,0]==4., 3], yerr=data_Hubble_model1[data_Hubble_model1[:,0]==4., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[0])

ax_Hubble_model1.plot(fits_Hubble_model1[fits_Hubble_model1[:,0]==5., 1], fits_Hubble_model1[fits_Hubble_model1[:,0]==5., 2], color=colors[1], lw=2., label=r"$z = 5$")
ax_Hubble_model1.errorbar(data_Hubble_model1[data_Hubble_model1[:,0]==5., 1], data_Hubble_model1[data_Hubble_model1[:,0]==5., 3], yerr=data_Hubble_model1[data_Hubble_model1[:,0]==5., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[1])

ax_Hubble_model1.plot(fits_Hubble_model1[fits_Hubble_model1[:,0]==6., 1], fits_Hubble_model1[fits_Hubble_model1[:,0]==6., 2], color=colors[2], lw=2., label=r"$z = 6$")
ax_Hubble_model1.errorbar(data_Hubble_model1[data_Hubble_model1[:,0]==6., 1], data_Hubble_model1[data_Hubble_model1[:,0]==6., 3], yerr=data_Hubble_model1[data_Hubble_model1[:,0]==6., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[2])

ax_Hubble_model1.plot(fits_Hubble_model1[fits_Hubble_model1[:,0]==7., 1], fits_Hubble_model1[fits_Hubble_model1[:,0]==7., 2], color=colors[3], lw=2., label=r"$z = 7$")
ax_Hubble_model1.errorbar(data_Hubble_model1[data_Hubble_model1[:,0]==7., 1], data_Hubble_model1[data_Hubble_model1[:,0]==7., 3], yerr=data_Hubble_model1[data_Hubble_model1[:,0]==7., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[3])

ax_Hubble_model1.plot(fits_Hubble_model1[fits_Hubble_model1[:,0]==8., 1], fits_Hubble_model1[fits_Hubble_model1[:,0]==8., 2], color=colors[4], lw=2., label=r"$z = 8$")
ax_Hubble_model1.errorbar(data_Hubble_model1[data_Hubble_model1[:,0]==8., 1], data_Hubble_model1[data_Hubble_model1[:,0]==8., 3], yerr=data_Hubble_model1[data_Hubble_model1[:,0]==8., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[4])

ax_Hubble_model1.plot(fits_Hubble_model1[fits_Hubble_model1[:,0]==9., 1], fits_Hubble_model1[fits_Hubble_model1[:,0]==9., 2], color=colors[5], lw=2., label=r"$z = 9$")
ax_Hubble_model1.errorbar(data_Hubble_model1[data_Hubble_model1[:,0]==9., 1], data_Hubble_model1[data_Hubble_model1[:,0]==9., 3], yerr=data_Hubble_model1[data_Hubble_model1[:,0]==9., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[5])

ax_Hubble_model1.plot(fits_Hubble_model1[fits_Hubble_model1[:,0]==10., 1], fits_Hubble_model1[fits_Hubble_model1[:,0]==10., 2], color=colors[6], lw=2., label=r"$z = 10$")
ax_Hubble_model1.errorbar(data_Hubble_model1[data_Hubble_model1[:,0]==10., 1], data_Hubble_model1[data_Hubble_model1[:,0]==10., 3], yerr=data_Hubble_model1[data_Hubble_model1[:,0]==10., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[6])

ax_Hubble_model1.set_ylabel(r"$\Phi_\mathrm{UV}\ [\mathrm{Mpc^{-3}\,mag^{-1}}]$", labelpad=8, fontsize=27)

ax_Hubble_model1.set_yscale('log')
ax_Hubble_model1.axis(xmin=-26, xmax=-16., ymin=1e-8, ymax=5e-2)

ax_Hubble_model1.legend(loc="lower right", frameon=False, markerfirst=False, prop={'size': 17}, handlelength=1.9, handletextpad=0.5, numpoints=1)

ax_Hubble_model1.set_xticks([-25., -23., -21., -19., -17.])
ax_Hubble_model1.axes.xaxis.set_ticklabels([])

ax_Hubble_model1.text(-25.5, 1.1e-2, r'$\mathbf{HST - model\ I}$', fontsize=22, color='k', rotation=0)

##############

ax_Hubble_model2.plot(fits_Hubble_model2[fits_Hubble_model2[:,0]==4., 1], fits_Hubble_model2[fits_Hubble_model2[:,0]==4., 2], color=colors[0], lw=2., label=r"$z = 4$")
ax_Hubble_model2.errorbar(data_Hubble_model2[data_Hubble_model2[:,0]==4., 1], data_Hubble_model2[data_Hubble_model2[:,0]==4., 3], yerr=data_Hubble_model2[data_Hubble_model2[:,0]==4., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[0])

ax_Hubble_model2.plot(fits_Hubble_model2[fits_Hubble_model2[:,0]==5., 1], fits_Hubble_model2[fits_Hubble_model2[:,0]==5., 2], color=colors[1], lw=2., label=r"$z = 5$")
ax_Hubble_model2.errorbar(data_Hubble_model2[data_Hubble_model2[:,0]==5., 1], data_Hubble_model2[data_Hubble_model2[:,0]==5., 3], yerr=data_Hubble_model2[data_Hubble_model2[:,0]==5., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[1])

ax_Hubble_model2.plot(fits_Hubble_model2[fits_Hubble_model2[:,0]==6., 1], fits_Hubble_model2[fits_Hubble_model2[:,0]==6., 2], color=colors[2], lw=2., label=r"$z = 6$")
ax_Hubble_model2.errorbar(data_Hubble_model2[data_Hubble_model2[:,0]==6., 1], data_Hubble_model2[data_Hubble_model2[:,0]==6., 3], yerr=data_Hubble_model2[data_Hubble_model2[:,0]==6., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[2])

ax_Hubble_model2.plot(fits_Hubble_model2[fits_Hubble_model2[:,0]==7., 1], fits_Hubble_model2[fits_Hubble_model2[:,0]==7., 2], color=colors[3], lw=2., label=r"$z = 7$")
ax_Hubble_model2.errorbar(data_Hubble_model2[data_Hubble_model2[:,0]==7., 1], data_Hubble_model2[data_Hubble_model2[:,0]==7., 3], yerr=data_Hubble_model2[data_Hubble_model2[:,0]==7., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[3])

ax_Hubble_model2.plot(fits_Hubble_model2[fits_Hubble_model2[:,0]==8., 1], fits_Hubble_model2[fits_Hubble_model2[:,0]==8., 2], color=colors[4], lw=2., label=r"$z = 8$")
ax_Hubble_model2.errorbar(data_Hubble_model2[data_Hubble_model2[:,0]==8., 1], data_Hubble_model2[data_Hubble_model2[:,0]==8., 3], yerr=data_Hubble_model2[data_Hubble_model2[:,0]==8., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[4])

ax_Hubble_model2.plot(fits_Hubble_model2[fits_Hubble_model2[:,0]==9., 1], fits_Hubble_model2[fits_Hubble_model2[:,0]==9., 2], color=colors[5], lw=2., label=r"$z = 9$")
ax_Hubble_model2.errorbar(data_Hubble_model2[data_Hubble_model2[:,0]==9., 1], data_Hubble_model2[data_Hubble_model2[:,0]==9., 3], yerr=data_Hubble_model2[data_Hubble_model2[:,0]==9., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[5])

ax_Hubble_model2.plot(fits_Hubble_model2[fits_Hubble_model2[:,0]==10., 1], fits_Hubble_model2[fits_Hubble_model2[:,0]==10., 2], color=colors[6], lw=2., label=r"$z = 10$")
ax_Hubble_model2.errorbar(data_Hubble_model2[data_Hubble_model2[:,0]==10., 1], data_Hubble_model2[data_Hubble_model2[:,0]==10., 3], yerr=data_Hubble_model2[data_Hubble_model2[:,0]==10., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[6])

ax_Hubble_model2.set_yscale('log')
ax_Hubble_model2.axis(xmin=-26, xmax=-16., ymin=1e-8, ymax=5e-2)

ax_Hubble_model2.axes.xaxis.set_ticklabels([])
ax_Hubble_model2.axes.yaxis.set_ticklabels([])
ax_Hubble_model2.set_xticks([-25., -23., -21., -19., -17.])

ax_Hubble_model2.text(-25.5, 1.1e-2, r'$\mathbf{HST - model\ II}$', fontsize=22, color='k', rotation=0)

##############

ax_Hubble_model3.plot(fits_Hubble_model3[fits_Hubble_model3[:,0]==4., 1], fits_Hubble_model3[fits_Hubble_model3[:,0]==4., 2], color=colors[0], lw=2., label=r"$z = 4$")
ax_Hubble_model3.errorbar(data_Hubble_model3[data_Hubble_model3[:,0]==4., 1], data_Hubble_model3[data_Hubble_model3[:,0]==4., 3], yerr=data_Hubble_model3[data_Hubble_model3[:,0]==4., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[0])

ax_Hubble_model3.plot(fits_Hubble_model3[fits_Hubble_model3[:,0]==5., 1], fits_Hubble_model3[fits_Hubble_model3[:,0]==5., 2], color=colors[1], lw=2., label=r"$z = 5$")
ax_Hubble_model3.errorbar(data_Hubble_model3[data_Hubble_model3[:,0]==5., 1], data_Hubble_model3[data_Hubble_model3[:,0]==5., 3], yerr=data_Hubble_model3[data_Hubble_model3[:,0]==5., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[1])

ax_Hubble_model3.plot(fits_Hubble_model3[fits_Hubble_model3[:,0]==6., 1], fits_Hubble_model3[fits_Hubble_model3[:,0]==6., 2], color=colors[2], lw=2., label=r"$z = 6$")
ax_Hubble_model3.errorbar(data_Hubble_model3[data_Hubble_model3[:,0]==6., 1], data_Hubble_model3[data_Hubble_model3[:,0]==6., 3], yerr=data_Hubble_model3[data_Hubble_model3[:,0]==6., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[2])

ax_Hubble_model3.plot(fits_Hubble_model3[fits_Hubble_model3[:,0]==7., 1], fits_Hubble_model3[fits_Hubble_model3[:,0]==7., 2], color=colors[3], lw=2., label=r"$z = 7$")
ax_Hubble_model3.errorbar(data_Hubble_model3[data_Hubble_model3[:,0]==7., 1], data_Hubble_model3[data_Hubble_model3[:,0]==7., 3], yerr=data_Hubble_model3[data_Hubble_model3[:,0]==7., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[3])

ax_Hubble_model3.plot(fits_Hubble_model3[fits_Hubble_model3[:,0]==8., 1], fits_Hubble_model3[fits_Hubble_model3[:,0]==8., 2], color=colors[4], lw=2., label=r"$z = 8$")
ax_Hubble_model3.errorbar(data_Hubble_model3[data_Hubble_model3[:,0]==8., 1], data_Hubble_model3[data_Hubble_model3[:,0]==8., 3], yerr=data_Hubble_model3[data_Hubble_model3[:,0]==8., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[4])

ax_Hubble_model3.plot(fits_Hubble_model3[fits_Hubble_model3[:,0]==9., 1], fits_Hubble_model3[fits_Hubble_model3[:,0]==9., 2], color=colors[5], lw=2., label=r"$z = 9$")
ax_Hubble_model3.errorbar(data_Hubble_model3[data_Hubble_model3[:,0]==9., 1], data_Hubble_model3[data_Hubble_model3[:,0]==9., 3], yerr=data_Hubble_model3[data_Hubble_model3[:,0]==9., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[5])

ax_Hubble_model3.plot(fits_Hubble_model3[fits_Hubble_model3[:,0]==10., 1], fits_Hubble_model3[fits_Hubble_model3[:,0]==10., 2], color=colors[6], lw=2., label=r"$z = 10$")
ax_Hubble_model3.errorbar(data_Hubble_model3[data_Hubble_model3[:,0]==10., 1], data_Hubble_model3[data_Hubble_model3[:,0]==10., 3], yerr=data_Hubble_model3[data_Hubble_model3[:,0]==10., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[6])

ax_Hubble_model3.set_xlabel(r"$M_\mathrm{UV}\ [\mathrm{mag}]$", labelpad=9, fontsize=27)
ax_Hubble_model3.set_ylabel(r"$\Phi_\mathrm{UV}\ [\mathrm{Mpc^{-3}\,mag^{-1}}]$", labelpad=8, fontsize=27)

ax_Hubble_model3.set_yscale('log')
ax_Hubble_model3.axis(xmin=-26, xmax=-16., ymin=1e-8, ymax=5e-2)
ax_Hubble_model3.set_xticks([-25., -23., -21., -19., -17.])
ax_Hubble_model3.set_xticklabels([r'$-25$', r'$-23$', r'$-21$', r'$-19$', r'$-17$'])

ax_Hubble_model3.text(-25.5, 1.1e-2, r'$\mathbf{HST - model\ III}$', fontsize=22, color='k', rotation=0)

##############

ax_TNG_model1.plot(fits_TNG[fits_TNG[:,0]==4., 1], fits_TNG[fits_TNG[:,0]==4., 2], color=colors[0], lw=2., label=r"$z = 4$")
ax_TNG_model1.errorbar(data_TNG[data_TNG[:,0]==4., 1], data_TNG[data_TNG[:,0]==4., 3], yerr=data_TNG[data_TNG[:,0]==4., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[0])

ax_TNG_model1.plot(fits_TNG[fits_TNG[:,0]==5., 1], fits_TNG[fits_TNG[:,0]==5., 2], color=colors[1], lw=2., label=r"$z = 5$")
ax_TNG_model1.errorbar(data_TNG[data_TNG[:,0]==5., 1], data_TNG[data_TNG[:,0]==5., 3], yerr=data_TNG[data_TNG[:,0]==5., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[1])

ax_TNG_model1.plot(fits_TNG[fits_TNG[:,0]==6., 1], fits_TNG[fits_TNG[:,0]==6., 2], color=colors[2], lw=2., label=r"$z = 6$")
ax_TNG_model1.errorbar(data_TNG[data_TNG[:,0]==6., 1], data_TNG[data_TNG[:,0]==6., 3], yerr=data_TNG[data_TNG[:,0]==6., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[2])

ax_TNG_model1.plot(fits_TNG[fits_TNG[:,0]==7., 1], fits_TNG[fits_TNG[:,0]==7., 2], color=colors[3], lw=2., label=r"$z = 7$")
ax_TNG_model1.errorbar(data_TNG[data_TNG[:,0]==7., 1], data_TNG[data_TNG[:,0]==7., 3], yerr=data_TNG[data_TNG[:,0]==7., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[3])

ax_TNG_model1.plot(fits_TNG[fits_TNG[:,0]==8., 1], fits_TNG[fits_TNG[:,0]==8., 2], color=colors[4], lw=2., label=r"$z = 8$")
ax_TNG_model1.errorbar(data_TNG[data_TNG[:,0]==8., 1], data_TNG[data_TNG[:,0]==8., 3], yerr=data_TNG[data_TNG[:,0]==8., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[4])

ax_TNG_model1.plot(fits_TNG[fits_TNG[:,0]==9., 1], fits_TNG[fits_TNG[:,0]==9., 2], color=colors[5], lw=2., label=r"$z = 9$")
ax_TNG_model1.errorbar(data_TNG[data_TNG[:,0]==9., 1], data_TNG[data_TNG[:,0]==9., 3], yerr=data_TNG[data_TNG[:,0]==9., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[5])

ax_TNG_model1.plot(fits_TNG[fits_TNG[:,0]==10., 1], fits_TNG[fits_TNG[:,0]==10., 2], color=colors[6], lw=2., label=r"$z = 10$")
ax_TNG_model1.errorbar(data_TNG[data_TNG[:,0]==10., 1], data_TNG[data_TNG[:,0]==10., 3], yerr=data_TNG[data_TNG[:,0]==10., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[6])

ax_TNG_model1.set_xlabel(r"$M_\mathrm{UV}\ [\mathrm{mag}]$", labelpad=9, fontsize=27)

ax_TNG_model1.set_yscale('log')
ax_TNG_model1.axis(xmin=-26, xmax=-16., ymin=1e-8, ymax=5e-2)

ax_TNG_model1.axes.yaxis.set_ticklabels([])
ax_TNG_model1.set_xticks([-25., -23., -21., -19., -17.])
ax_TNG_model1.set_xticklabels([r'$-25$', r'$-23$', r'$-21$', r'$-19$', r'$-17$'])

ax_TNG_model1.text(-25.5, 1.1e-2, r'$\mathbf{IllustrisTNG - model\ I}$', fontsize=22, color='k', rotation=0)

plt.subplots_adjust(wspace=0.042, hspace=0.05)

plt.savefig("Hubble_IllustrisTNG_data.pdf")