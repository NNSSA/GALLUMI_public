import numpy as np
from matplotlib import pyplot as plt
plt.style.use("../template.mplstyle")

# purple - green - darkgoldenrod - blue - red
colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145', "#cf630a"]
linestyles = [(0, (1,1.05)), (0, (3, 1, 1, 1)), (0, (1,3)), (0, (3,3.65)), (0, (3,2.772)), (0, (3, 1, 1, 1, 1, 1))]

#########################################################################################

h = 0.6732
h_theory = 0.6736
TT = np.loadtxt("measurements_TT.txt")
EE = np.loadtxt("measurements_EE.txt")
phiphi = np.loadtxt("measurements_phiphi.txt")
WL = np.loadtxt("measurements_WL.txt")
Reid = np.loadtxt("measurements_Reid.txt")
Lya = np.loadtxt("measurements_LyA.txt")
theory = np.loadtxt("pk_theory.txt", unpack=True)

TT[:3] *= h
TT[3:] /= h**3
EE[:3] *= h
EE[3:] /= h**3
phiphi[:3] *= h
phiphi[3:] /= h**3
WL[:3] *= h
WL[3:] /= h**3
Reid[0] *= h
Reid[1:] /= h**3
Lya[0] *= h
Lya[1:] /= h**3
theory[0] *= h_theory
theory[1] /= h_theory**3

UVLF_2bins_1 = np.array([1.06, 67.09780887180366])
UVLF_2bins_1_xerror = np.array([[1.06-0.5], [2.25-1.06]])
UVLF_2bins_1_yerror = np.array([[14.456526986007667], [30.59404548201624]])

UVLF_2bins_2 = np.array([4.7, 1.2508772017466772])
UVLF_2bins_2_xerror = np.array([[4.7-2.25], [10-4.7]])
UVLF_2bins_2_yerror = np.array([[0.3417109173832944], [0.7688495641124118]])

#########################################################################################

plt.figure(figsize=(8.,6.5))
ax1 = plt.subplot(111)
ax1.tick_params(axis='x', which='major', pad=6)
ax1.tick_params(axis='both', which='major', labelsize=25)
ax1.tick_params(axis='both', which='minor', labelsize=25)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(2.2)

ax1.plot(theory[0], theory[1], color="#363636", lw=2., zorder=-7)

ax1.errorbar(TT[0], TT[3], xerr=np.array((TT[1], TT[2])), yerr=TT[4], ls="None", marker=".", markersize=5, markeredgewidth=1, elinewidth=1.2, color=colors[3], label=r"$\mathrm{Planck\ 2018\ TT}$", alpha=0.5)

ax1.errorbar(EE[0], EE[3], xerr=np.array((EE[1], EE[2])), yerr=EE[4], ls="None", marker=".", markersize=5, markeredgewidth=1, elinewidth=1.2, color=colors[1], label=r"$\mathrm{Planck\ 2018\ EE}$", alpha=0.5)

ax1.errorbar(phiphi[0], phiphi[3], xerr=np.array((phiphi[1], phiphi[2])), yerr=phiphi[4], ls="None", marker=".", markersize=5, markeredgewidth=1, elinewidth=1.2, color=colors[0], label=r"$\mathrm{Planck\ 2018\ \phi\phi}$", alpha=0.5)

ax1.errorbar(WL[0], WL[3], xerr=np.array((WL[1],WL[2])), yerr=WL[4], ls="None", marker=".", markersize=5, markeredgewidth=1, elinewidth=1.2, color=colors[4], label=r"$\mathrm{DES\ Y1\ cosmic\ shear}$", alpha=0.5)

ax1.errorbar(Reid[0], Reid[1], yerr=Reid[2], ls="None", marker=".", markersize=5, markeredgewidth=1, elinewidth=1.2, color=colors[5], label=r"$\mathrm{SDSS\ DR7\ LRG}$", zorder=-6, alpha=0.5)

ax1.errorbar(Lya[0], Lya[1], yerr=Lya[2], ls="None", marker=".", markersize=5, markeredgewidth=1, elinewidth=1.2, color=colors[2], label=r"$\mathrm{eBOSS\ DR14\ Ly\hspace{-1.2mm}-\hspace{-1.2mm}\alpha\ forest}$", zorder=-6, alpha=0.5)

ax1.errorbar(UVLF_2bins_1[0], UVLF_2bins_1[1], xerr=UVLF_2bins_1_xerror, yerr=UVLF_2bins_1_yerror, ls="None", marker=".", markersize=5, markeredgewidth=1.5, elinewidth=1.5, color="black", label=r"$\mathrm{UV\ LF\ (this\ work)}$", zorder=0)

ax1.errorbar(UVLF_2bins_2[0], UVLF_2bins_2[1], xerr=UVLF_2bins_2_xerror, yerr=UVLF_2bins_2_yerror, ls="None", marker=".", markersize=5, markeredgewidth=1.5, elinewidth=1.5, color="black", zorder=0)

ax1.set_ylabel(r'$P(k)\ [\mathrm{Mpc}^3]$', fontsize=27)
ax1.set_xlabel(r'$k\ \mathrm{[}\mathrm{Mpc^{-1}]}$', fontsize=27)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.axis(xmin=1e-4, xmax=20., ymin=1e-1, ymax=2e5)

ax1.legend(loc="lower center", bbox_to_anchor=(0.42, 0.05), frameon=False, markerfirst=True, prop={'size': 17.5}, handlelength=1.9, handletextpad=0.5, numpoints=1)

plt.savefig("Power_spectrum_plot.pdf")
