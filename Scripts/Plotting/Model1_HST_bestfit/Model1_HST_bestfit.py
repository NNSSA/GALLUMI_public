import numpy as np
from scipy.interpolate import PchipInterpolator
from matplotlib import pyplot as plt
plt.style.use("../template.mplstyle")

colors = ['purple', '#3F7BB6', "#12a38c", "#6e9773", 'darkgoldenrod', "#cf630a", '#BF4145']
linestyles = [(0, (1,1.05)), (0, (3, 1, 1, 1)), (0, (1,3)), (0, (3,3.65)), (0, (3,2.772)), (0, (3, 1, 1, 1, 1, 1))]

#########################################################################################

fits = np.loadtxt("model1_bestfit.txt")
data = np.loadtxt("model1_bestfit_data.txt")
Mh_MUV = np.loadtxt("Mh_MUV_bestfit_model1.txt")

plt.figure(figsize=(8.,6.5))
ax = plt.subplot(111)
ax.tick_params(axis='x', which='major', pad=6)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.tick_params(axis='both', which='minor', labelsize=25)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.2)

plt.plot(fits[fits[:,0]==4., 1], fits[fits[:,0]==4., 2], color=colors[0], lw=2., label=r"$z = 4$")
plt.errorbar(data[data[:,0]==4., 1], data[data[:,0]==4., 3], yerr=data[data[:,0]==4., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[0])

plt.plot(fits[fits[:,0]==5., 1], fits[fits[:,0]==5., 2], color=colors[1], lw=2., label=r"$z = 5$")
plt.errorbar(data[data[:,0]==5., 1], data[data[:,0]==5., 3], yerr=data[data[:,0]==5., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[1])

plt.plot(fits[fits[:,0]==6., 1], fits[fits[:,0]==6., 2], color=colors[2], lw=2., label=r"$z = 6$")
plt.errorbar(data[data[:,0]==6., 1], data[data[:,0]==6., 3], yerr=data[data[:,0]==6., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[2])

plt.plot(fits[fits[:,0]==7., 1], fits[fits[:,0]==7., 2], color=colors[3], lw=2., label=r"$z = 7$")
plt.errorbar(data[data[:,0]==7., 1], data[data[:,0]==7., 3], yerr=data[data[:,0]==7., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[3])

plt.plot(fits[fits[:,0]==8., 1], fits[fits[:,0]==8., 2], color=colors[4], lw=2., label=r"$z = 8$")
plt.errorbar(data[data[:,0]==8., 1], data[data[:,0]==8., 3], yerr=data[data[:,0]==8., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[4])

plt.plot(fits[fits[:,0]==9., 1], fits[fits[:,0]==9., 2], color=colors[5], lw=2., label=r"$z = 9$")
plt.errorbar(data[data[:,0]==9., 1], data[data[:,0]==9., 3], yerr=data[data[:,0]==9., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[5])

plt.plot(fits[fits[:,0]==10., 1], fits[fits[:,0]==10., 2], color=colors[6], lw=2., label=r"$z = 10$")
plt.errorbar(data[data[:,0]==10., 1], data[data[:,0]==10., 3], yerr=data[data[:,0]==10., 4], ls="None", marker=".", markersize=6, markeredgewidth=1.2, elinewidth=1.6, color=colors[6])

plt.xlabel(r"$M_\mathrm{UV}\ [\mathrm{mag}]$", labelpad=9, fontsize=27)
plt.ylabel(r"$\Phi_\mathrm{UV}\ [\mathrm{Mpc^{-3}\,mag^{-1}}]$", labelpad=8, fontsize=27)

plt.semilogy()
plt.axis(xmin=-26, xmax=-16., ymin=1e-8, ymax=1e-1)

leg = plt.legend(loc="lower right", frameon=False, markerfirst=False, prop={'size': 17}, handlelength=1.9, handletextpad=0.5, numpoints=1)

ax2 = ax.twiny()
new_tick_locations = np.array([-24.5, -21.88, -17.93])

length = 3000
interpolator = PchipInterpolator(Mh_MUV[2, length+1:][::-1], np.log(Mh_MUV[2, 1:length+1][::-1]))

def sci_notation(number, sig_fig=2):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a,b = ret_string.split("e")
    b = int(b)
    return r"${:.0f}\times 10^{}$".format(float(a),"{" + str(b) + "}")

def tick_function(X):
    V = [np.exp(interpolator(M)) for M in X]
    return [sci_notation(z, 1) for z in V]

ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations))
ax2.set_xlabel(r'$M_\mathrm{h}\ \mathrm{at}\ z = 6\ [M_\odot]$', labelpad=15, fontsize=27)
ax2.minorticks_off()
ax2.tick_params(axis='both', which='major', labelsize=24)
ax2.tick_params(axis='both', which='minor', labelsize=24)

plt.savefig("Model1_HST_bestfit.pdf")