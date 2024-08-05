import numpy as np
from classy import Class
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use("scientific")
colors = ['purple', '#306B37', 'darkgoldenrod', '#3F7BB6', '#BF4145', "#cf630a"]

cosmo = Class()
args = {'P_k_max_1/Mpc': 40., 'T_ncdm': 0.71611, 'N_ur': 2.0308, 'N_ncdm': 1, 'tau_reio': 0.0544, 'n_s': 0.9649 , 'k_pivot': 0.05, 'omega_b': 0.02236, 'm_ncdm': 0.06, 'h': 0.6727, 'z_max_pk': 10.0, 'output': 'mPk, dTk', 'omega_cdm': 0.1202, 'ln10^{10}A_s': 3.045}
cosmo.set(args)
cosmo.compute(['lensing'])

# The Gaussian bump added to the power spectrum
def Gaussian_bump(k, amp, mean, sigma):
    k1 = 0.5
    if k < k1:
        return 0.
    return amp * np.exp(-(np.log(k) - mean)**2 / (2. * sigma**2))

# Calculate the mass variance manually intead from CLASS
def pk(k_arr, pk, z, amp, mean, sigma):
    mps = np.vectorize(lambda y: pk(y, z) * (1. + Gaussian_bump(y, amp, mean, sigma)))
    return mps(k_arr)

#######################

UVLF_2bins_1 = np.array([1.06, 67.09780887180366])
UVLF_2bins_1_xerror = np.array([[1.06-0.5], [2.25-1.06]])
UVLF_2bins_1_yerror = np.array([[14.456526986007667], [30.59404548201624]])

UVLF_2bins_2 = np.array([4.7, 1.2508772017466772])
UVLF_2bins_2_xerror = np.array([[4.7-2.25], [10-4.7]])
UVLF_2bins_2_yerror = np.array([[0.3417109173832944], [0.7688495641124118]])

#######################

plt.figure(figsize=(8.7,7))
ax = plt.subplot(111)
ax.tick_params(axis='x', which='major', pad=6)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.tick_params(axis='both', which='minor', labelsize=25)

z = 8.
k_array = np.geomspace(1e-2, 20., 1000)
k_array2 = np.geomspace(2e-2, 0.5, 286)
k_array3 = np.geomspace(6.5, 60., 250)
offsets = np.linspace(-10, 10, 150)
offsets2 = np.linspace(-10, 10, 50)

sigma = 0.1

# plt.fill_between(k_array[k_array<0.5], np.repeat(0., len(k_array[k_array<0.5])), np.repeat(1e9, len(k_array[k_array<0.5])), facecolor="none", color="none", hatch="xx", edgecolor="gray", linewidth=0.0, zorder=-9, alpha=1.)
plt.loglog(k_array[k_array<1], k_array[k_array<1]**3 * pk(k_array[k_array<1], cosmo.pk_cb_lin, z, 10., -0.58, sigma) / 2. / np.pi**2, color=colors[3], alpha=0.8, ls=(0,(2,1,2,1)), lw=2.5)
plt.loglog(k_array[k_array>3], k_array[k_array>3]**3 * pk(k_array[k_array>3], cosmo.pk_cb_lin, z, 0.7, 1.5, sigma) / 2. / np.pi**2, color=colors[2], alpha=0.8, ls=(0,(2,1,2,1)), lw=2.5)
plt.loglog(k_array, k_array**3 * pk(k_array, cosmo.pk_cb_lin, z, 2., 0.5, sigma) / 2. / np.pi**2, color="black", alpha=0.8, lw=2)
plt.loglog(k_array, k_array**3 * pk(k_array, cosmo.pk_cb_lin, z, 0., 1., sigma) / 2. / np.pi**2, color="black", ls=(0,(1,1)), zorder=-9)

N_samp = len(k_array2)
step=10
alphas2 = [np.min([0.+(0.04*i)**2.3,1]) for i in range(int(N_samp/step))][::-1]
alphas3 = [np.min([0.+(0.03*i)**2.3,1]) for i in range(int(N_samp/step))][::-1]

for offset in offsets:
    [plt.loglog(k_array2[step*i:step*(i+2)], (k_array2[step*i:step*(i+2)])**1. * 10**offset, alpha=np.min([0.+(0.03*i)**2.3,1]), color="gray", lw=1, zorder=-9) for i in range(int(N_samp/step))]
    [plt.loglog(k_array2[step*i:step*(i+2)], (k_array2[step*i:step*(i+2)])**(-1.) * 10**(offset+0.273), alpha=np.min([0.+(0.04*i)**2.3,1]), color="gray", lw=1, zorder=-9) for i in range(int(N_samp/step))]
    [plt.loglog(k_array3[step*i:step*(i+2)], (k_array3[step*i:step*(i+2)])**(-1.) * 10**(offset+0.212), alpha=alphas3[numi], color="gray", lw=1, zorder=-9) for numi, i in enumerate(range(int(N_samp/step)))]
    [plt.loglog(k_array3[step*i:step*(i+2)], (k_array3[step*i:step*(i+2)])**(1.) * 10**(offset), alpha=alphas2[numi2], color="gray", lw=1, zorder=-9) for numi2, i in enumerate(range(int(N_samp/step)))]
# for offset in offsets2:
#     [plt.loglog(k_array2[step*i:step*(i+1)], (k_array2[step*i:step*(i+1)])**(-4.) * 10**(offset+0.28), alpha=np.min([0.+(0.03*i)**2,1]), color="gray", lw=1, zorder=-9) for i in range(int(N_samp/step))]


plt.annotate('', xytext=(1.8, 0.67), xy=(4., 0.5), arrowprops={'arrowstyle': '-|>', 'lw':1.3, 'color': "black"}, color="black")
plt.annotate('', xytext=(0.67, 0.949), xy=(1.5, 0.71), arrowprops={'arrowstyle': '<|-', 'lw':1.3, 'color': "black"}, color="black")
plt.annotate('', xytext=(1.24, 1.48), xy=(1.24, 0.83), arrowprops={'arrowstyle': '-|>', 'lw':1., 'color': colors[3]}, color=colors[3])
plt.annotate('', xytext=(3.1, 1.1), xy=(3.1, 0.63), arrowprops={'arrowstyle': '<|-', 'lw':1., 'color': colors[2]}, color=colors[2])



# plt.errorbar(UVLF_2bins_1[0], UVLF_2bins_1[1], xerr=UVLF_2bins_1_xerror, yerr=UVLF_2bins_1_yerror, ls="None", marker=".", markersize=5, markeredgewidth=1.5, elinewidth=1.5, color="black", label=r"$\mathrm{UV\ LF\ (this\ work)}$", zorder=10)
# plt.errorbar(UVLF_2bins_2[0], UVLF_2bins_2[1], xerr=UVLF_2bins_2_xerror, yerr=UVLF_2bins_2_yerror, ls="None", marker=".", markersize=5, markeredgewidth=1.5, elinewidth=1.5, color="black", zorder=10)

ax.text(7.5, 0.052, r'$f_\star > 1$', weight='bold', fontsize=26, color="black", zorder=12, alpha=0.75, rotation=-30.8)
ax.text(0.07, 0.049, r'$\mathrm{Other\ Cosmic}$', weight='bold', fontsize=26, color="black", zorder=12, alpha=0.75, rotation=-28.3)
ax.text(0.1, 0.05, r'$\mathrm{Data}$', weight='bold', fontsize=26, color="black", zorder=12, alpha=0.75, rotation=-28.3)
ax.text(0.624, 7.3e-3, r'$\mathrm{Hubble}$', weight='bold', fontsize=26, color="black", zorder=6, alpha=0.6)
ax.text(0.59, 4e-3, r'$\mathrm{UV\ LFs}$', weight='bold', fontsize=26, color="black", zorder=6, alpha=0.6)
ax.text(0.88, 1.04, r'$f_\star$', weight='bold', fontsize=23, color=colors[3], zorder=6)
ax.text(2.2, 0.756, r'$f_\star$', weight='bold', fontsize=23, color=colors[2], zorder=6)



lines = np.geomspace(0.1, 10, 1000)
alphas = np.sin(np.linspace(0., np.pi/2, int(len(lines)/2)))
alphas = np.concatenate((alphas, alphas[::-1]))
for numline, line in enumerate(lines):
    plt.axvline(line, alpha=alphas[numline], lw=1.5, color="#dadada", zorder=-10)

plt.ylabel(r'$\Delta^2_\mathrm{m}(z = 8)$', fontsize=27)
plt.xlabel(r'$k\ \mathrm{[}\mathrm{Mpc^{-1}]}$', fontsize=27)
plt.semilogx()
plt.semilogy()
plt.xlim(5e-2, 20.)
plt.ylim(1e-3, 1e1)

plt.savefig("pk_with_bumps.pdf")