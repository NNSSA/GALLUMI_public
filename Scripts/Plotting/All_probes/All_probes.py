import numpy as np
from matplotlib import pyplot as plt
import matplotlib
plt.style.use("../template.mplstyle")
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# purple - green - darkgoldenrod - blue - red
colors = ['purple', '#3F7BB6', "#12a38c", "#6e9773", 'darkgoldenrod', "#cf630a", '#BF4145']
linestyles = [(0, (1,1.05)), (0, (3, 1, 1, 1)), (0, (1,3)), (0, (3,3.65)), (0, (3,2.772)), (0, (3, 1, 1, 1, 1, 1))]

#########################################################################################


fig = plt.figure(figsize=(8,6.7), facecolor='white')

rhoM = 2.77463e11 * 0.7**2 * 0.3

left_margin = 0.6 / fig.get_figwidth()
right_margin = 0.25 / fig.get_figwidth()
bottom_margin = 0.75 / fig.get_figheight()
top_margin = 0.25 / fig.get_figwidth()
mid_margin = 0.3 / fig.get_figwidth()  
f = [0.95, 0.05]
Naxes = len(f)
x0, y0 = left_margin, bottom_margin
h = 1 - (bottom_margin + top_margin)
wtot = 1 - (left_margin + right_margin + (Naxes-1)*mid_margin)
w = wtot*f[0]
ax = fig.add_axes([x0, y0, w, h], frameon=True, facecolor='none')
x0 += w + mid_margin
w = wtot*f[1]
ax2 = fig.add_axes([x0, y0, w, h], frameon=True, facecolor='none')
x0 += w + mid_margin

ax.tick_params(axis='x', which='major', pad=6)
ax2.tick_params(axis='x', which='major', pad=6)
ax.tick_params(axis='both', which='major', labelsize=29)
ax.tick_params(axis='both', which='minor', labelsize=29)
ax2.tick_params(axis='both', which='major', labelsize=29)
ax2.tick_params(axis='both', which='minor', labelsize=29)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)
    ax2.spines[axis].set_linewidth(2.5)

ax.set_xlim(1., 30.) 
ax.set_ylim(1e-2, 1e2) 
ax2.set_xlim(900, 1000.)
ax2.set_ylim(1e-2, 1e2) 
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax2.yaxis.tick_right()
ax2.tick_params(axis='x', which='minor', bottom=False, top=False)
ax2.tick_params(axis='y', which='both', labelright=False)

ax3 = ax2.twinx()
ax3.set_ylim(1e-2, 1e2) 
ax3.spines['left'].set_visible(False)
ax3.yaxis.tick_right()
ax3.set_ylabel(r'$M_\mathrm{h}\ \mathrm{at}\ z = 0\ [M_\odot]$', labelpad=35, fontsize=31, rotation=-90)
ax3.tick_params(axis='both', which='major', labelsize=28)
ax3.tick_params(axis='both', which='minor', labelsize=28)
ax3.tick_params(axis='y', which='minor', left=False, right=False)
ax2.tick_params(axis='y', which='both',length=0)
ax3.set_yscale('log')

ax.set_ylabel(r'$k\ [\mathrm{Mpc}^{-1}]$', fontsize=31, labelpad=-5)
ax.set_xlabel(r'$1 + z$', fontsize=31, va='center', ha='center')
ax.xaxis.set_label_coords(0.54, 0.006, transform=fig.transFigure)

d = .015
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1-d/2,1+d/2), (-d,+d), **kwargs)
ax.plot((1-d/2.5,1+d/2),(1-d,1+d), **kwargs)
d = .015
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d*8.3,+d*8.3), (1-d,1+d), **kwargs)
ax2.plot((-d*8.3,+d*8.3), (-d,+d), **kwargs)

nonlinear = np.loadtxt("nonlinear.txt", unpack=True)
ax.plot(nonlinear[0], nonlinear[1], color="black", lw=2, alpha=0.7, zorder=-1)
ax.annotate(text='', xytext=(4.57, 8.8), xy=(1.7, 30), arrowprops={'arrowstyle': '-|>', 'color': 'black', 'alpha':0.7}, color='black')
ax.text(2.1, 11.4, r'$\mathrm{non-linear}$', fontsize=20, color='k', rotation=-22.2)

ax.add_patch(plt.Rectangle(xy=(1., 0.8), width=0.15, height=25-0.8, color=np.array([0.501, 0, 0.501, 0.5]), zorder=1))
ax.add_patch(plt.Rectangle(xy=(1.15, 0.07), width=0.85, height=2.8-0.07, color=np.array([0.721, 0.525, 0.043,0.55]), zorder=2))
ax.add_patch(plt.Rectangle(xy=(1.3, 0.01), width=0.7, height=0.2-0.01, color=np.array([0.070, 0.639, 0.549, 0.5]), zorder=1))
ax.add_patch(plt.Rectangle(xy=(2., 0.01), width=9., height=0.5-0.01, color=np.array([0.188, 0.419, 0.215, 0.45]), zorder=0))
ax.add_patch(plt.Rectangle(xy=(3.1, 0.07), width=2.7, height=7.-0.07, color=np.array([0.749, 0.254, 0.270, 0.55]), zorder=1))
ax.add_patch(plt.Rectangle(xy=(5, 0.1), width=6, height=10.-0.1, facecolor=np.array([0.243, 0.478, 0.713, 0.55]), linestyle=(1,(2,1)), linewidth=3, edgecolor=np.array([0.196, 0.384, 0.572]), zorder=2))
ax.add_patch(plt.Rectangle(xy=(6., 1.), width=24., height=50.-1., color=np.array([0., 0., 0., 0.25]), zorder=1))
ax2.add_patch(plt.Rectangle(xy=(910, 0.01), width=90, height=0.2-0.01, color=np.array([0.811, 0.388, 0.039, 0.55]), zorder=1))

ax.text(1.2, 3.55, r'$\mathbf{MW\ Sat.}$', fontsize=22, rotation=90, color=np.array([0.501, 0, 0.501, 1.]))
ax.text(1.145, 0.8, r'$\mathbf{Cosmic}$', fontsize=22, color=np.array([0.576, 0.419, 0.035, 1.]))
ax.text(1.24, 0.49, r'$\mathbf{shear}$', fontsize=22, color=np.array([0.576, 0.419, 0.035, 1.]))
ax.text(1.331, 0.013, r'$\mathbf{LRG}$', fontsize=22, color=np.array([0.062, 0.576, 0.494, 1.]))
ax.text(3.8, 0.013, r'$\mathbf{CMB\ lensing}$', fontsize=22, color=np.array([0.188, 0.419, 0.215, 1.]))
ax.text(3.87, 0.193, r'$\mathbf{Lyman}\boldsymbol{-\alpha}$', fontsize=22, rotation=90, color=np.array([0.450, 0.152, 0.160, 0.9]))
ax.text(5.7, 1.3, r'$\mathbf{UV\ LF}$', fontsize=22, color=np.array([0.145, 0.286, 0.427, 0.75]))
ax.text(6.52, 0.74, r'$\mathbf{This}$', fontsize=19, color=np.array([0.145, 0.286, 0.427, 0.75]))
ax.text(6.22, 0.51, r'$\mathbf{Work}$', fontsize=19, color=np.array([0.145, 0.286, 0.427, 0.75]))
ax.text(5.65, 0.57, r'$\mathbf{(}$', fontsize=40, color=np.array([0.145, 0.286, 0.427, 0.75]))
ax.text(8.95, 0.57, r'$\mathbf{)}$', fontsize=40, color=np.array([0.145, 0.286, 0.427, 0.75]))
ax.text(16.2, 1.25, r'$\mathbf{21\hspace{-2mm}\boldsymbol{-}\hspace{-2mm}cm}$', fontsize=22, color=np.array([0,0,0, 0.6]))
ax.text(30.2, 0.0238, r'$\mathbf{CMB}$', fontsize=22, rotation=90, color=np.array([0.729, 0.349, 0.035, 1.]))

ax.semilogy()
ax.semilogx()
ax2.semilogy()
ax2.semilogx()
ax.set_xticks([1., 5., 10.])
ax.set_xticklabels([r'$1$', r'$5$', r'$10$'])
Mhs = np.array([1e8, 1e11, 1e14, 1e17])
ks = 1/(np.power(3. * Mhs / (4. * np.pi * rhoM), 1./3) / 2.5)
ax3.set_yticks([ks[0], ks[1], ks[2], ks[3]])
ax3.set_yticklabels([r'$10^{8}$', r'$10^{11}$', r'$10^{14}$', r'$10^{17}$'])

plt.savefig("All_probes.pdf")