from re import I
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from numpy.core.getlimits import iinfo
from scipy.optimize import curve_fit as fit
from matplotlib.gridspec import GridSpec
from matplotlib import rc

# Code for creating scatter plots of Galaxy catalogue
# with filaments detected by diserpse overplotted

fonts = {
    "font.family": "serif",
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    'text.usetex': True, 
}
plt.rcParams.update(fonts)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

data = np.loadtxt("../../../../Documents/summerproject/Haakon/MultiDarkSimulations/HaloSample2/halos_realspace_z1.txt", skiprows=2)
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
vx = data[:, 3]
vy = data[:, 4]
vz = data[:, 5]


filaments1 = np.load("filament_segs/filament_segs_MD2_all.npy", allow_pickle=True)
filaments2 = np.load("filament_segs/filament_segs_MD2_s1.npy", allow_pickle=True)
filaments3 = np.load("filament_segs/filament_segs_MD2_s2.npy", allow_pickle=True)
filaments4 = np.load("filament_segs/filament_segs_MD2_s3.npy", allow_pickle=True)

filament_arr = [filaments1, filaments2, filaments3, filaments4]
fig = plt.figure()
gs=GridSpec(2,2)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
c  = 0
for m in range(2):
    for n in range(2):
        print(m,n)
        ax = fig.add_subplot(gs[m, n])
        filaments = filament_arr[c]
        filament_lengths = []
        z_max = 1010
        z_min = 990
        index = np.where(np.logical_and(z >= z_min, z <= z_max))
        
        ax.scatter(x[index], y[index], s= 0.1)
        for i in range(0, len(filaments) - 1): # Loop over filaments
            l_fil = 0
            #f.write("Filament nr: {0}\n".format(i-1))
            start_point = filaments[i][0]
            color = next(ax._get_lines.prop_cycler)['color']
            for j in range(1, len(filaments[i])): # Loop over filament segments
                segments = len(filaments[i])
                r0 = filaments[i][j-1] # Start point of filament
                r1 = filaments[i][j]   # End point of filament
                end_point = r1
                r1_0 = r1 - r0
                r_mid_fil = r0 + 0.5 * (r1 - r0) # Middle point of filament. 
                seg_length = np.linalg.norm(r1 - r0)
                l_fil += seg_length
            filament_lengths.append(l_fil)
            if z_min < r1[2] < z_max and z_min < r1[2] < z_max and 0 < l_fil < 10000: 
                for j in range(1, segments):
                    r0 = filaments[i][j-1] # Start point of filament
                    r1 = filaments[i][j]   # End point of filament
                    x_axis = [r0[0], r1[0]]
                    y_axis = [r0[1], r1[1]] 
                    ax.plot(x_axis, y_axis, color=color, linewidth = 1.0)

        y_ind = np.arange(0,3000,500)

        
        if c == 0:
            ax.get_xaxis().set_ticks([])
            plt.xticks(y_ind,['']*len(y_ind))
            ax.get_yaxis().set_ticks([0, 500, 1000, 1500, 2000, 2500])
            ax.set_ylabel("y [Mpc/h]")
            ax.set_title(r"MD2, no cuts")
        if c == 1:
            ax.set_title(r"MD2, $\sigma = 1$")
            ax.get_xaxis().set_ticks([])
            plt.xticks(y_ind,['']*len(y_ind))
            ax.get_yaxis().set_ticks([])
            plt.yticks(y_ind,['']*len(y_ind))
        if c == 2:
            ax.set_title(r"MD2, $\sigma = 2$")
            ax.get_xaxis().set_ticks([0, 500, 1000, 1500, 2000, 2500])
            ax.get_yaxis().set_ticks([0, 500, 1000, 1500, 2000, 2500])
            ax.set_ylabel("y [Mpc/h]")
            ax.set_xlabel("x [Mpc/h]")
        if c == 3:
            ax.set_title(r"MD2, $\sigma = 3$")
            ax.set_xlabel("x [Mpc/h]")
            ax.get_yaxis().set_ticks([])
            plt.yticks(y_ind,['']*len(y_ind))
            ax.get_xaxis().set_ticks([0, 500, 1000, 1500, 2000, 2500])
        c+=1
plt.tight_layout()
fig.savefig("figures/scatterplots/scatter_MD2.pdf")

"""
fig2, ax2 = plt.subplots()
#hist, bins = np.histogram(filament_lengths, 50)
bins = np.arange(0, 200, 5)
ax2.hist(filament_lengths, bins)
ax2.set_xlabel("Filament Length [Mpc/h]")
ax2.set_ylabel("nr in bin")
plt.xticks([0,50,100,150,200])
ax2.set_title(r"MD1, $\sigma = 3$")
plt.tight_layout()
fig2.savefig("figures/histograms/filament_histMD1_s3.pdf")
"""