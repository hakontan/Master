import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import curve_fit as fit
from mpl_toolkits import mplot3d
from matplotlib import rc

fonts = {
    "font.family": "serif",
    "axes.labelsize": 22,
    "font.size": 22,
    "legend.fontsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    'text.usetex': True, 
}
plt.rcParams.update(fonts)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

filaments = np.load("filament_segs/filament_segs_MD3_s2.npy", allow_pickle=True)
data = np.loadtxt("../../../../Documents/summerproject/Haakon/MultiDarkSimulations/HaloSample3/halos_realspace_z1.txt", skiprows=2)
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
vx = data[:, 3]
vy = data[:, 4]
vz = data[:, 5]


galaxy_cat = np.zeros(shape=(len(x), 3))
galaxy_cat[:, 0] = x
galaxy_cat[:, 1] = y
galaxy_cat[:, 2] = z
velocity_cat = np.zeros(shape=(len(x), 3))
velocity_cat[:, 0] = vx
velocity_cat[:, 1] = vy
velocity_cat[:, 2] = vz

box = 2500.0
h = 0.7

rho_background = len(x) / box**3
print(len(x))
print(rho_background)


noise = 0
N = 150
r_max = 300
radius = np.linspace(0, r_max, N+1)
debug_iterations = 1000
velocity_profile = np.zeros(N)
density = np.zeros(N)
N_in_velocity = np.zeros(N)
V_tot = np.zeros(N)

#f = open("Value_controll_per_fil.txt", "w")

for i in range(1, len(filaments) - 1): # Loop over filaments
    print(i)
    density_fil = np.zeros(N)
    V_fil = np.zeros(N)
    velocity_fil = np.zeros(N)
    N_in_velocity_fil = np.zeros(N)
    num_v_in_bin = np.zeros(N)
    l_fil = 0
    #f.write("Filament nr: {0}\n".format(i-1))
    for j in range(1, len(filaments[i])): # Loop over filament segments
        r0 = filaments[i][j-1] # Start point of filament
        r1 = filaments[i][j]   # End point of filament
        r1_0 = r1 - r0
        r_mid_fil = r0 + 0.5 * (r1 - r0) # Middle point of filament. 
        segment_length = np.linalg.norm(r1 - r0)
        l_fil += segment_length
        r_mid_norm = np.linalg.norm(r_mid_fil)
        # Creating bounding box for filament
        P0 = lambda x: r1_0[0] * (x[:,0] - r0[0]) + r1_0[1] * (x[:,1] - r0[1]) + r1_0[2] * (x[:,2] - r0[2])
        P1 = lambda x: r1_0[0] * (x[:,0] - r1[0]) + r1_0[1] * (x[:,1] - r1[1]) + r1_0[2] * (x[:,2] - r1[2])
        
        # Masking particles far away from the filament        
        galaxy_cat_bbox = galaxy_cat[np.logical_and(P0(galaxy_cat) > 0, P1(galaxy_cat) < 0)]
        velocities_bbox = velocity_cat[np.logical_and(P0(galaxy_cat) > 0, P1(galaxy_cat) < 0)]

        # Optional code for plotting 3d cube with galaxy positions and filament segments
        """
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(r0[0], r0[1], r0[2], color="red", s=10)
        ax.scatter(r1[0], r1[1], r1[2], color="red", s=10)
        ax.scatter(x, y,z, color="blue", s=0.01)
        plt.tight_layout()
        fig.savefig("catalogue_nocut.png")
        plt.show()
        """
        galaxy_cat_center = galaxy_cat_bbox - r_mid_fil # Setting center of coordinate system to midpoint of filament
        galaxy_cat_center = (galaxy_cat_center + 0.5 * box) % box - 0.5 * box # Periodic boundary conditions
        
        #Calculating distance from the filament for all particles 
        distance_from_filament = np.linalg.norm(np.cross(galaxy_cat_center, r1_0), axis=1) / np.linalg.norm(r1_0)
        halos_near_filament = galaxy_cat_center[np.where(distance_from_filament <= r_max)]
        # Masking particles far away from filament
        velocities = velocities_bbox[np.where(distance_from_filament <= r_max)]
        distance_from_filament = distance_from_filament[np.where(distance_from_filament <= r_max)]
        
        # Calculating perpendicular component of all particles.
        r_par = np.sqrt(np.linalg.norm(halos_near_filament, axis=1)**2
                         - distance_from_filament**2)
        r_par_vec = np.repeat(r_par[:, np.newaxis], 3, axis=1) * r1_0 / segment_length
        r_perp = halos_near_filament - r_par_vec
        
        radial_velocity = (velocities * r_perp).sum(axis=1) / np.linalg.norm(r_perp, axis= 1)
        
        # Binning particles a given distance from filament spine
        hist, bins = np.histogram(distance_from_filament, radius)
        
        radial_velocity_binned = np.zeros(N)
        N_radial_velocity_in_bin = np.zeros(N)
        
        V_seg = np.zeros(N)
        for k in range(1, N + 1):     
            V_seg[k-1] += np.pi * (radius[k]**2 - radius[k-1]**2) * segment_length
            v_in_bin = (np.sum(radial_velocity[np.logical_and(distance_from_filament > radius[k-1],
                                                              distance_from_filament < radius[k])]))

            if hist[k-1] > 0:
                num_v_in_bin[k-1] += hist[k-1]
                radial_velocity_binned[k-1] += v_in_bin
                N_radial_velocity_in_bin[k-1] += hist[k-1]

        #for k in range(1, N + 1):
        #    V_fil[k-1] += np.pi * (radius[k]**2 - radius[k-1]**2) *
        #    segment_length
        #V_fil += V_seg
        density_fil += hist / V_seg
        #density += hist
        velocity_fil += radial_velocity_binned
        N_in_velocity_fil += N_radial_velocity_in_bin
        
    if l_fil < 10 or l_fil > 100:
        print("Short/long filament")
        noise += 1
    elif np.isnan(radial_velocity_binned).any() or np.isnan(N_radial_velocity_in_bin).any() or np.isnan(density).any():
        noise += 1
        print("Nan appeared")
    else:
        for k in range(1, N + 1):
            V_fil[k-1] = np.pi * (radius[k]**2 - radius[k-1]**2) * l_fil
        #print(density)
        density += density_fil / len(filaments[i])
        V_tot += V_fil #/ V_seg * l_fil
        #velocity_profile += velocity_fil / np.maximum(np.ones(N), num_v_in_bin)
        N_in_velocity += N_in_velocity_fil
        velocity_profile += velocity_fil

def exponent(x, A, q):
    return A*x**q
#f.close()
#popt, cov = fit(exponent, radius[1:], density)
#print(popt)
rho_final = (density / (len(filaments)-noise))/ rho_background
#v_final = velocity_profile / (debug_iterations-noise)
v_final = (velocity_profile / np.maximum(np.ones(N), N_in_velocity))
np.save("datafiles/density_profiles/rho_MD3_s2_test", rho_final)
np.save("datafiles/velocity_profiles/v_finalN_MD3_s2_test", v_final)
print(rho_final)

fig, ax = plt.subplots()
#ax[0].loglog(radius[1:], exponent(radius[1:], popt[0], popt[1]), "r--", label=r"$\propto r^{{{0:.2f}}}$".format(popt[1]))
ax.loglog(radius[1:], rho_final)
ax.set_xlabel("radius [Mpc/h]")
ax.set_ylabel(r"$\rho(r)/\bar{\rho}$")
ax.legend()
fig.savefig("figures/density_profiles/density_MD3_s2_test.pdf")

fig, ax = plt.subplots()
ax.plot(radius[1:], v_final)
ax.set_xlabel("radius [Mpc/h]")
ax.set_ylabel("v(r)")
fig.savefig("figures/velocity_profiles/velocity_MD3_s2_test.pdf")

