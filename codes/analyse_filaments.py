import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import curve_fit as fit


filaments = np.load("filament_segs_MD1.npy", allow_pickle=True)
data = np.loadtxt("halos_realspace_z1MD1.txt", skiprows=2)
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

box = 2500
h = 0.7

rho_background = len(x) / box**3
print(len(galaxy_cat[:,0]))
print(rho_background)


noise = 0
N = 100
r_max = 200
radius = np.linspace(0, r_max, N+1)
debug_iterations = 2500
velocity_profile = np.zeros(N)
density = np.zeros(N)
for i in range(1, debug_iterations): # len(filaments) - 1): # Loop over filaments
    print(i)
    density_fil = np.zeros(N)
    V_fil = np.zeros(N)
    velocity_fil = np.zeros(N)
    num_v_in_bin = np.zeros(N)
    l_fil = 0
    for j in range(1, len(filaments[i])): # Loop over filament segments
        r0 = filaments[i][j-1] # Start point of filament
        r1 = filaments[i][j]   # End point of filament
        r1_0 = r1 - r0
        r_mid_fil = r0 + 0.5 * (r1 - r0) # Middle point of filament. 
        filament_length = np.linalg.norm(r1 - r0)
        l_fil += filament_length
        r_mid_norm = np.linalg.norm(r_mid_fil)
        # Creating bounding box for filament
        P0 = lambda x: r1_0[0] * (x[:,0] - r0[0]) + r1_0[1] * (x[:,1] - r0[1]) + r1_0[2] * (x[:,2] - r0[2])
        P1 = lambda x: r1_0[0] * (x[:,0] - r1[0]) + r1_0[1] * (x[:,1] - r1[1]) + r1_0[2] * (x[:,2] - r1[2])
        
        # Masking particles far away from the filament        
        galaxy_cat_bbox = galaxy_cat[np.logical_and(P0(galaxy_cat) > 0, P1(galaxy_cat) < 0)]
        velocities_bbox = velocity_cat[np.logical_and(P0(galaxy_cat) > 0, P1(galaxy_cat) < 0)]

        galaxy_cat_center = galaxy_cat_bbox - r_mid_fil # Setting center of coordinate system to midpoint of filament
        galaxy_cat_center = (galaxy_cat_center + 0.5 * box) % box - 0.5 * box # Periodic boundary conditions
        
        distance_from_filament = np.linalg.norm(np.cross(galaxy_cat_center, r1_0), axis=1) / np.linalg.norm(r1_0)
        halos_near_filament = galaxy_cat_center[np.where(distance_from_filament <= r_max)]
        velocities = velocities_bbox[np.where(distance_from_filament <= r_max)]
        distance_from_filament = distance_from_filament[np.where(distance_from_filament <= r_max)]
        #galaxy_cat_center = galaxy_cat_center[np.where(distance_from_filament <= r_max)]
        
        r_par = np.sqrt(np.linalg.norm(halos_near_filament, axis=1)**2
                         - distance_from_filament**2)
        r_par_vec = np.repeat(r_par[:, np.newaxis], 3, axis=1) * r1_0 / filament_length
        r_perp = halos_near_filament - r_par_vec
        
        radial_velocity = (velocities * r_perp).sum(axis=1) / np.linalg.norm(r_perp, axis= 1)

        #hist, bins = np.histogram(distance_from_filament, radius)
        #print(len(hist), len(bins), len(radius))
        radial_velocity_binned = np.zeros(N)
        hist = np.zeros(N)
        V_seg = np.zeros(N)
        for k in range(1, N + 1):     
            V_seg[k-1] = np.pi * filament_length * (radius[k]**2 - radius[k-1]**2)
            v_in_bin = (np.sum(radial_velocity[np.logical_and(distance_from_filament > radius[k-1],
                                                              distance_from_filament < radius[k])]))
            hist[k-1] = distance_from_filament[np.logical_and(distance_from_filament > radius[k-1],
                                                              distance_from_filament < radius[k])].size
            #print(hist[k])
            #print(v_in_bin)
            if hist[k-1] > 0:
                num_v_in_bin[k-1] += hist[k-1]
                radial_velocity_binned[k-1] += v_in_bin

        density_fil += hist
        V_fil += V_seg
        velocity_fil += radial_velocity_binned
    if l_fil < 3 or l_fil > 30 :
        # Filament too short or too long (Tempel et al. 2014)
        print("Short/long filament")
        noise += 1
    else:
        density += density_fil / V_fil
        velocity_profile += velocity_fil / np.maximum(np.ones(N), num_v_in_bin)
        

def exponent(x, A, q):
    return A*x**q

#popt, cov = fit(exponent, radius[1:], density)
#print(popt)
rho_final = density / ((debug_iterations-noise) * rho_background)
v_final = velocity_profile / (debug_iterations-noise)

np.save("rho_N100_MD1_newbin", rho_final)
np.save("v_finalN100_MD1_newbin", v_final)

fig, ax = plt.subplots(2)
#ax[0].loglog(radius[1:], exponent(radius[1:], popt[0], popt[1]), "r--", label=r"$\propto r^{{{0:.2f}}}$".format(popt[1]))
ax[0].loglog(radius[:-1], density / ((debug_iterations-noise) * rho_background))
ax[1].plot(radius[:-1], density / ((debug_iterations-noise) * rho_background))
ax[0].set_xlabel("radius (Mpc/h")
ax[0].legend()
fig.savefig("density_MD1.pdf", dpi=1000)

fig, ax = plt.subplots(2)
ax[0].loglog(radius[:-1], velocity_profile / (debug_iterations - noise))
ax[1].plot(radius[:-1], velocity_profile / (debug_iterations - noise))
ax[0].set_xlabel("radius (Mpc/h")
fig.savefig("velocity_MD1.pdf", dpi=1000)



