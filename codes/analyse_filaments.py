import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import curve_fit as fit



filaments = np.load("filament_segs_MD2.npy", allow_pickle=True)
data = np.loadtxt("halos_realspace_z1.txt", skiprows=2)
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
galaxy_cat = np.zeros(shape=(len(x), 3))
galaxy_cat[:, 0] = x
galaxy_cat[:, 1] = y
galaxy_cat[:, 2] = z

box = 2500
h = 0.7

rho_background = len(galaxy_cat[:,0]) / box**3
print(len(galaxy_cat[:,0]))
print(rho_background)

test1 = np.array([1027.641, 1207.161, -574.81])
#print(test1)
test2 = np.array([-6.58, -7.952, 0.14])
#print(test2)
#print(np.cross(test1, test2))
noise = 0
N = 200
r_max = 120
radius = np.linspace(0, r_max, N)

debug_iterations = 1000
density = np.zeros(N-1)
for i in range(1, debug_iterations): # len(filaments) - 1): # Loop over filaments
    print(i)
    density_fil = np.zeros(N-1)
    V_fil = np.zeros(N-1)
    l_fil = 0
    for j in range(1,len(filaments[i])): # Loop over filament segments
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
        #galaxy_cat_bbox  = galaxy_cat[np.where(np.linalg.norm(galaxy_cat - r_mid_fil, axis=1) < (2./3)*filament_length)]

        galaxy_cat_center = galaxy_cat_bbox - r_mid_fil # Setting center of coordinate system to midpoint of filament
        galaxy_cat_center = (galaxy_cat_center + 0.5 * box) % box - 0.5 * box # Periodic boundary conditions
        
        distance_from_filament = np.linalg.norm(np.cross(galaxy_cat_center, r1_0), axis=1) / np.linalg.norm(r1_0)
        distance_from_filament = distance_from_filament[np.where(distance_from_filament <= r_max)]
        hist, bins = np.histogram(distance_from_filament, radius)
        V_seg = np.zeros(N-1)
        for k in range(N-1):     
            V_seg[k] = np.pi * filament_length * (radius[k+1]**2 - radius[k]**2)#(np.pi * filament_length * (radius[k+1]-radius[k])**2)
        density_fil += hist
        V_fil += V_seg
    if l_fil < 3 or l_fil > 30 :
        # Filament too short or too long (Tempel et al. 2014)
        print("Short/long filament")
        noise += 1
    else:
        density += density_fil / V_fil
        
    #print(density)

def exponent(x, A, q):
    return A*x**q
popt, cov = fit(exponent, radius[1:], density)
#print(popt)
print(density /((debug_iterations-noise) * rho_background))
fig, ax = plt.subplots(2)
ax[0].loglog(radius[1:], exponent(radius[1:], popt[0], popt[1]), "r--", label=r"$\propto r^{{{0:.2f}}}$".format(popt[1]))
ax[0].loglog(radius[1:], density / ((debug_iterations-noise) * rho_background))
ax[1].plot(radius[1:], density / ((debug_iterations-noise) * rho_background))
ax[0].set_xlabel("radius (Mpc/h")
ax[0].legend()
fig.savefig("particle_dist_MD2.pdf", dpi=1000)



