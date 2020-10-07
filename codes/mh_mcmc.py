import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random
from theory_model import TheoryModel
import scipy.interpolate as interpolate
import scipy.integrate as integrate



def q(input_arr, mu, sigma):
    Normal_dist =  1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(sigma)))
    Normal_dist *= np.exp(-0.5 * np.dot(np.dot((input_arr - mu).T, np.linalg.inv(sigma)), (input_arr - mu)))
    return Normal_dist

def chi_square(params, model, data, r):
    # Multivariate function
    f  = params[0]
    bias = params[1]
    omega_m = params[2]
    omega_Lambda = params[3]
    perp = params[4]
    par = perp**(-1./3)
    s_s, xi0_s, xi2_s = model(f, bias, omega_m, omega_Lambda, par, perp, 50, 50, streaming=True)
    xi2 = interpolate.interp1d(s_s, xi2_s, kind="cubic")
    s_s[np.where(s_s > r[-1])] = r[-1]
    s_s[np.where(s_s < r[0])] = r[0]
    res = np.sum(data(s_s) - xi2(s_s))**2
    return res / 0.006**2



#y.append(np.random.multivariate_normal(mu, sigma))
def Metropolis_hastings(mu, sigma, model, data, N, r):
    print "Starting metropolis chain" 
    u = np.random.rand(N)
    y = []
    y.append(mu)
    for i in range(N-1):
        print i
        y_new = np.random.multivariate_normal(mu, sigma)
        y_prev = y[i]
        alpha = min(1, chi_square(y_new, model, data, r) * q(y_prev, mu, sigma) / (chi_square(y_prev, model, data, r) * q(y_new, mu, sigma)))
        if u[i] < alpha:
            y.append(y_new)
        else:
            y.append(y_prev)  
    y = np.array(y)
    return y


void_file = "../MD3/zobov-void_cat.txt"
galaxy_file  = "../../summerproject/Haakon/MultiDarkSimulations/HaloSample3/halos_realspace_z1.txt"


model = TheoryModel(50, 2500.0, 1.0, void_file, galaxy_file, "MD3")
model.xi_vg_real_func("MD3void_real.txt", "MD3galaxy_real.txt", "xi_vg_realMD3.npy")
model.delta_and_sigma_vz_galaxy(["deltaMD3.npy", "sigma_vzMD3.npy"])
model.contrast_galaxy("contrastMD3.npy")

rsd_galaxy_file = "../../summerproject/Haakon/MultiDarkSimulations/HaloSample3/halos_redshiftspace_z1.txt"
x, corr, paircounts = model.compute_angular_cross_correlation("MD3void_real.txt",
                                                              rsd_galaxy_file,
                                                              "corr.txt",
                                                              2500.0)

# Fetch the data
r  = x[0]
mu = x[1]
corr_of_mu_s = interpolate.interp2d(mu, r, corr, kind='cubic')
xi0_rsd = np.zeros(len(r))
xi2_rsd = np.zeros(len(r))
# Compute multipoles...
for index, value in enumerate(r):
  xi0_rsd[index] = integrate.quad(lambda mu: 1.0*corr_of_mu_s(mu, value)*1.0,
                          0,
                          1,
                          full_output=1)[0]
  xi2_rsd[index] = integrate.quad(lambda mu: 5.0*corr_of_mu_s(mu, value) * 0.5 *(3 * mu * mu -1 ),
                                  0,
                                  1,
                                   full_output=1)[0]
 
xi2_rsd = interpolate.interp1d(r, xi2_rsd, kind="cubic")
mu_f = 0.872
mu_bias = 2.77
mu_Omega_m = 0.307
mu_Omega_lambda = 1.0 - 0.307 
mu_alpha_perp = 1.0


params = np.array([mu_f, mu_bias, mu_Omega_m, mu_Omega_lambda, mu_alpha_perp])
sigma = 0.006 * np.eye(5)
N = int(1e4)

dist = Metropolis_hastings(params, sigma, model.correlation_rsd_theory, xi2_rsd, N, r) 
np.save("dist", dist)
