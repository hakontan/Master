import numpy as np
import sys
sys.path.insert(1, '/uio/hume/student-u88/haakota/Documents/summerproject/Python3CUTEbox')
import pycutebox
import scipy
import scipy.spatial
import sys
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from periodic_kdtree import PeriodicCKDTree
import scipy.interpolate as interpolate
import scipy.integrate as integrate

class TheoryModel():
    """
    Module for calculating the theoretical model for the correlation function between voids in realspace
    and galaxies in redshiftspace and related functions as described in S.Nadathur et al 2019.
    Dependencies:
    ------------
    * This module relies on the pyCUTE library for calculating 2-point correlation functions from data.
    Documentation for pyCUTE is found here https://github.com/seshnadathur/pyCUTE/tree/master/PythonCUTEbox.
    * This module also relies on the periodic_kdtree module for calculating density profile and velocity dispersion.
    Documentation for periodic_kdtree can be found here https://github.com/patvarilly/periodic_kdtree.
    """
    
    def __init__(self, N, box_size, z, void_cat, galaxy_cat):
        
        self.N = N # Number of iterations for creating splines

        # catalogue specific parameters
        self.box_size = box_size # Size of simulation box
        self.z = z # Redshift
        
        
        self.h = 0.7
        self.H0 = 100 # km/s/ h Mpc

        self.converted = False # Used to check wether splines have been converted to r_real coordinates
                               # So that it will not be run multiple times.

        void_cat = np.loadtxt(void_cat, skiprows=2)
        
        void_x = void_cat[:, 1]
        void_y = void_cat[:, 2]
        void_z = void_cat[:, 3]

        print "Reading galaxy catalogue"
        galaxy_cat = np.loadtxt(galaxy_cat)
    
        galaxy_x = galaxy_cat[:, 0]
        galaxy_y = galaxy_cat[:, 1]
        galaxy_z = galaxy_cat[:, 2]
        self.galaxy_vz = galaxy_cat[:, 5]
       
        # Stacking galaxy and void catalogues as 2D arrays with x, y and z positions on
        # each column
        self.galaxy_cat = np.column_stack((galaxy_x, galaxy_y, galaxy_z))
        self.void_cat   = np.column_stack((void_x, void_y, void_z))

    def delta_and_sigma_vz_galaxy(self):
        """
        Calculates the density profile and velocity dispersion of voids in real space.
        Requires xi_vg_real_func() to be run first as this gives the upper and lower bounds
        for the radius array to avoid out of bounds for splines.
        """
        bounds = np.array([self.box_size, self.box_size, self.box_size])
        tree = PeriodicCKDTree(bounds, self.galaxy_cat)
        radius_array = np.linspace(0, self.r_corr[-1], self.N + 1)

        delta = np.zeros(self.N + 1)
        v_z   = np.zeros(self.N + 1) 
        E_vz  = np.zeros(self.N + 1)
        E_vz2 = np.zeros(self.N + 1)
        sigma_vz = np.zeros(self.N + 1)
        galaxies_in_shell_arr = np.zeros(self.N + 1)

        print "Starting density profile and velocity dispersion calculation"
        for i in range(len(self.void_cat[:, 0])):
            current_number_of_galaxies = 0
            current_E_vz = 0
            current_E_vz2 = 0
            E_vz_in_shell = 0
            E_vz2_in_shell = 0
           
            for j in range(1, self.N + 1):
                # Find galaxy position and velocity in a given radius around the current void
                neighbor_inds = tree.query_ball_point(self.void_cat[i, :], r=radius_array[j])
                shell_volume = 4.0 * np.pi * (radius_array[j]**3 - radius_array[j-1]**3) / 3.0
                velocity_near_point = self.galaxy_vz[neighbor_inds]
                galaxies_near_point = self.galaxy_cat[neighbor_inds]
                galaxies_near_point = len(galaxies_near_point[:,0])
                galaxies_in_shell = galaxies_near_point - current_number_of_galaxies # Subtracting previous sphere to get galaxies in current shell.
                
                # calulcating terms used in expectation values E[v_z**2] and E[v_z]**2
                if galaxies_near_point > 0:
                    E_vz2_in_shell = (sum(velocity_near_point**2) - current_E_vz2)
                    E_vz_in_shell  = (sum(velocity_near_point)  - current_E_vz)
               
                
                galaxies_in_shell_arr[j] += galaxies_in_shell
                
                E_vz [j] += E_vz_in_shell
                E_vz2[j] += E_vz2_in_shell
                delta[j] += galaxies_in_shell / shell_volume
                
                

                current_E_vz += E_vz_in_shell
                current_E_vz2 += E_vz2_in_shell
                current_number_of_galaxies += galaxies_in_shell
            

        delta /= (len(self.void_cat[:, 0]) * len(self.galaxy_cat[:,0]) / self.box_size ** 3)
        delta -= 1
        for j in range(self.N + 1):
            if galaxies_in_shell_arr[j] > 0:
                E_vz[j] /= galaxies_in_shell_arr[j]
                E_vz2[j] /= galaxies_in_shell_arr[j]
        sigma_vz = np.sqrt(E_vz2 - E_vz**2)
        
        # Replacing zero values to avoid division by zero later
        sigma_vz[np.where(sigma_vz < 10)] = np.mean(sigma_vz)
        
        fig, ax = plt.subplots()
        ax.plot(radius_array, delta)
        fig.savefig("delta_test.png")
        fig, ax = plt.subplots()
        ax.plot(radius_array, sigma_vz)
        fig.savefig("sigmavz_test.png")
        print "Splining density profile"
        self.delta = interpolate.interp1d(radius_array, delta, kind="cubic")
        self.sigma_vz = interpolate.interp1d(radius_array, sigma_vz, kind="cubic")
        
        return self.delta, self.sigma_vz

    def contrast_galaxy(self):
        """
        Calculates the density contrast given by equation 4 in S.Nadathur et al 2019.
        Requires the delta_and_sigma_vz_galaxy to be run first as this gives the density profile
        required in the integral.
        """
        r = np.linspace(self.r_corr[0], self.r_corr[-1], self.N)
        contrast = np.zeros(len(r))
        print "calculating density contrast"

        # calculating the density contrast given by eq (4).
        for index, radius in enumerate(r):
            if radius == 0:
                # Avoid division by zero
                contrast[index] = - 1.0
            else:
                contrast[index] = 3 * integrate.quad(lambda radius: self.delta(radius) * radius**2,
                                                          0,
                                                          radius,
                                                          full_output=1)[0] / radius**3
        fig, ax = plt.subplots()
        ax.plot(r, contrast)
        fig.savefig("contrast_test.png")
        self.contrast = interpolate.interp1d(r, contrast, kind="cubic")
        return self.contrast

    def compute_angular_cross_correlation(self, tracerfile1, tracerfile2, outputfile, box):
        pycutebox.set_CUTEbox_parameters(data_filename=tracerfile1,
                                         data_filename2=tracerfile2,
                                         output_filename=outputfile,
                                         input_format=0,
                                         corr_type=3,
                                         box_size=self.box_size,
                                         do_CCF=1)
        return pycutebox.runCUTEbox()
    
    def xi_vg_real_func(self, void_file, galaxy_file, output_filename="corr.txt"):
        """
        Calculate the correlation function between voids and galaxies in real space
        used in the theoretical model. This function has to be run first as this determines the upper and lower
        bounds for the radius array used in other calculations to prevent out of bounds for splines.
        Parameters:
        -------------
        Void_file:
            .txt file with void center positions.
        galaxy_file:
            .txt file with positon of galaxies.
        output_filename:
            name of the output filename. Not used in this module but required by the
            CUTEbox module
        
        See CUTEbox documentation for format requirements of input files.
        """
        x, corr, paircounts = self.compute_angular_cross_correlation(void_file,
                                                                     galaxy_file,
                                                                     output_filename,
                                                                     self.box_size)
        self.r_corr = x[0]
        mu = x[1]
        corr_of_mu_s = interpolate.interp2d(mu, self.r_corr, corr, kind='cubic')
        xi_vg_real = np.zeros(len(self.r_corr))
        for index, value in enumerate(self.r_corr):
            xi_vg_real[index] = integrate.quad(lambda mu: 1.0*corr_of_mu_s(mu, value)*1.0,
                                                0,
                                                1,
                                                full_output=1)[0]
        fig, ax = plt.subplots()
        ax.plot(self.r_corr, xi_vg_real)
        plt.savefig("xi_vg_real_test.png")
        self.xi_vg_real = interpolate.interp1d(self.r_corr, xi_vg_real, kind="cubic")
        return self.xi_vg_real

    def H_of_z_func(self, omega_m, Omega_Lambda):
        """
        Function return the Hubble parameter (H)
        as a function of redshift z.
        """
        print(self.H0)
        return self.H0 * np.sqrt(Omega_m * (1.0 + self.z)**3 + Omega_Lambda)


    def convert_splines(self, r_real):
        """
        Converts the splines for delta, sigma_vz, density contrast and
        xi_vg_real to use the rescaled radius r_real in correspondance with
        equation 12 in S.Nadathur et al 2019. This method should only be called
        by correlation_rsd_theory() function.
        Parameters:
        ---------------
        r_real:
            Rescaled radius array. 
        """
        
        # Avoid out of range on splines
        r_real[np.where(r > self.r_corr[-1])] = self.r_corr[-1]
        r_real[np.where(r < self.r_corr[0])] = self.r_corr[0]

        xi_vg_real = self.xi_vg_real(r_real)
        delta = self.delta(r_real)
        contrast = self.contrast(r_real)
        sigma_vz = self.sigma_vz(r_real)

        self.xi_vg_real = interpolate.interp1d(r_real, xi_vg_real, kind="cubic")
        self.delta = interpolate.interp1d(r_real, delta, kind="cubic")
        self.contrast = interpolate.interp1d(r_real, contrast, kind="cubic")
        self.sigma_vz = interpolate.interp1d(r_real, sigma_vz, kind="cubic")
        


    def correlation_rsd_theory(self, f, bias, Omega_m, Omega_Lambda, alpha_par, alpha_perp, n_mu, n_s, streaming = False):
        """
        Calculate the theoretical model for the cross correlation function
        between realspace voids and redshiftspace galaxies and calculates multipoles.
        Parameters:
        -----------------
        f:
            growth rate for galaxy catalogue
        bias:
            dark matter bias for galaxy catalogue
        Omega_m:
            density parameter for matter
        Omega_Lambda:
            density parameter for cosmological constant
        Alpha_par:
            parallel component of Alcock-Paczynski parameter
        Alpha_perp:
            perpendicual component of Alcock-Paczynski parameter
        n_mu:
            number of points for a linearly space array for mu between 0, 1. (mu = cos(theta))
        n_s:
            number of points for aa linearly space s array where s is distance in redshiftspace.
            Upper and lower bounds is determined by the radius array given by the CUTEbox correlation function
            in xi_vg_real_func().

        streaming:
            Determines wether on should use the full theory model with a gaussian streaming profile or the simpler theory model.
        """
        xi_vg_rsd = np.zeros(shape=(n_s, n_mu))
        s_array   = np.linspace(self.r_corr[0] + 1.0 , self.r_corr[-1] - 1.0, n_s) # +1 and -1 to prevent out of range on splines
        mu_array  = np.linspace(0.0, 1.0, n_mu)
        r_fid = np.linspace(self.r_corr[0], self.r_corr[-1], n_s)
        
        # Rescaling radius in correspondance with equation 12 is S.Nadathur et
        # al 2019
        r_integrand = alpha_par * np.sqrt(1 + (1 - mu_array**2) * ((alpha_par/alpha_perp)**2) - 1)
        r_factor = np.trapz(r_integrand, mu_array)
        r_real  = r_fid * r_factor

        # Since splines are stored as class variables, they should only be
        # converted once.
        if not self.converted:
            self.convert_splines(r_real)
        
        xi_vg0 = np.zeros(len(s_array))
        xi_vg2 = np.zeros(len(s_array))
        print "Calculating theoretical model"
        if streaming:
            # Calculate equation 7 in S.Nadathur et al 2019
            aH = (1.0 / (1.0 + self.z)) * self.H_of_z_func(Omega_m, Omega_Lambda) # Hubble parameter times scale factor (a = 1 / (1 + z))
            for i in range(len(s_array)):
                s = s_array[i]
                for j in range(len(mu_array)):
                    mu = mu_array[j]

                    # Performing change of variable where y = v_par / (a * H)
                    # giving dv_par = a*H*dy
                    
                    sigma_tilde = self.sigma_vz(s) / aH
                    
                    y = np.linspace(-5.0 * sigma_tilde, 5.0 * sigma_tilde, 100) # Integration domain determined by gaussian 
                                                                                # part of the integrand. It is basically zero elsewhere
                                                                                                          
                    # Length coordinates in real- and redshiftspace
                    s_par  = s * mu
                    s_perp = s * np.sqrt(1 - mu**2)

                    r_par  = s_par + s * f * self.delta(s) * mu / (3.0 * bias) - y
                    r_perp = s_perp
                    r      = np.sqrt(r_par**2 + r_perp**2)

                    # Avoid out of range on splines
                    r[np.where(r > self.r_corr[-1])] = self.r_corr[-1]
                    r[np.where(r < self.r_corr[0])] = self.r_corr[0]
                    #print r
                
                    # Collecting terms with velocity dispersion added.
                    xi_s_perp_s_par = ((1 + self.xi_vg_real(r)) * (1 + (f / bias * self.contrast(r)/ 3.0
                                        - y * mu / r) * (1 - mu**2)
                                        + f * (self.delta(r) - 2 * self.contrast(r) / 3.0) / bias * mu**2))
                    integrand = (1 + xi_s_perp_s_par) / ((np.sqrt(2 * np.pi) * sigma_tilde))
                    integrand *= np.exp(-0.5 * y**2 / sigma_tilde**2)
                    xi_vg_rsd[i, j] = np.trapz(integrand, y) - 1

        else:
            # Calculate equation 6 in S.Nadathur et al 2019
            for i in range(len(s_array)):
                s = s_array[i]
                for j in range(len(mu_array)):
                    mu = mu_array[j]

                    # Length coordinates in real- and redshiftspace
                    s_par  = s * mu
                    s_perp = s * np.sqrt(1 - mu**2)

                    r_par  = s_par + s * self.f * self.delta(s) * mu / (3.0 * self.bias)
                    r_perp = s_perp
                    r      = np.sqrt(r_par**2 + r_perp**2)

                    # Base theory model for cross correlation between voids in
                    # realspace and galaxies in redshiftspace
                    xi_vg_rsd[i, j] = (self.xi_vg_real(r) + (self.f / 3.0) * self.contrast(r) / self.bias
                                      * (1.0 + self.xi_vg_real(r))
                                      + self.f * mu**2 * (self.delta(r) - self.contrast(r)) / self.bias
                                      * (1.0 + self.xi_vg_real(r)))

        xi_vg_rsd_spline = interpolate.interp2d(mu_array, s_array, xi_vg_rsd, kind="cubic")
        
        print "Computing multipoles"
        # Compute multipoles for the theory model i redshiftspace
        for index, value in enumerate(s_array):
            xi_vg0[index] = integrate.quad(lambda mu: 1.0*xi_vg_rsd_spline(mu, value)*1.0,
                                                0,
                                                1,
                                                full_output=1)[0]
            xi_vg2[index] = integrate.quad(lambda mu: 5.0*xi_vg_rsd_spline(mu, value) * 0.5 * (3 * mu * mu - 1),
                                                0,
                                                1,
                                                full_output=1)[0]
            
        return s_array, xi_vg0, xi_vg2
        


if __name__=="__main__":
    void_file = "../350ksample/zobov-void_cat.txt"
    galaxy_file  = "../../summerproject/Haakon/galaxies_realspace_z0.42.txt"
    model = TheoryModel(50, 1024.0, 0.42, void_file, galaxy_file)
    model.xi_vg_real_func("void_cat_cutebox.txt", "galaxy_cat_cutebox.txt", "corr.txt")
    model.delta_and_sigma_vz_galaxy()
    model.contrast_galaxy()


    s, xi0, xi2 = model.correlation_rsd_theory(2.0, 0.69, 0.307, 1.0 - 0.307, alpha_par, alpha_perp, 100, 100)
    s_s, xi0_s, xi2_s = model.correlation_rsd_theory(2.0, 0.69, 0.307, 1.0 - 0.307, alpha_par, alpha_perp, 100, 100, streaming=True)

    rsd_galaxy_file = "../../summerproject/Haakon/galaxies_redshiftspace_z0.42.txt"
    x, corr, paircounts = model.compute_angular_cross_correlation("void_cat_cutebox.txt",
                                                                  rsd_galaxy_file,
                                                                  "corr.txt",
                                                                  1024.0)
    
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
    

    fig, ax = plt.subplots()
    ax.plot(s, xi2, label="Theroy model, no streaming")
    ax.plot(s_s, xi2_s, label="Theory model, streaming")
    ax.plot(r, xi2_rsd, label="Computed model")
    ax.legend()
    fig.savefig("test_xi_rsd.pdf")
