import numpy as np
import sys
sys.path.insert(1, '/mn/stornext/u3/haakota/Documents/summerproject/Python3CUTEbox')
import pycutebox
import scipy
import scipy.spatial
import sys
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
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
    
    def __init__(self, N, box_size, z, void_cat, galaxy_cat, handle=None, min_r = 0.0, max_r = 1000.0):
        
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

        self.N = N # Number of iterations for creating splines

        # catalogue specific parameters
        self.box_size = box_size # Size of simulation box
        self.z = z # Redshift
        self.min_r = min_r # Minimum radius for void cut
        self.max_r = max_r # Maximum radius for void cut
        
        self.h = 0.7
        self.H0 = 100 # km/s/ h Mpc

        self.handle = handle

        void_cat = np.loadtxt(void_cat, skiprows=2)
        
        void_x = void_cat[:, 1]
        void_y = void_cat[:, 2]
        void_z = void_cat[:, 3]
        void_radius = void_cat[:, 4]

        print "Reading galaxy catalogue"
        galaxy_cat = np.loadtxt(galaxy_cat)
    
        galaxy_x = galaxy_cat[:, 0]
        galaxy_y = galaxy_cat[:, 1]
        galaxy_z = galaxy_cat[:, 2]
        
        
        galaxy_vx = galaxy_cat[:, 3]
        galaxy_vy = galaxy_cat[:, 4]
        self.galaxy_vz = galaxy_cat[:, 5]
       
        # Stacking galaxy and void catalogues as 2D arrays with x, y and z positions on
        # each column
        self.galaxy_cat = np.column_stack((galaxy_x, galaxy_y, galaxy_z))
        void_cat   = np.column_stack((void_x, void_y, void_z))
        fig, ax = plt.subplots()
        bins=np.arange(0,100,2)
        ax.hist(void_radius,bins=bins, histtype="stepfilled")
        ax.set_xlabel("Void radius [Mpc/h]")
        ax.set_ylabel("nr in bin")
        ax.set_title(self.handle)
        #fig.savefig("figures/histograms/void_histogram" + self.handle +".pdf")
        self.void_cat = void_cat[np.logical_and(void_radius>min_r, void_radius<max_r)] # Applying cuts to the void catalog
        np.savetxt("void_cat" + self.handle + ".txt", self.void_cat)
        #np.savetxt("../MD4/zobov-void_cat" + self.handle, self.void_cat)
        self.velocity_cat = np.column_stack((galaxy_vx, galaxy_vy, self.galaxy_vz))


    def velocity_profile(self):
        radius_array = np.linspace(0, 200, self.N + 1)
        velocity_profile = np.zeros(self.N + 1)
        N_in_velocity = np.zeros(self.N + 1)

        bounds = np.array([self.box_size, self.box_size, self.box_size])
        tree = PeriodicCKDTree(bounds, self.galaxy_cat)
        print "Calculating velocity profile"
        for i in range(len(self.void_cat[:, 0])):
            #print i
            current_number_of_galaxies = 0
            current_velocity = 0
            for j in range(1, self.N + 1):
                neighbor_inds = tree.query_ball_point(self.void_cat[i, :], r=radius_array[j])
                r_void = self.void_cat[i]
                galaxies_near_point = self.galaxy_cat[neighbor_inds]
                v_galaxy = self.velocity_cat[neighbor_inds]
                r_vec  = r_void - galaxies_near_point
                galaxies_near_point = len(galaxies_near_point[:,0])
                galaxies_in_shell = galaxies_near_point - current_number_of_galaxies

                radial_velocity = (v_galaxy * r_vec).sum(axis=1) / np.linalg.norm(r_vec, axis= 1)
                radial_velocity = np.sum(radial_velocity) - current_velocity

                velocity_profile[j] += radial_velocity / np.maximum(1.0, galaxies_in_shell)
                N_in_velocity[j] += galaxies_in_shell

                current_velocity += radial_velocity
                current_number_of_galaxies += galaxies_in_shell
            #print velocity_profile / np.maximum(np.ones(self.N+1), N_in_velocity)
        v_final = (velocity_profile / len(self.void_cat[:, 0]))#/ np.maximum(np.ones(self.N+1), N_in_velocity))
        fig, ax = plt.subplots()
        ax.plot(radius_array, v_final)
        ax.set_xlabel("radius [Mpc/h]")
        ax.set_xlabel(r"$v_r(r)$ km/s")
        np.save("datafiles/velocity_profiles/velocity_profile" + self.handle, v_final)
        fig.savefig("figures/velocity_profiles/velocity_profile" + self.handle +".pdf")


    def delta_and_sigma_vz_galaxy(self, array_files=None, dictionary=False):
        """
        Calculates the density profile and velocity dispersion of voids in real space.
        Requires xi_vg_real_func() to be run first as this gives the upper and lower bounds
        for the radius array to avoid out of bounds for splines.
        """
        #radius_array = np.linspace(0, self.r_corr[-1], self.N + 1)
        radius_array = np.linspace(1, 200, self.N + 1)
        if array_files == None:
            bounds = np.array([self.box_size, self.box_size, self.box_size])
            tree = PeriodicCKDTree(bounds, self.galaxy_cat)

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
            sigma_vz[np.where(sigma_vz < 10.0)] = 100.0

            if dictionary:
                #Output for victor code
                r_dict = np.linspace(2.11, 118.0, 30)
                sigma_vz_spline = interpolate.interp1d(radius_array, sigma_vz)
                delta_spline = interpolate.interp1d(radius_array, delta)

                delta_new = delta_spline(r_dict)
                sigma_vz_new = sigma_vz_spline(r_dict)
                vr_dict = {}
                vr_dict["rvals"]  = r_dict
                vr_dict["sigma_v_los"] = sigma_vz_new
                np.save("datafiles/velocity_profiles/sigma_vz_dict" + self.handle, vr_dict)

                delta_dict = {}
                delta_dict["rvals"] = r_dict
                delta_dict["delta"] = delta_new
                np.save("datafiles/density_profiles/delta_dict" + self.handle, delta_dict)

            fig, ax = plt.subplots()
            ax.plot(radius_array, delta)
            fig.savefig("delta_test.png")
            fig, ax = plt.subplots()
            ax.plot(radius_array, sigma_vz)
            fig.savefig("sigmavz_test.png")
            print len(delta)
            np.save("datafiles/density_profiles/delta" + self.handle, delta)
            np.save("datafiles/velocity_profiles/sigma_vz" + self.handle, sigma_vz)
        else:
            
            delta = np.load(array_files[0])
            sigma_vz = np.load(array_files[1])
            fig, ax = plt.subplots()
            ax.plot(radius_array, delta)
            fig.savefig("delta_test.png")
            fig, ax = plt.subplots()
            ax.plot(radius_array, sigma_vz)
            fig.savefig("sigmavz_test.png")
            

        print "Splining density profile"
        print len(radius_array), len(delta)
        self.delta = interpolate.interp1d(radius_array, delta, kind="cubic")
        self.sigma_vz = interpolate.interp1d(radius_array, sigma_vz, kind="cubic")
    

        return self.delta, self.sigma_vz

    def contrast_galaxy(self, array_file=None):
        """
        Calculates the density contrast given by equation 4 in S.Nadathur et al 2019.
        Requires the delta_and_sigma_vz_galaxy to be run first as this gives the density profile
        required in the integral.
        """
        #r = np.linspace(self.r_corr[0], self.r_corr[-1], self.N)
        r = np.linspace(2, 199, self.N)
        print self.delta(2)
        if array_file == None:
            contrast = np.zeros(len(r))
            print "calculating density contrast"

            # calculating the density contrast given by eq (4).
            for index, radius in enumerate(r):
                if radius == 0.0:
                    # Avoid division by zero
                    contrast[index] = - 1.0
                else:
                    #print self.delta(radius)
                    #contrast[index] = 3.0 * integrate.quad(lambda radius: self.delta(radius) * radius**2, r[0], radius, full_output=1)[0] / radius**3
                    contrast[index] = self.delta(r[0])*(r[0]/r[index])**3 + 3.0 * integrate.quad(lambda radius: self.delta(radius) * radius**2, r[0], radius, full_output=1)[0] / radius**3
            fig, ax = plt.subplots()
            ax.plot(r, contrast)
            fig.savefig("contrast_test.png")
            np.save("datafiles/density_profiles/contrast" + self.handle, contrast)
        else:
            contrast = np.load(array_file)
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
    
    def xi_vg_real_func(self, galaxy_file, void_file=None, array_file=None):
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
        if void_file == None:
            void_file = "void_cat" + self.handle + ".txt"
        if array_file == None:
            x, corr, paircounts = self.compute_angular_cross_correlation(void_file,
                                                                         galaxy_file,
                                                                         "corr.txt",
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
            np.save("xi_vg_real" + self.handle, xi_vg_real)
            np.save("r_corr" + self.handle, self.r_corr)
            fig, ax = plt.subplots()
            ax.plot(self.r_corr, xi_vg_real)
            plt.savefig("xi_vg_real_test.png")
        else:
            xi_vg_real = np.load(array_file)
            self.r_corr = np.load("r_corr" + self.handle + ".npy")
        fig, ax = plt.subplots()
        ax.plot(self.r_corr, xi_vg_real)
        fig.savefig("xi_r_test.png")

        self.xi_vg_real = interpolate.interp1d(self.r_corr, xi_vg_real, kind="cubic")
        return self.xi_vg_real

    def H_of_z_func(self, Omega_m, Omega_Lambda):
        """
        Function return the Hubble parameter (H)
        as a function of redshift z.
        """
        #print(self.H0)
        return self.H0 * np.sqrt(Omega_m * (1.0 + self.z)**3 + Omega_Lambda)


    def convert_splines(self, r_real, r_fid, r_scale=None):
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
        #r_real[np.where(r_real > self.r_corr[-1])] = self.r_corr[-1]
        #r_real[np.where(r_real < self.r_corr[0])] = self.r_corr[0]

        xi_vg_real = self.xi_vg_real(r_fid)
        delta = self.delta(r_fid)
        contrast = self.contrast(r_fid)
        sigma_vz = self.sigma_vz(r_fid)
        #print r_real
        xi_vg_real = interpolate.interp1d(r_real, xi_vg_real, kind="cubic")
        delta = interpolate.interp1d(r_real, delta, kind="cubic")
        contrast = interpolate.interp1d(r_real, contrast, kind="cubic")
        sigma_vz = interpolate.interp1d(r_real, sigma_vz, kind="cubic")
        
        return xi_vg_real, delta, contrast, sigma_vz


    def correlation_rsd_theory(self, beta, Omega_m, Omega_Lambda, epsilon, n_mu, n_s, streaming = False, Achitouv=False):
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
            number of points for a linearly space s array where s is distance in redshiftspace.
            Upper and lower bounds is determined by the radius array given by the CUTEbox correlation function
            in xi_vg_real_func().

        streaming:
            Determines wether on should use the full theory model with a gaussian streaming profile or the simpler theory model.
        """
        # Rescaling radius in correspondance with equation 12 is S.Nadathur et
        # al 2019
        alpha_par = epsilon ** (-2./3)
        alpha_perp = epsilon * alpha_par
        mu_array  = np.linspace(0.0, 1.0, n_mu)
        r_fid = np.linspace(self.r_corr[0], self.r_corr[-1], n_s)
        r_integrand = alpha_par * np.sqrt(1 + (1 - mu_array**2) * ((alpha_par/alpha_perp)**2 - 1))
        r_factor = np.trapz(r_integrand, mu_array)
        r_real  = r_fid * r_factor

        # Since splines are stored as class variables, they should only be
        # converted once.
        
        xi_vg_real, delta, contrast, sigma_vz = self.convert_splines(r_real, r_fid)

        xi_vg_rsd = np.zeros(shape=(n_s, n_mu))

        s_array   = np.linspace(r_real[0] + 1.0 , r_real[-1] - 1.0, n_s) # +1 and -1 to prevent out of range on splines
        
        contrast_squared = np.zeros(len(r_fid)) 
        for index, radius in enumerate(r_real):
                if radius == 0.0:
                    # Avoid division by zero
                    contrast_squared[index] = - 1.0
                else:
                    #print self.delta(radius)
                    #contrast[index] = 3.0 * integrate.quad(lambda radius: self.delta(radius) * radius**2, r[0], radius, full_output=1)[0] / radius**3
                    contrast_squared[index] = delta(r_real[0])**2*(r_real[0]/r_real[index])**6 + 3.0 * integrate.quad(lambda radius: delta(radius)**2 * radius**2, r_real[0], radius, full_output=1)[0] / radius**3
        contrast_squared  = interpolate.interp1d(r_real, contrast_squared)
        xi_vg0 = np.zeros(len(s_array))
        xi_vg2 = np.zeros(len(s_array))
        #print "Calculating theoretical model"
        if Achitouv:
            # Calculate equation 7 in S.Nadathur et al 2019
            aH = (1.0 / (1.0 + self.z)) * self.H_of_z_func(Omega_m, Omega_Lambda) # Hubble parameter times scale factor (a = 1 / (1 + z))
            aH /= alpha_par
            for i in range(len(s_array)):
                s = s_array[i]
                for j in range(len(mu_array)):
                    mu = mu_array[j]

                    # Performing change of variable where y = v_par / (a * H)
                    # giving dv_par = a*H*dy
                    
                    sigma_tilde = sigma_vz(s) / aH
                    
                    y = np.linspace(-5.0 * sigma_tilde, 5.0 * sigma_tilde, 100) # Integration domain determined by gaussian 
                                                                                # part of the integrand. It is basically zero elsewhere
                                                                                                         
                    # Length coordinates in real- and redshiftspace
                    s_par  = s * mu * alpha_par
                    s_perp = s * np.sqrt(1 - mu**2) * alpha_perp

                    r_par  = s_par + s * beta * delta(s) * mu / (3.0) - y
                    r_perp = s_perp
                    r      = np.sqrt(r_par**2 + r_perp**2)
        
                    # Avoid out of range on splines
                    r[np.where(r > r_real[-1])] = r_real[-1]
                    r[np.where(r < r_real[0])] = r_real[0]

                    eps = np.ones_like(r)
                    eps[contrast(r) < 0] *= -1.0
                    dy = np.gradient(y, r)
                    
                    v_r = (- beta * contrast(r) / 3.0 + 3.0 * eps * 2.0 * beta * delta(r) / (42.0 * 2.11) + y * mu / r)

                    # this term corresponds to 1/aH times derivative wrt r of radial outflow velocity component
                    v_prime = (-beta * (delta(r) - 2.0 * contrast(r) / 3.0) + 9.0 * eps * 2.0 * beta * (contrast(r) - 2.0 * contrast_squared(r) / 3.0) / (42 * 2.11) + (dy / r) * mu)
                    # Collecting terms with velocity dispersion added.
                    xi_s_perp_s_par = (1 + xi_vg_real(r)) * (1.0 / (1.0 + v_r + (v_prime - v_r / r) * mu ** 2.0)) 
                    integrand = (1 + xi_s_perp_s_par) / ((np.sqrt(2 * np.pi) * sigma_tilde))
                    integrand *= np.exp(-0.5 * y**2 / sigma_tilde**2)
                    xi_vg_rsd[i, j] = np.trapz(integrand, y) - 1

        elif streaming:
            # Calculate equation 7 in S.Nadathur et al 2019
            aH = (1.0 / (1.0 + self.z)) * self.H_of_z_func(Omega_m, Omega_Lambda) # Hubble parameter times scale factor (a = 1 / (1 + z))
            aH /= alpha_par
            for i in range(len(s_array)):
                s = s_array[i]
                for j in range(len(mu_array)):
                    mu = mu_array[j]

                    # Performing change of variable where y = v_par / (a * H)
                    # giving dv_par = a*H*dy
                    
                    sigma_tilde = sigma_vz(s) / aH
                    
                    y = np.linspace(-5.0 * sigma_tilde, 5.0 * sigma_tilde, 100) # Integration domain determined by gaussian 
                                                                                # part of the integrand. It is basically zero elsewhere
                                                                                                          
                    # Length coordinates in real- and redshiftspace
                    s_par  = s * mu * alpha_par
                    s_perp = s * np.sqrt(1 - mu**2) * alpha_perp

                    r_par  = s_par + s * beta * delta(s) * mu / (3.0) - y
                    r_perp = s_perp
                    r      = np.sqrt(r_par**2 + r_perp**2)
                    dy = np.gradient(y, r)
                    
                    # Avoid out of range on splines
                    r[np.where(r > r_real[-1])] = r_real[-1]
                    r[np.where(r < r_real[0])] = r_real[0]

                    v_r = (- beta * contrast(r) / 3.0  + y * mu / r)

                    # this term corresponds to 1/aH times derivative wrt r of radial outflow velocity component
                    v_prime = (-beta * (delta(r) - 2.0 * contrast(r) / 3.0) + (dy / r) * mu)

                    # Collecting terms with velocity dispersion added.
                    xi_s_perp_s_par = (1 + xi_vg_real(r)) * (1.0 / (1.0 + v_r + (v_prime - v_r / r) * mu ** 2.0)) 
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
                    s_par  = s * mu * alpha_par
                    s_perp = s * np.sqrt(1 - mu**2) * alpha_perp

                    r_par  = s_par + s * f * delta(s) * mu / (3.0 * bias)
                    r_perp = s_perp
                    r      = np.sqrt(r_par**2 + r_perp**2)
                    
                    # Base theory model for cross correlation between voids in
                    # realspace and galaxies in redshiftspace
                    xi_vg_rsd[i, j] = (xi_vg_real(r) + (f / 3.0) * contrast(r) / bias
                                      * (1.0 + xi_vg_real(r))
                                      + f * mu**2 * (delta(r) - contrast(r)) / bias
                                      * (1.0 + xi_vg_real(r)))

        xi_vg_rsd_spline = interpolate.interp2d(mu_array, s_array, xi_vg_rsd, kind="cubic")
        
        #print "Computing multipoles"
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

        def correlation_rsd_theory_DM(self, fs8, r_scale, Omega_m, Omega_Lambda, epsilon, n_mu, n_s, streaming = False):
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
                number of points for a linearly space s array where s is distance in redshiftspace.
                Upper and lower bounds is determined by the radius array given by the CUTEbox correlation function
                in xi_vg_real_func().

            streaming:
                Determines wether on should use the full theory model with a gaussian streaming profile or the simpler theory model.
            """
            # Rescaling radius in correspondance with equation 12 is S.Nadathur et
            # al 2019
            alpha_par = epsilon ** (-2./3)
            alpha_perp = epsilon * alpha_par
            mu_array  = np.linspace(0.0, 1.0, n_mu)
            r_fid = np.linspace(self.r_corr[0], self.r_corr[-1], n_s)
            r_integrand = alpha_par * np.sqrt(1 + (1 - mu_array**2) * ((alpha_par/alpha_perp)**2 - 1))
            r_factor = np.trapz(r_integrand, mu_array)
            r_real  = r_fid * r_factor

            # Since splines are stored as class variables, they should only be
            # converted once.

            xi_vg_real, delta, contrast, sigma_vz = self.convert_splines(r_real, r_fid)
            xi_vg_rsd = np.zeros(shape=(n_s, n_mu))

            s_array   = np.linspace(r_real[0] + 1.0 , r_real[-1] - 1.0, n_s) # +1 and -1 to prevent out of range on splines


            xi_vg0 = np.zeros(len(s_array))
            xi_vg2 = np.zeros(len(s_array))
            #print "Calculating theoretical model"

            # Calculate equation 7 in S.Nadathur et al 2019
            aH = (1.0 / (1.0 + self.z)) * self.H_of_z_func(Omega_m, Omega_Lambda) # Hubble parameter times scale factor (a = 1 / (1 + z))
            aH /= alpha_par
            for i in range(len(s_array)):
                s = s_array[i]
                for j in range(len(mu_array)):
                    mu = mu_array[j]
                    # Performing change of variable where y = v_par / (a * H)
                    # giving dv_par = a*H*dy

                    sigma_tilde = sigma_vz(s) / aH

                    y = np.linspace(-5.0 * sigma_tilde, 5.0 * sigma_tilde, 100) # Integration domain determined by gaussian 
                                                                                # part of the integrand. It is basically zero elsewhere

                    # Length coordinates in real- and redshiftspace
                    s_par  = s * mu * alpha_par
                    s_perp = s * np.sqrt(1 - mu**2) * alpha_perp
                    r_par  = s_par + s * beta * delta(s) * mu / (3.0) - y
                    r_perp = s_perp
                    r      = np.sqrt(r_par**2 + r_perp**2)

                    # Avoid out of range on splines
                    r[np.where(r > r_real[-1])] = r_real[-1]
                    r[np.where(r < r_real[0])] = r_real[0]

                    # Collecting terms with velocity dispersion added.
                    xi_s_perp_s_par = ((1 + xi_vg_real(r)) * (1 + (beta * contrast(r)/ 3.0
                                        - y * mu / r) * (1 - mu**2)
                                        + beta * (delta(r) - 2 * contrast(r) / 3.0) * mu**2))
                    integrand = (1 + xi_s_perp_s_par) / ((np.sqrt(2 * np.pi) * sigma_tilde))
                    integrand *= np.exp(-0.5 * y**2 / sigma_tilde**2)
                    xi_vg_rsd[i, j] = np.trapz(integrand, y) - 1

            xi_vg_rsd_spline = interpolate.interp2d(mu_array, s_array, xi_vg_rsd, kind="cubic")

            #print "Computing multipoles"
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
    
    void_file = "../MD4/zobov-void_cat.txt"
    galaxy_file  = "../../summerproject/Haakon/MultiDarkSimulations/HaloSample4/halos_realspace_z1.txt"
    """
    model = TheoryModel(50, 2500.0, 1.0, void_file, galaxy_file, "MD2")
    model.xi_vg_real_func("MD2void_real.txt", "MD2galaxy_real.txt")

    """
    fonts = {

    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    'text.usetex': True, 
    }
    plt.rcParams.update(fonts)
    """
    # FIGURE 1
    model = TheoryModel(60, 2500.0, 1.0, void_file, galaxy_file, "MD2", min_r=0.0, max_r=200.0)
    model.xi_vg_real_func("MD2galaxy_real.txt")
    model.delta_and_sigma_vz_galaxy(dictionary=False, array_files=["datafiles/density_profiles/deltaMD2.npy", "datafiles/velocity_profiles/old/sigma_vzMD2.npy"])
    model.contrast_galaxy()

    #model.velocity_profile()
    #model.contrast_galaxy()
    

    
    fig, ax = plt.subplots()
    #s, xi0, xi2 = model.correlation_rsd_theory(0.872, 2.40, 0.307, 0.692, alpha_par, alpha_perp, 100, 100)
    s_s, xi0_s, xi2_s = model.correlation_rsd_theory(0.30180, 0.307, 0.692, 1.00351, 100, 100, streaming=True)
    ax.plot(s_s, xi2_s, label=r"Streaming model")
    s_s, xi0_s, xi2_s = model.correlation_rsd_theory(0.31023, 0.307, 0.692, 0.99380, 100, 100, streaming=False, Achitouv=True)
    ax.plot(s_s, xi2_s, label=r"Achitouv 2017 term")


    rsd_galaxy_file = "../../summerproject/Haakon/MultiDarkSimulations/HaloSample2/halos_redshiftspace_z1.txt"
    x, corr, paircounts = model.compute_angular_cross_correlation("MD2void_real.txt",
                                                                  rsd_galaxy_file,
                                                                  "corr.txt",
                                                                  2500.0)
    
    # Fetch the data
    r  = x[0]
    mu = x[1]
    corr_of_mu_s = interpolate.interp2d(mu, r, corr, kind='cubic')


    xi0_rsd = np.zeros(len(r))
    xi2_rsd = np.zeros(len(r))
    ax.set_xlabel("s [Mpc/h]")
    ax.set_ylabel(r"$xi ^{s}_{vg}$")


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
    

    #ax.plot(s, xi2, label="Theroy model, no streaming")
    ax.plot(r, xi2_rsd, label="pyCUTE")
    ax.legend()
    fig.suptitle("MD2.")
    fig.savefig("xi_rsd_MD2.pdf")
    """
    # FIGURE 2
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    fig, ax = plt.subplots()
    model = TheoryModel(60, 2500.0, 1.0, void_file, galaxy_file, "MD4", min_r=0.0, max_r=200.0)
    model.delta_and_sigma_vz_galaxy(dictionary=False, array_files=["datafiles/density_profiles/deltaMD4.npy", "datafiles/velocity_profiles/old/sigma_vzMD4.npy"])
    model.xi_vg_real_func("MD4galaxy_real.txt")
    #model.delta_and_sigma_vz_galaxy(dictionary=True)
    model.contrast_galaxy()


    s_s, xi0_s, xi2_s = model.correlation_rsd_theory(0.30509, 0.307, 0.692, 1.00354, 100, 100, streaming=True)
    ax.plot(s_s, xi2_s, label=r"Streaming model")
    s_s, xi0_s, xi2_s = model.correlation_rsd_theory(0.31876, 0.307, 0.692, 0.99136, 100, 100, streaming=False, Achitouv=True)
    ax.plot(s_s, xi2_s, label=r"Achitouv 2017 term")
    rsd_galaxy_file = "../../summerproject/Haakon/MultiDarkSimulations/HaloSample4/halos_redshiftspace_z1.txt"
    x, corr, paircounts = model.compute_angular_cross_correlation("void_catMD4.txt",
                                                                  rsd_galaxy_file,
                                                                  "corr.txt",
                                                                  2500.0)
    
    # Fetch the data
    r  = x[0]
    mu = x[1]
    corr_of_mu_s = interpolate.interp2d(mu, r, corr, kind='cubic')

    xi0_rsd = np.zeros(len(r))
    xi2_rsd = np.zeros(len(r))
    ax.set_xlabel("s [Mpc/h]")
    ax.set_ylabel(r"$\xi_2^s(s)$")

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
    

    #ax.plot(s, xi2, label="Theroy model, no streaming")
    ax.plot(r, xi2_rsd, label="pyCUTE")
    ax.legend()
    fig.suptitle(r"MD4")
    fig.savefig("xi_rsd_MD4_correct.pdf")   
    