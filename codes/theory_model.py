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
    Doc-string here

    """
    def __init__(self, N, box_size, radius, bias, f, void_cat, galaxy_cat):
        self.N = N
        self.box_size = box_size
        self.radius = radius
        self.bias = bias
        self.f = f # Growth factor


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
        print np.shape(self.galaxy_vz)
        # Stacking galaxy and void catalogues as 2D arrays with x, y and z positions on
        # each column
        self.galaxy_cat = np.column_stack((galaxy_x, galaxy_y, galaxy_z))
        self.void_cat   = np.column_stack((void_x, void_y, void_z))

    def delta_galaxy(self):
        
        bounds = np.array([self.box_size, self.box_size, self.box_size])
        tree = PeriodicCKDTree(bounds, self.galaxy_cat)
        radius_array = np.linspace(0, self.r_corr[-1], self.N + 1)

        delta = np.zeros(self.N + 1)
        v_z   = np.zeros(self.N + 1) 
        E_vz  = np.zeros(self.N + 1)
        E_vz2 = np.zeros(self.N + 1)
        sigma_vz = np.zeros(self.N+1)
        galaxies_in_shell_arr = np.zeros(self.N + 1)
        print "Starting density profile calculation"

        for i in range(len(self.void_cat[:, 0])):
            current_number_of_galaxies = 0
            current_vel = 0
            current_E_vz = 0
            current_E_vz2 = 0
            E_vz_in_shell = 0
            E_vz2_in_shell = 0
            galaxy_prev_shell = 0

            for j in range(1, self.N + 1):
                # Find galaxie position and velocity in a given radius around the current void
                neighbor_inds = tree.query_ball_point(self.void_cat[i, :], r=radius_array[j])
                shell_volume = 4.0 * np.pi * (radius_array[j]**3 - radius_array[j-1]**3) / 3.0
                
                    
                #print velocity_in_shell
                velocity_near_point = self.galaxy_vz[neighbor_inds]
                galaxies_near_point = self.galaxy_cat[neighbor_inds]
                galaxies_near_point = len(galaxies_near_point[:,0])
                if galaxies_near_point > 0:
                    E_vz2_in_shell = (sum(velocity_near_point**2) - current_E_vz2)
                    E_vz_in_shell  = (sum(velocity_near_point)**2  - current_E_vz)

                # Assigning density- and expectation values for velocity values around void in a given shell
                
                galaxies_in_shell = galaxies_near_point - current_number_of_galaxies
                galaxies_in_shell_arr[j] += galaxies_in_shell
                
                E_vz [j] += E_vz_in_shell
                E_vz2[j] += E_vz2_in_shell
                delta[j] += galaxies_in_shell / shell_volume
                
                #galaxies_in_shell_arr[j] = galaxies_in_shell

                current_E_vz += E_vz_in_shell
                current_E_vz2 += E_vz2_in_shell
                current_number_of_galaxies += galaxies_in_shell
                prev_indices = neighbor_inds
            

        delta /= (len(self.void_cat[:, 0]) * len(self.galaxy_cat[:,0]) / self.box_size ** 3)
        delta -= 1
        E_vz /= len(self.void_cat[:, 0])
        E_vz2 /= len(self.void_cat[:, 0])
        galaxies_in_shell_arr /= len(self.void_cat[:,0])
        for j in range(self.N + 1):
            if galaxies_in_shell_arr[j] > 0:
                E_vz[j] /= galaxies_in_shell_arr[j]**2
                E_vz2[j] /= galaxies_in_shell_arr[j]
        self.sigma_vz = np.sqrt(E_vz2 - E_vz)
        

        fig, ax = plt.subplots()
        ax.plot(radius_array, delta)
        fig.savefig("delta_test.png")
        fig, ax = plt.subplots()
        ax.plot(radius_array, self.sigma_vz)
        fig.savefig("sigmavz_test.png")
        print "Splining density profile"
        self.delta = interpolate.interp1d(radius_array, delta, kind="cubic")
        
        return self.delta

    def contrast_galaxy(self):
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
    
    def xi_vg_real_func(self, void_file, galaxy_file, output_filename):
        """
        Calculate the 
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
                                                          
    def correlation_rsd_theory(self, n_mu, n_s, streaming = False):
        """
        Calculate the theoretical model for the cross correlation function
        between realspace voids and redshiftspace galaxies.
        """
        xi_vg_rsd = np.zeros(shape=(n_s, n_mu))
        s_array   = np.linspace(self.r_corr[0] + 1.0 , self.r_corr[-1] - 1.0, n_s) # +1 and -1 to prevent out of range on splines
        mu_array  = np.linspace(0.0, 1.0, n_mu)
        
        self.xi_vg0 = np.zeros(len(s_array))
        self.xi_vg2 = np.zeros(len(s_array))
        print "Calculating theoretical model"
        if streaming:
            print 3
        else:
            for i in range(len(s_array)):
                s = s_array[i]
                for j in range(len(mu_array)):
                    mu = mu_array[j]

                    # Hvor kommer disse likningene fra?
                    # Length coordinates in real- and redshiftspace
                    s_par  = s * mu
                    s_perp = s * np.sqrt(1 - mu**2)

                    r_par  = s_par + s * self.f * self.delta(s) * mu / (3.0 * self.bias)
                    r_perp = s_perp
                    r      = np.sqrt(r_par**2 + s_perp**2)

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
            self.xi_vg0[index] = integrate.quad(lambda mu: 1.0*xi_vg_rsd_spline(mu, value)*1.0,
                                                0,
                                                1,
                                                full_output=1)[0]
            self.xi_vg2[index] = integrate.quad(lambda mu: 5.0*xi_vg_rsd_spline(mu, value) * 0.5 * (3 * mu * mu - 1),
                                                0,
                                                1,
                                                full_output=1)[0]
            
        return s_array, self.xi_vg0, self.xi_vg2
        


if __name__=="__main__":
    void_file = "../350ksample/zobov-void_cat.txt"
    galaxy_file  = "../../summerproject/Haakon/galaxies_realspace_z0.42.txt"
    model = TheoryModel(50, 1024.0, 100.0, 2.0, 0.69, void_file, galaxy_file)
    model.xi_vg_real_func("void_cat_cutebox.txt", "galaxy_cat_cutebox.txt", "corr.txt")
    model.delta_galaxy()
    model.contrast_galaxy()
    s, xi0, xi2 = model.correlation_rsd_theory(100, 50)


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
    ax.plot(s, xi2, label="Theroy model")
    ax.plot(r, xi2_rsd, label="Computed model")
    ax.legend()
    fig.savefig("test_xi_rsd.pdf")
    ax.plot(s, xi2, label="Theroy model")
    ax.plot(r, xi2_rsd, label="Computed model")
    ax.legend()
    fig.savefig("test_xi_rsd.pdf")
