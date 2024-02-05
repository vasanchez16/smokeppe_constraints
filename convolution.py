import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

class conv_gauss_t:
    """
    Used to analyze the data that is distributed along a convolved gaussian and student-t
    distribution.
    """
    def __init__(self,dists,varis):
        """
        Numerical convolution of gaussian and student-t distributions.

        Arguments:
            dists: numpy array
            Array of distance calculations for satellite vs emulator.
            varis: numpy array
            Array of summed variances for satellite and emulator.
        """
        self.dists = dists
        self.varis = varis

        # Remove nan values
        try:
            self.dists_filt = dists[~np.isnan(dists)]
            self.varis_filt = varis[~np.isnan(dists)]
        except:
            raise('TypeError: dists and varis must be arrays')

        # form list of tuples pairing each distance respective variance
        pairs = [(i,j) for i,j in zip(self.dists_filt,self.varis_filt)]
        # sort these pairs in ascending order of distances
        self.pairs = sorted(pairs,key = lambda x: x[0])

        # save dists and varis after sorting and filtering
        self.dists_filt_sort = np.array([i[0] for i in self.pairs])
        self.varis_filt_sort = np.array([i[1] for i in self.pairs])
        self.std_filt_sort = np.sqrt(self.varis_filt_sort)
    
    #---------------------------Statistic Functions---------------------------------

    def gauss_pdf(self,x,sigma,mu = 0,epsilon = 0):
        """
        Calculates the value for the gaussian pdf at a certain sample value.

        Arguments:
            x: float
            Value of sample in the gaussian distribution.
            sigma: float
            Standard deviation of gaussian distribution.
            mu: float
            Mean of the gaussian distribution, for the use here it is always zero as 
            Satellite and Emulator disagreements should be distributed around zero.
            epsilon: float
            Mean correction term used to shift the distribution if it is skewed to one side,
            for example satellites often overpredict emulator values.

        Returns:
            f_N: float
            pdf value for given sample.
        """
        coeff = 1 / np.sqrt(2*np.pi*sigma**2)

        exp_factor = np.exp((-1/2) * (x-mu+epsilon)**2 / sigma**2)

        f_N = coeff * exp_factor
        return f_N
    
    def t_pdf(self,y,sigma,nu,mu=0,epsilon = 0):
        """
        Calculates the value for student-t distribution pdf at a certain sample value.

        Arguments:
            y: float
            Value of sample in the student-t distribution.
            sigma: float
            Standard deviation of student-t distribution.
            mu: float
            Mean of the student-t distribution, for the use here it is always zero as 
            Satellite and Emulator disagreements should be distributed around zero.
            epsilon: float
            Mean correction term used to shift the distribution if it is skewed to one side,
            for example satellites often overpredict emulator values.

        Returns:
            f_t: float
            pdf value for given sample.
        """
        coeff = scipy.special.gamma((nu + 1) / 2) / (scipy.special.gamma(nu/2) * np.sqrt(np.pi * nu * sigma**2))

        factor2 = 1 + (1/nu) * ((y-mu+epsilon)**2) / sigma**2

        f_t = coeff * factor2**(-1*(nu+1)/2)
        return f_t
    
    def conv_pdf(self,z,sigma_t,nu_t):
        """
        Arguments:
            z: float
            Sample from the convolved distribution.
            simga_t: float
            Standard deviation for the t distribution.
            nu_t: int
            Degrees of freedom for the t distribution

        Return: 
            fz: float
            Probability of sample value z in convolved distribution.
        """
        f_t_here = self.t_pdf(z-self.dists_filt_sort,sigma_t,nu_t)

        fz_to_sum = self.gauss_vals*f_t_here
        # currently a left reimann sum
        fz_to_sum = fz_to_sum[:-1]*self.dx
        
        return sum(fz_to_sum)
    
    def negative_likelihood(self,dec_vars):
        """
        Calulate the log-Likelihood value for a given standard deviation and 
        degrees of freedom for the student-t distribution.

        Arguments:
            dec_vars: [float, int]
            Two element iterable containing values for:
            [student-t st. deviation, student-t dof]
        
        Returns:
            -1*logL_sum: float
            Negative log Likelihood, this is used here to allow for the 
            implementation of the scipy.optimize.minimize function to 
            therefore find the maximum likelihood.
        """

        # initiate the decision variables
        simga_opt = dec_vars[0]
        nu_opt = dec_vars[1]

        # Get gauss pdf values
        gauss_pdf_vals = self.gauss_vals
        # get t pdf values
        t_pdf_vals = self.t_pdf(self.z_minus_x,simga_opt,nu_opt)
        # conv pdf vals
        fz = gauss_pdf_vals * t_pdf_vals
        fz = fz[:,:-1] * self.dx
        L_i = np.sum(fz,axis=1)
        # quantify likelihood
        logL_i = np.log(L_i)
        logL_sum = sum(logL_i)

        return -1*logL_sum
    
    def negative_likelihood_mod(self,z_minus_x_samples):
        fz = self.gauss_vals * self.t_pdf(z_minus_x_samples,self.sigma_tune,self.nu_tune)
        fz = fz[:,:-1] * self.dx
        L_i = np.sum(fz,axis=1)
        # quantify likelihood
        logL_i = np.log(L_i)
        logL_sum = sum(logL_i)
        return -1*logL_sum
    
    #------------------------------Optimization Functions---------------------------------

    def tune_this(self,simga_opt,nu_opt):
        self.sigma_tune = simga_opt
        self.nu_tune = nu_opt
        
    def opt_func(self,dec_vars):
        self.tune_this(dec_vars[0],dec_vars[1])
        with ProcessPoolExecutor() as executor:
            logL_batches = executor.map(self.negative_likelihood_mod,self.batches)
            neg_logL = sum(logL_batches)
        return neg_logL
    
    def run_opt(self,batch_size=1000,x_0=[0.02,5]):
        """
        Optimizes the student-t distribution parameters to maximize log likelihood

        Arguments:
            x_0: [float, int]
            Two element iterable containing guesses for minimize function 
            which are:
            [student-t st. deviation, student-t dof]

        Returns:
            self.res: scipy.optimize.minimize results obj
            The results output of the optimization call.
        """
        self.find_z_minus_x_difs()
        self.get_batches(batch_size)
        self.calc_dx()
        self.calc_gauss_vals()
        # x_0 = [.02, 5]
        self.res = minimize(self.opt_func,x_0,bounds=[(0,1),(0,150)],method='Powell')
        return self.res
    
    #---------------------------------Helper Functions------------------------------------
        
    def calc_dx(self):
        """
        Calculates dx array to be used in the convolution integral.
        Where integral is: 
        f_Z(z) = sum_over_x(f_X(x)*f_Y(z-x)*dx)
        """
        sorted_dists = [ i[0] for i in self.pairs]
        self.dx = np.array([sorted_dists[i+1] - sorted_dists[i] for i in range(len(sorted_dists)-1)])

    def calc_gauss_vals(self):
        """
        Calculate the gaussian pdf values for all samples in the dists_filt_sort 
        array.
        """
        self.gauss_vals = self.gauss_pdf(self.dists_filt_sort,self.varis_filt_sort)
    
    def find_z_minus_x_difs(self):
        """
        Calculate the z-x values for the convolution integral, where the integral
        is:
        f_Z(z) = sum_over_x(f_X(x)*f_Y(z-x)*dx)
        """
        self.z_minus_x = self.dists_filt_sort[:,np.newaxis] - self.dists_filt_sort

    def get_batches(self,batch_size):
        batch_list = []
        step = 0
        while len(batch_list)*batch_size < len(self.z_minus_x):
            batch_list.append(self.z_minus_x[step*batch_size:(step+1)*batch_size,:])
            step += 1
        self.batches = batch_list
    