from Worker import WorkerClass
import numpy as np
from scipy.optimize import minimize

class GovernmentClass(WorkerClass):

    def __init__(self,par=None):

        # a. defaul setup
        self.setup_worker()
        self.setup_government()

        # b. update parameters
        if not par is None: 
            for k,v in par.items():
                self.par.__dict__[k] = v

        # c. random number generator
        self.rng = np.random.default_rng(12345)

    def setup_government(self):

        par = self.par

        # a. workers
        par.N = 100  # number of workers
        par.sigma_p = 0.3  # std dev of productivity

        # b. pulic good
        par.chi = 50.0 # weight on public good in SWF
        par.eta = 0.1 # curvature of public good in SWF

    def draw_productivities(self): # Exercise 2.1

        par = self.par
        sol = self.sol

        par = self.par

        mu = -0.5*par.sigma_p**2
        sigma = par.sigma_p

        # draw N productivities and store them
        sol.p = np.exp(self.rng.normal(loc=mu, scale=sigma, size=par.N))

    def solve_workers(self):
        par, sol = self.par, self.sol
        sol.ell = np.empty(par.N)
        sol.u   = np.empty(par.N)
        sol.c   = np.empty(par.N)

        nu_backup = par.nu # Exercise 4

        for i, p_i in enumerate(sol.p):
            opt = self.optimal_choice(p_i)
            sol.ell[i] = opt.ell
            sol.u[i]   = opt.U
            sol.c[i]   = opt.c

        par.nu = nu_backup # Exercise 4


    def tax_revenue(self): # Exercise 2.1

        par = self.par
        sol = self.sol

        tax_revenue = 0.0
        labor_tax = par.tau * par.w * np.sum(sol.p * sol.ell)
        lump_sum  = par.N * par.zeta

        tax_revenue = lump_sum + labor_tax

        return tax_revenue
    
    def SWF(self): # Exercise 2.1

        par = self.par
        sol = self.sol

        G =  self.tax_revenue()
        if G < 0:
            SWF = np.nan
        else:
            SWF = par.chi * (G**par.eta) + np.sum(sol.u)

        return SWF
    
    def optimal_taxes(self,tau,zeta): # Exercise 2.2

        par = self.par
        sol = self.sol

        # a. objective function
        def obj(x):

            par.tau = x[0]
            par.zeta = x[1]

            p_min = np.min(sol.p) # Constraint
            zeta_max = (1.0 - par.tau)*par.w*p_min*par.ell_max
            if par.zeta >= zeta_max:
                return 1e6 
            
            self.solve_workers() # Solve for worker and SWF
            SWF_val = self.SWF()

            if np.isnan(SWF_val):
                return 1e6
            
        
            return -SWF_val
                     
        x0 = np.array([tau, zeta])
        bounds = [(0.0, 0.99), (-2.0, 2.0)]

        res = minimize(obj, x0, bounds=bounds)

        par.tau, par.zeta = res.x
        tau_star, zeta_star = res.x
        SWF_star = -res.fun

        return tau_star, zeta_star, SWF_star