from types import SimpleNamespace

import numpy as np

from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar

class WorkerClass:

    def __init__(self,par=None):

        # a. setup
        self.setup_worker()

        # b. update parameters
        if not par is None: 
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup_worker(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preferences
        par.nu = 0.015 # weight on labor disutility
        par.epsilon = 1.0 # curvature of labor disutility
        
        # b. productivity and wages
        par.w = 1.0 # wage rate
        par.ps = np.linspace(0.5,3.0,100) # productivities
        par.ell_max = 16.0 # max labor supply
        
        # c. taxes
        par.tau = 0.50 # proportional tax rate
        par.zeta = 0.10 # lump-sum tax
        par.kappa = np.nan # income threshold for top tax
        par.omega = 0.20 # top rate rate

    def utility(self, c, ell): # Exercise 1.1.1

        par = self.par

        u = np.log(c) - par.nu * (ell**(1 + par.epsilon)) / (1 + par.epsilon)
        
        return u
    
    def tax(self,pre_tax_income):

        par = self.par

        tax = par.tau * pre_tax_income + par.zeta

        return tax
    
    def income(self,p,ell):
        par = self.par

        income = par.w * p * ell

        return income

    def post_tax_income(self,p,ell):

        pre_tax_income = self.income(p,ell)
        tax = self.tax(pre_tax_income)

        return pre_tax_income - tax
    
    def max_post_tax_income(self,p):

        par = self.par
        return self.post_tax_income(p,par.ell_max)

    def value_of_choice(self,p,ell):

        par = self.par

        c = self.post_tax_income(p,ell)
        U = self.utility(c,ell)

        return U
    
    def get_min_ell(self,p): # Exercise 1.1
    
        par = self.par

        min_ell = par.zeta/(par.w*p*(1-par.tau))

        return np.fmax(min_ell,0.0) + 1e-8
    
    def optimal_choice(self, p): # Exercise 1.1
        par = self.par
        opt = SimpleNamespace()

        def objective(ell):
            return -self.value_of_choice(p, ell)

        min_ell = self.get_min_ell(p)
        bounds = (min_ell, par.ell_max)
        res = minimize_scalar(objective, bounds=bounds, method='bounded')

        # results
        opt.ell = res.x
        opt.U = -res.fun
        opt.c = self.post_tax_income(p, opt.ell)
        return opt
    
    def optimal_choice_FOC(self,p): # Exercise 1.1

        par = self.par
        opt = SimpleNamespace()

        min_ell = self.get_min_ell(p)


        return opt
    
    def optimal_choice_numerical(self, p): # Exercise 1.1
        par = self.par
        opt = SimpleNamespace()

        # Objective function (negative for minimization)
        def objective(ell):
            return -self.value_of_choice(p, ell)

        # Bounds and minimization
        min_ell = self.get_min_ell(p)
        bounds = (min_ell, par.ell_max)
        res = minimize_scalar(objective, bounds=bounds, method='bounded')

        # Results
        opt.ell = res.x
        opt.U = -res.fun
        opt.c = self.post_tax_income(p, opt.ell)
        opt.method = "Numerical Optimizer"
        return opt
    
    def FOC(self, p, ell): # Exercise 1.1
        par = self.par
        c = self.post_tax_income(p, ell)
        FOC = ((1 - par.tau) * par.w * p) / c - par.nu * (ell**par.epsilon)
        return FOC
    
    def optimal_choice_root(self, p): # Exercise 1.1
        par = self.par
        opt = SimpleNamespace()

        min_ell = self.get_min_ell(p)
        
        # Try to find root of FOC
        try:
            res = root_scalar(lambda ell: self.FOC(p, ell), 
                            bracket=[min_ell, par.ell_max], 
                            method='brentq')
            opt.ell = res.root
            opt.converged = res.converged
        except:
            # If no solution in interior, check corners
            U_min = self.value_of_choice(p, min_ell)
            U_max = self.value_of_choice(p, par.ell_max)
            if U_min > U_max:
                opt.ell = min_ell
            else:
                opt.ell = par.ell_max
            opt.converged = False

        opt.U = self.value_of_choice(p, opt.ell)
        opt.c = self.post_tax_income(p, opt.ell)
        opt.method = "Root Finder"
        return opt
    
    def FOC(self,p,ell): # Exercise 1.1

        par = self.par

        c = self.post_tax_income(p,ell) 
        FOC = ((1 - par.tau) * par.w * p) / c - par.nu * (ell**par.epsilon)

        return FOC
    
    def post_tax_income_top(self,p,ell): # Exercise 3.1
        par = self.par
        income = par.w * p * ell                   
        extra_base = max(income - par.kappa,0.0)  
        y = (1 - par.tau)*income - par.omega*extra_base - par.zeta
        return y
    
    
    def optimal_choice_top_FOC(self,p): # Exercise 3.1
            
        par = self.par
        opt = SimpleNamespace()

        # cutoff
        ell_k = par.kappa/(par.w*p)
        eps = 1e-8

        # Utility
        def U_from_ell(ell):
            c = self.post_tax_income_top(p,ell)
            if c <= 0:
                return -np.inf
            return self.utility(c,ell)

        # Step 1
        Ub = -np.inf
        ell_b = None

        a1 = eps
        b1 = min(ell_k - eps, par.ell_max)
        if b1 > a1:
            def phi_low(ell):
                c = self.post_tax_income_top(p,ell)
                return ((1 - par.tau)*par.w*p)/c - par.nu*(ell**par.epsilon)

            try:
                res1 = root_scalar(phi_low, bracket=[a1,b1], method='brentq')
                if res1.converged:
                    ell_b = res1.root
                    Ub = U_from_ell(ell_b)
            except ValueError:
                pass

        # Step 2
        ell_k_eff = np.clip(ell_k, eps, par.ell_max)
        Uk = U_from_ell(ell_k_eff)

        # Step 3
        Ua = -np.inf
        ell_a = None

        a2 = max(ell_k + eps, eps)
        b2 = par.ell_max
        if b2 > a2:
            def phi_high(ell):
                c = self.post_tax_income_top(p,ell)
                return ((1 - par.tau - par.omega)*par.w*p)/c - par.nu*(ell**par.epsilon)

            try:
                res2 = root_scalar(phi_high, bracket=[a2,b2], method='brentq')
                if res2.converged:
                    ell_a = res2.root
                    Ua = U_from_ell(ell_a)
            except ValueError:
                pass

        #  Step 4
        utilities = [Ub,    Uk,        Ua]
        ells      = [ell_b, ell_k_eff, ell_a]

        idx_best = int(np.argmax(utilities))
        ell_star = ells[idx_best]
        U_star   = utilities[idx_best]
        c_star   = self.post_tax_income_top(p,ell_star)

        opt.ell = ell_star
        opt.U   = U_star
        opt.c   = c_star
        opt.method = "4-step FOC with top tax"

        return opt
