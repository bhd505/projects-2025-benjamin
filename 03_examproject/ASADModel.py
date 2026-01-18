from pyexpat import model
import numpy as np
import matplotlib.pyplot as plt

class ASADModelClass:

    def __init__(self, par=None):
    
        par = dict(
            ybar    = 1.0,
            pi_star = 0.02,
            b       = 0.6,
            a1      = 1.5,
            a2      = 0.10,
            gamma   = 4.0,
            phi     = 0.6
        )

        self.par = par.copy()

    def _alpha_z(self, v):

        p = self.par

        alpha = p['b'] * p['a1'] / (1.0 + p['b'] * p['a2'])
        z = v / (1.0 + p['b'] * p['a2'])

        return alpha, z

    def AD_curve(self, y, v):

        p = self.par
        alpha, z = self._alpha_z(v)
        inv_alpha = 1.0 / alpha
        return p['pi_star'] - inv_alpha * ((y - p['ybar']) - z)

    def SRAS_curve(self, y, pi_e):
        
        p = self.par
        return pi_e + p['gamma'] * (y - p['ybar'])

    def equilibrium(self, pi_e, v):
        p = self.par
        alpha, z = self._alpha_z(v)
        inv_alpha = 1.0 / alpha

        # solve AD = SRAS in terms of x = (y - ybar)
        denom = p['gamma'] + inv_alpha
        x = (p['pi_star'] - pi_e + inv_alpha * z) / denom

        y_star = p['ybar'] + x
        pi_star_t = pi_e + p['gamma'] * x

        return y_star, pi_star_t

    # compute sd(y_gap), sd(pi), corr(y_gap, pi)

    def simulate(self, T=5, v0=0.1, pars=None, pad=0.12, ngrid=600):

        model = ASADModelClass(pars)
        p = model.par
        T = int(T)

        v = np.zeros(T) # demand shock
        v[0] = v0

        y_star = np.empty(T)
        pi_star_t = np.empty(T)
        pi_e = np.empty(T)

        pi_e[0] = p["pi_star"]

        y_star[0], pi_star_t[0] = model.equilibrium(pi_e[0], v[0]) # initial equilibrium

        for t in range(1, T):
            pi_e[t] = p["phi"] * pi_e[t-1] + (1.0 - p["phi"]) * pi_star_t[t-1] # adaptive expectations
            y_star[t], pi_star_t[t] = model.equilibrium(pi_e[t], v[t])

        y_grid = np.linspace(p["ybar"] - pad, p["ybar"] + pad, ngrid) # output grid for plotting

        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
        axes = axes.ravel()

        for t in range(T): # plot AD and SRAS curves for each period
            ax = axes[t]
            pi_ad   = self.AD_curve(y_grid, v=v[t])
            pi_sras = self.SRAS_curve(y_grid, pi_e=pi_e[t])

            ax.plot(y_grid, pi_ad, linewidth=2, label="AD")
            ax.plot(y_grid, pi_sras, linewidth=2, label="SRAS")

            ax.scatter([y_star[t]], [pi_star_t[t]], zorder=5)
            ax.annotate(r"$(y_t^\star,\pi_t^\star)$",
                        xy=(y_star[t], pi_star_t[t]),
                        xytext=(y_star[t] + 0.01, pi_star_t[t] + 0.01),
                        arrowprops=dict(arrowstyle="->", lw=1.0))

            ax.axvline(p["ybar"], linestyle=":", linewidth=1)
            ax.axhline(p["pi_star"], linestyle=":", linewidth=1)

            ax.set_title(f"t={t}: v_t={v[t]:.2f},  πᵉ_t={pi_e[t]:.4f}")
            ax.grid(True, alpha=0.25)

        axes[-1].axis("off")

        handles, labels = axes[0].get_legend_handles_labels() 
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

        fig.text(0.5, 0.04, r"Output, $y_t$", ha="center")
        fig.text(0.04, 0.5, r"Inflation, $\pi_t$", va="center", rotation="vertical")
        plt.tight_layout(rect=[0.05, 0.06, 1, 0.92])
        plt.show()
        
        return {"y_star": y_star, "pi_star": pi_star_t, "v": v, "pi_e": pi_e, "par": p}
    


    def simulate_ar1(self, model, T=500, rho=0.8, sigma_eps=0.01, seed=123, eps=None): # Simulate AS-AD model with AR(1) demand shocks

        p = model.par
        T = int(T)
    
        if eps is None: # generate shocks if not provided
            rng = np.random.default_rng(seed)
            eps = rng.normal(loc=0.0, scale=sigma_eps, size=T)
        else:
            eps = np.asarray(eps)
            assert len(eps) == T

        v = np.empty(T) # demand shock
        v_prev = 0.0
        for t in range(T):
            v[t] = rho * v_prev + eps[t]
            v_prev = v[t]

        y_star = np.empty(T) # equilibrium output gap
        pi_star = np.empty(T)   # equilibrium inflation
        pi_e = np.empty(T)  # expected inflation

        pi_e[0] = p["pi_star"]

        y_star[0], pi_star[0] = model.equilibrium(pi_e[0], v[0]) # initial equilibrium

        for t in range(1, T): # simulate over time
            pi_e[t] = p["phi"] * pi_e[t-1] + (1.0 - p["phi"]) * pi_star[t-1]
            y_star[t], pi_star[t] = model.equilibrium(pi_e[t], v[t])

        return {"y_star": y_star, "pi_star": pi_star, "v": v, "eps": eps, "pi_e": pi_e}


    def moments(self, y, pi): # Compute standard deviations and correlation
        sd_y = np.std(y, ddof=1)
        sd_pi = np.std(pi, ddof=1)
        corr = np.corrcoef(y, pi)[0, 1]
        return sd_y, sd_pi, corr

    def plot_series(self, sim, title=""): # Plot time series of output gap, inflation, and demand shock
        T = len(sim["y_star"])
        t = np.arange(T)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(t, sim["y_star"]) # plot output gap
        axes[0].set_ylabel(r"$y_t^\star$")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(t, sim["pi_star"]) # plot inflation
        axes[1].set_ylabel(r"$\pi_t^\star$")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(t, sim["v"]) # plot demand shock
        axes[2].set_ylabel(r"$v_t$")
        axes[2].set_xlabel("t")
        axes[2].grid(True, alpha=0.25)

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
