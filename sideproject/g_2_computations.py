# ------------------------------------------------------------
# g_2_computations.py  – 4‑loop QCD running with JAX
# ------------------------------------------------------------
"""
Run this file directly   →   python g_2_computations.py

This script calculates the Koide parameters for up-type and down-type
quarks by evolving their running masses to the Grand Unification (GUT) scale
using 4-loop renormalization group equations (RGEs).

The Koide relationship K = Σm / (Σ√m)² ≈ 2/3 is believed to hold
for the running masses at this high-energy scale.
"""
import math, functools
import jax
import jax.numpy as jnp
from functools import partial

zeta3 = 1.2020569031595942854
# -------------------- β and γ coefficients ----------------------------
@jax.jit
def beta_coeffs(nf):
    b0 = 11 - 2*nf/3
    b1 = 102 - 38*nf/3
    b2 = 2857/2 - 5033*nf/18 + 325*nf**2/54
    b3 = (149753/6 + 3564*zeta3
          - (1078361/162 + 6508/27*zeta3)*nf
          + (50065/162 + 6472/81*zeta3)*nf**2
          + 1093/729*nf**3)
    return jnp.array([b0, b1, b2, b3])

@jax.jit
def gamma_coeffs(nf):
    g0 = 4
    g1 = 202/3 - 20*nf/9
    g2 = 1249 - 2216*nf/27 - 140*nf**2/81
    g3 = (4603055/162 + 135680*zeta3/27
          - (91723/27 + 34192*zeta3/9)*nf
          + (5242/243 + 800*zeta3/9)*nf**2
          + 332*nf**3/243)
    return jnp.array([g0, g1, g2, g3])

@partial(jax.jit, static_argnames=['loops'])
def beta_alpha(a, nf, loops=4):
    b = beta_coeffs(nf)
    x = a/(4*jnp.pi)
    powers = jnp.array([x**k for k in range(loops)])
    return -2*a*a*jnp.dot(b[:loops], powers)/(4*jnp.pi)

@partial(jax.jit, static_argnames=['loops'])
def gamma_m(a, nf, loops=4):
    g = gamma_coeffs(nf)
    x = a/(4*jnp.pi)
    powers = jnp.array([x**(k+1) for k in range(loops)])
    return -jnp.dot(g[:loops], powers)

# ------------------ RK4 step & segment integrate ----------------------
@partial(jax.jit, static_argnames=['nf','loops'])
def rk4_step(alpha, mass, dln, nf, loops):
    k1a = beta_alpha(alpha, nf, loops);      k1m = mass*gamma_m(alpha, nf, loops)
    k2a = beta_alpha(alpha+0.5*dln*k1a, nf, loops)
    k2m = (mass+0.5*dln*k1m)*gamma_m(alpha+0.5*dln*k1a, nf, loops)
    k3a = beta_alpha(alpha+0.5*dln*k2a, nf, loops)
    k3m = (mass+0.5*dln*k2m)*gamma_m(alpha+0.5*dln*k2a, nf, loops)
    k4a = beta_alpha(alpha+dln*k3a, nf, loops)
    k4m = (mass+dln*k3m)*gamma_m(alpha+dln*k3a, nf, loops)
    alpha += (dln/6)*(k1a+2*k2a+2*k3a+k4a)
    mass  += (dln/6)*(k1m+2*k2m+2*k3m+k4m)
    return alpha, mass

@partial(jax.jit, static_argnames=['nf','loops','n_steps'])
def evolve_segment(mu0, mu1, alpha0, m0, nf, loops=4, n_steps=8192):
    log_mu = jnp.linspace(jnp.log(mu0), jnp.log(mu1), n_steps)
    dln = (log_mu[1]-log_mu[0]).astype(alpha0.dtype)
    def body(carry, _):
        a,m = carry; return rk4_step(a,m,dln,nf,loops), None
    (af,mf), _ = jax.lax.scan(body, (alpha0,m0), None, length=n_steps-1)
    return af, mf

# ------------------ RGE running functions -----------------
alpha_s_MZ = 0.1181;  MZ = 91.1876
THRS = [(1.27,4), (4.18,5), (172.76,6)]  # (μ_thresh, n_f_above)

def run_to_high(mu0, alpha0, m0, mu1, loops=4):
    nf = 3 if mu0 < 1.27 else 4 if mu0 < 4.18 else 5 if mu0 < 172.76 else 6
    alpha, mass, start = alpha0, m0, mu0
    
    all_thresholds = sorted([(t, nf_a) for t, nf_a in THRS if mu0 < t < mu1] + [(mu1, None)])

    for thr, nf_above in all_thresholds:
        if start < thr:
            alpha, mass = evolve_segment(start, thr, alpha, mass, nf, loops=loops)
            start = thr
        if nf_above is not None:
            nf = nf_above
            
    return alpha, mass

def alphas_down(mu_target, loops=4):
    alpha, start = alpha_s_MZ, MZ
    for thr, nf_above in reversed(THRS):
        if mu_target < thr < start:
            alpha, _ = evolve_segment(start, thr, alpha, 0.0, nf_above, loops=loops)
            start = thr
    nf_final = 3 if mu_target < 1.27 else 4 if mu_target < 4.18 else 5
    alpha, _ = evolve_segment(start, mu_target, alpha, 0.0, nf_final, loops=loops)
    return alpha

# --------------- Koide helper ----------------------------------------
def koide(m1, m2, m3):
    """Computes the Koide formula for three masses."""
    return (m1 + m2 + m3) / (math.sqrt(m1) + math.sqrt(m2) + math.sqrt(m3))**2

# --------------------- Main execution ------------------------------------
if __name__ == "__main__":
    # Use running masses at their respective scales. m_t(m_t) is ~163 GeV.
    pdg = {
        'u': (0.00216, 2.0),  'd': (0.00467, 2.0),  's': (0.093, 2.0),
        'c': (1.27,   1.27), 'b': (4.18,   4.18), 't': (163.0, 163.0),
    }

    masses_at_gut = {}
    for q, (m0, mu0) in pdg.items():
        a0 = alphas_down(mu0)
        # Evolve to the GUT scale
        _, m_hi = run_to_high(mu0, a0, m0, mu1=1e16)
        masses_at_gut[q] = m_hi

    # The Koide relation is believed to hold for the running masses at the GUT scale.
    koide_u = koide(masses_at_gut['u'], masses_at_gut['c'], masses_at_gut['t'])
    koide_d = koide(masses_at_gut['d'], masses_at_gut['s'], masses_at_gut['b'])

    print(f"Koide_u  = {koide_u}")
    print(f"Koide_d  = {koide_d}")
