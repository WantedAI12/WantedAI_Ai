"""Positive two-reservoir transfer with analytic sink integrals (per material)."""
import numpy as np


def bidirectional_step(film, air, evaporation, uptake, return_rate, ventilation, dt):
    """Integrate film <-> air, film -> skin and air -> exhaust for fixed rates.

Rates are min^-1. The spectral form avoids unstable hyperbolic exponentials
and cancellation when the two eigenvalues have very different magnitudes.
"""
    a, d = evaporation + uptake, return_rate + ventilation
    delta = a - d
    gap = np.hypot(delta, 2 * np.sqrt(evaporation * return_rate))
    fast = (a + d + gap) / 2
    determinant = evaporation * ventilation + uptake * return_rate + uptake * ventilation
    slow = np.divide(determinant, fast, out=np.zeros_like(fast), where=fast > 0)
    # Diagonal elements of A + fast*I, computed without gap - abs(delta).
    large = (gap + np.abs(delta)) / 2
    small = np.divide(evaporation * return_rate, large, out=np.zeros_like(large), where=large > 0)
    diagonal_film = np.where(delta >= 0, small, large)
    diagonal_air = np.where(delta >= 0, large, small)
    vector_film = diagonal_film * film + return_rate * air
    vector_air = evaporation * film + diagonal_air * air
    exp_fast = np.exp(-fast * dt)
    divided_exp = np.exp(-slow * dt) * np.divide(
        -np.expm1(-gap * dt), gap, out=np.full_like(gap, dt), where=gap > 0)
    new_film = exp_fast * film + divided_exp * vector_film
    new_air = exp_fast * air + divided_exp * vector_air

    phi_fast = np.divide(-np.expm1(-fast * dt), fast, out=np.full_like(fast, dt), where=fast > 0)
    phi_slow = np.divide(-np.expm1(-slow * dt), slow, out=np.full_like(slow, dt), where=slow > 0)
    divided_phi = np.divide(phi_slow - phi_fast, gap, out=np.zeros_like(gap), where=gap > 0)
    close = gap <= 1e-7 * np.maximum(fast, 1 / dt)
    if np.any(close):
        rate = (fast[close] + slow[close]) / 2
        x = rate * dt
        integral = np.empty_like(x)
        small_x = x < .001
        z = x[small_x]
        integral[small_x] = dt**2 * (.5 - z / 3 + z*z / 8 - z*z*z / 30)
        z = x[~small_x]
        integral[~small_x] = (1 - (1 + z) * np.exp(-z)) / rate[~small_x]**2
        divided_phi[close] = integral
    integrated_film = phi_fast * film + divided_phi * vector_film
    integrated_air = phi_fast * air + divided_phi * vector_air
    return new_film, new_air, uptake * integrated_film, ventilation * integrated_air
