"""Exomoon detectability pipeline v31.

Runs moon injection--recovery, no-moon false-positive tests, plotting,
reporting, and optional nested-sampling validation.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from math import pi, log
from typing import Optional, Dict, Tuple, Any, Sequence, Mapping, List
import json
import time as _time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_distribution
G = 6.6743e-11
MSUN = 1.98847e+30
RSUN = 695700000.0
MJUP = 1.89813e+27
RJUP = 71492000.0
MEARTH = 5.9722e+24
REARTH = 6371000.0
AU = 149597870700.0
DAY = 86400.0
WHITE_DWARF_TARGETS = {'helix', 'wd1856'}

@dataclass
class Star:
    name: str
    mass_msun: float
    radius_rsun: float
    limb_darkening_u1: float = 0.4
    limb_darkening_u2: float = 0.25

@dataclass
class Planet:
    name: str
    period_days: float
    radius_rj: float
    mass_mj: float
    semi_major_au: Optional[float] = None
    t0_days: float = 0.0
    impact_b: float = 0.0
    ecc: float = 0.0
    omega_deg: float = 90.0

@dataclass
class MoonGrid:
    radius_re_min: float
    radius_re_max: float
    a_rp_min: float
    hill_fraction_max: float
    mutual_inc_deg_max: float = 5.0
    ecc_min: float = 0.0
    ecc_max: float = 0.1
    force_coplanar: bool = False

@dataclass
class Noise:
    instrument: str
    cadence_min: float
    white_ppm: float
    red_ppm: float = 0.0
    red_timescale_hr: float = 6.0
    duty_cycle: float = 1.0

def telescope_noise_preset(instrument: str, target_mag: float, cadence_min: Optional[float]=None, red_ppm: Optional[float]=None, red_timescale_hr: Optional[float]=None, duty_cycle: Optional[float]=None) -> Noise:
    """Return a magnitude-scaled Kepler or TESS noise preset."""
    inst = str(instrument).strip().lower()
    mag = float(target_mag)
    if inst == 'kepler':
        ref_mag = 12.0
        ref_precision_ppm = 20.0
        ref_duration_hr = 6.5
        systematic_floor_ppm = 0.0
        cadence = 29.4 if cadence_min is None else float(cadence_min)
        red = 50.0 if red_ppm is None else float(red_ppm)
        tau = 6.0 if red_timescale_hr is None else float(red_timescale_hr)
        duty = 0.92 if duty_cycle is None else float(duty_cycle)
    elif inst == 'tess':
        ref_mag = 10.0
        ref_precision_ppm = 230.0
        ref_duration_hr = 1.0
        systematic_floor_ppm = 60.0
        cadence = 2.0 if cadence_min is None else float(cadence_min)
        red = 250.0 if red_ppm is None else float(red_ppm)
        tau = 3.0 if red_timescale_hr is None else float(red_timescale_hr)
        duty = 0.9 if duty_cycle is None else float(duty_cycle)
    else:
        raise ValueError("instrument must be 'kepler' or 'tess'")
    white_ref = max(ref_precision_ppm ** 2 - systematic_floor_ppm ** 2, 0.0) ** 0.5
    white_mag = white_ref * 10.0 ** (0.2 * (mag - ref_mag))
    precision_mag = (white_mag ** 2 + systematic_floor_ppm ** 2) ** 0.5
    cadence_hr = max(cadence / 60.0, 1e-12)
    white_per_cadence = precision_mag * (ref_duration_hr / cadence_hr) ** 0.5
    return Noise(instrument=inst, cadence_min=float(cadence), white_ppm=float(white_per_cadence), red_ppm=float(red), red_timescale_hr=float(tau), duty_cycle=float(duty))

@dataclass
class Config:
    target: str
    star: Star
    planet: Planet
    moon_grid: MoonGrid
    noise: Noise
    n_transits: int
    local_window_days: Optional[float] = None
    rng_seed: int = 1
    snr_threshold: float = 7.0
    logk_threshold: float = log(10.0)

@dataclass
class NestedConfig:
    """UltraNest settings for the optional v31 nested-sampling validation.

    The nested run is now a diagnostic Bayes-factor validation, not an extra
    empirical false-positive calibration.  The core grid/no-moon grid remains
    the main false-positive control.  A RegionSliceSampler is enabled by
    default because the M1 planet+moon posterior can be inefficient for
    UltraNest's default region drawing.
    """
    min_live_points: int = 80
    dlogz: float = 0.5
    max_ncalls: Optional[int] = None
    resume: str = 'overwrite'
    use_stepsampler: bool = True
    stepsampler_nsteps: Optional[int] = 100
    stepsampler_adaptive_nsteps: Any = 'move-distance'
    stepsampler_region_filter: bool = True
    frac_remain: Optional[float] = 0.5

def parameter_input_dataframes(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Return configuration values as notebook-ready tables."""
    return {'star': pd.DataFrame([asdict(cfg.star)]), 'planet': pd.DataFrame([asdict(cfg.planet)]), 'moon_grid': pd.DataFrame([asdict(cfg.moon_grid)]), 'noise': pd.DataFrame([asdict(cfg.noise)]), 'experiment': pd.DataFrame([{k: v for k, v in asdict(cfg).items() if k not in {'star', 'planet', 'moon_grid', 'noise'}}])}

def choose_truth_backend(cfg: Config) -> str:
    """Use the fast model for configured white-dwarf targets and Pandora otherwise."""
    return 'fast' if cfg.target in WHITE_DWARF_TARGETS else 'pandora'

def semi_major_axis_from_period(period_days: float, m1_kg: float, m2_kg: float=0.0) -> float:
    """Calculate semi-major axis from period and masses."""
    return (G * (m1_kg + m2_kg) * (period_days * DAY) ** 2 / (4.0 * pi ** 2)) ** (1.0 / 3.0)

def period_from_semi_major_axis(a_m: float, central_mass_kg: float) -> float:
    """Calculate orbital period from semi-major axis and central mass."""
    return 2.0 * pi * np.sqrt(a_m ** 3 / (G * central_mass_kg)) / DAY

def hill_radius(a_planet_m: float, m_planet_kg: float, m_star_kg: float) -> float:
    """Calculate the planetary Hill radius."""
    return a_planet_m * (m_planet_kg / (3.0 * m_star_kg)) ** (1.0 / 3.0)

def orbital_speed(a_m: float, period_days: float) -> float:
    """Calculate circular orbital speed."""
    return 2.0 * pi * a_m / (period_days * DAY)

def planet_semi_major_m(cfg: Config) -> float:
    """Return the planet semi-major axis in metres."""
    if cfg.planet.semi_major_au is not None:
        return cfg.planet.semi_major_au * AU
    return semi_major_axis_from_period(cfg.planet.period_days, cfg.star.mass_msun * MSUN, cfg.planet.mass_mj * MJUP)

def transit_duration_days(star_r_m: float, body_r_m: float, a_m: float, period_days: float, b: float) -> float:
    """Approximate the transit duration from chord length and orbital speed."""
    chord = np.sqrt(max((1.0 + body_r_m / star_r_m) ** 2 - b ** 2, 0.0)) * star_r_m
    return 2.0 * chord / orbital_speed(a_m, period_days) / DAY

def trapezoid(t: np.ndarray, t0: float, duration: float, ingress_fraction: float=0.12) -> np.ndarray:
    """Return a unit-depth trapezoidal transit profile."""
    half = 0.5 * duration
    ingress = max(duration * ingress_fraction, 1e-12)
    x = np.abs(t - t0)
    y = np.zeros_like(t, dtype=float)
    flat = x <= half - ingress
    ramp = (x > half - ingress) & (x <= half)
    y[flat] = 1.0
    y[ramp] = (half - x[ramp]) / ingress
    return y

def separation_bounds_rp(cfg: Config) -> Tuple[float, float]:
    """Return the requested minimum and Hill-limited maximum moon separations."""
    a_p = planet_semi_major_m(cfg)
    m_star = cfg.star.mass_msun * MSUN
    m_planet = cfg.planet.mass_mj * MJUP
    r_planet = cfg.planet.radius_rj * RJUP
    if a_p <= 0 or m_star <= 0 or m_planet <= 0 or (r_planet <= 0):
        raise ValueError('Invalid Hill-bound inputs: planet semi-major axis, stellar mass, planet mass, and planet radius must all be positive.')
    hill_rp = hill_radius(a_p, m_planet, m_star) / r_planet
    upper = cfg.moon_grid.hill_fraction_max * hill_rp
    lower = float(cfg.moon_grid.a_rp_min)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f'Invalid Hill-bound inputs produced non-finite separation bounds: a_rp_min={lower}, upper={upper}.')
    if lower <= 0 or upper <= 0:
        raise ValueError(f'Invalid moon-separation bounds: a_rp_min={lower:.6g} R_p and Hill-limited upper={upper:.6g} R_p must both be positive.')
    if upper <= lower:
        raise ValueError(f'No physically allowed moon-separation range for {cfg.planet.name}: a_rp_min={lower:.6g} R_p exceeds or equals the Hill-limited upper bound {upper:.6g} R_p. Computed R_H/R_p={hill_rp:.6g}; hill_fraction_max={cfg.moon_grid.hill_fraction_max:.6g}. Reduce a_rp_min, increase hill_fraction_max within a justified stability limit, or revise the adopted star/planet parameters.')
    return (float(lower), float(upper))

def make_time_grid(cfg: Config) -> np.ndarray:
    """Build observing windows around each planetary transit."""
    cadence_days = cfg.noise.cadence_min / (24.0 * 60.0)
    a_p = planet_semi_major_m(cfg)
    star_r = cfg.star.radius_rsun * RSUN
    planet_r = cfg.planet.radius_rj * RJUP
    dur = transit_duration_days(star_r, planet_r, a_p, cfg.planet.period_days, cfg.planet.impact_b)
    rh = hill_radius(a_p, cfg.planet.mass_mj * MJUP, cfg.star.mass_msun * MSUN)
    max_offset = cfg.moon_grid.hill_fraction_max * rh / orbital_speed(a_p, cfg.planet.period_days) / DAY
    window = cfg.local_window_days if cfg.local_window_days is not None else max(8.0 * dur + 2.0 * max_offset, 0.35)
    chunks = []
    for i in range(int(cfg.n_transits)):
        tc = cfg.planet.t0_days + i * cfg.planet.period_days
        chunks.append(np.arange(tc - window / 2, tc + window / 2 + cadence_days / 2, cadence_days))
    time = np.sort(np.concatenate(chunks))
    if cfg.noise.duty_cycle < 1.0:
        rng = np.random.default_rng(cfg.rng_seed + 17)
        time = time[rng.random(time.size) < cfg.noise.duty_cycle]
    return time

def _moon_to_pandora(cfg: Config, moon: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Convert internal moon parameters to Pandora inputs."""
    if moon is None:
        return dict(r_moon=1e-10, per_moon=10.0, tau_moon=0.0, Omega_moon=0.0, i_moon=0.0, ecc_moon=0.0, w_moon=0.0, M_moon=1e-12 * MJUP)
    radius_re = float(moon['radius_re'])
    mass_kg = float(moon.get('mass_mearth', max(radius_re, 1e-06) ** 3)) * MEARTH
    return dict(r_moon=radius_re * REARTH / (cfg.star.radius_rsun * RSUN), per_moon=float(moon['period_days']), tau_moon=float(moon.get('phase0', 0.0) / (2.0 * pi) % 1.0), Omega_moon=float(moon.get('Omega_moon', 0.0)), i_moon=float(moon.get('mutual_inc_deg', 0.0)), ecc_moon=float(moon.get('ecc', 0.0)), w_moon=float(moon.get('w_moon', 0.0)), M_moon=mass_kg)

def pandora_flux(time: np.ndarray, cfg: Config, moon: Optional[Dict[str, float]]=None) -> np.ndarray:
    """Generate a Pandora planet--moon light curve."""
    try:
        from pandoramoon.pandora import pandora
    except Exception as exc:
        raise ImportError("Pandora requested but pandoramoon is not importable. Install pandoramoon, or set truth_backend='fast'.") from exc
    star_r = cfg.star.radius_rsun * RSUN
    planet_r = cfg.planet.radius_rj * RJUP
    a_rs = planet_semi_major_m(cfg) / star_r
    mv = _moon_to_pandora(cfg, moon)
    _, _, flux_total, *_ = pandora(cfg.star.limb_darkening_u1, cfg.star.limb_darkening_u2, star_r, cfg.planet.period_days, a_rs, planet_r / star_r, cfg.planet.impact_b, cfg.planet.omega_deg, cfg.planet.ecc, cfg.planet.t0_days, 0.0, cfg.planet.mass_mj * MJUP, mv['r_moon'], mv['per_moon'], mv['tau_moon'], mv['Omega_moon'], mv['i_moon'], mv['ecc_moon'], mv['w_moon'], mv['M_moon'], cfg.planet.period_days, 1, 0.01, 1.1, 25, np.asarray(time, dtype=float), None)
    return np.asarray(flux_total, dtype=float)

def fast_planet_flux(time: np.ndarray, cfg: Config) -> np.ndarray:
    """Generate the simplified planet-only light curve."""
    star_r = cfg.star.radius_rsun * RSUN
    planet_r = cfg.planet.radius_rj * RJUP
    a_p = planet_semi_major_m(cfg)
    dur = transit_duration_days(star_r, planet_r, a_p, cfg.planet.period_days, cfg.planet.impact_b)
    depth = min((planet_r / star_r) ** 2, 1.0)
    flux = np.ones_like(time, dtype=float)
    for i in range(int(cfg.n_transits)):
        tc = cfg.planet.t0_days + i * cfg.planet.period_days
        flux -= depth * trapezoid(time, tc, dur, ingress_fraction=0.1)
    return np.clip(flux, 0.0, 1.5)

def fast_moon_delta(time: np.ndarray, cfg: Config, moon: Dict[str, float]) -> Tuple[np.ndarray, int]:
    """Generate the simplified moon signal and count visible moon transits."""
    star_r = cfg.star.radius_rsun * RSUN
    moon_r = moon['radius_re'] * REARTH
    a_p = planet_semi_major_m(cfg)
    v_p = orbital_speed(a_p, cfg.planet.period_days)
    dur = transit_duration_days(star_r, moon_r, a_p, cfg.planet.period_days, cfg.planet.impact_b)
    depth = min((moon_r / star_r) ** 2, 1.0)
    inc = np.deg2rad(moon.get('mutual_inc_deg', 0.0))
    delta = np.zeros_like(time, dtype=float)
    n_seen = 0
    for i in range(int(cfg.n_transits)):
        tc_p = cfg.planet.t0_days + i * cfg.planet.period_days
        phase = moon['phase0'] + 2.0 * pi * ((tc_p - cfg.planet.t0_days) / moon['period_days'])
        x = moon['a_pm_m'] * np.cos(phase)
        y = cfg.planet.impact_b * star_r + moon['a_pm_m'] * np.sin(phase) * np.sin(inc)
        if abs(y) <= star_r + moon_r:
            tc_m = tc_p + x / v_p / DAY
            delta += depth * trapezoid(time, tc_m, dur, ingress_fraction=0.12)
            n_seen += 1
    return (delta, n_seen)

def model_flux(time: np.ndarray, cfg: Config, moon: Optional[Dict[str, float]]=None, backend: str='fast') -> np.ndarray:
    """Generate a light curve with the selected forward model."""
    backend = backend.lower()
    if backend == 'pandora':
        return pandora_flux(time, cfg, moon)
    if backend != 'fast':
        raise ValueError("backend must be 'fast' or 'pandora'")
    flux = fast_planet_flux(time, cfg)
    if moon is not None:
        delta, _ = fast_moon_delta(time, cfg, moon)
        flux = flux - delta
    return np.clip(flux, 0.0, 1.5)

def red_noise_ar1(time: np.ndarray, red_ppm: float, tau_hr: float, rng: np.random.Generator) -> np.ndarray:
    """Generate stationary AR(1) red noise."""
    sigma = red_ppm * 1e-06
    if sigma <= 0 or time.size == 0:
        return np.zeros_like(time)
    tau_days = max(tau_hr / 24.0, 1e-12)
    y = np.zeros_like(time, dtype=float)
    y[0] = rng.normal(0.0, sigma)
    for i in range(1, time.size):
        rho = np.exp(-max(time[i] - time[i - 1], 0.0) / tau_days)
        y[i] = rho * y[i - 1] + sigma * np.sqrt(max(1.0 - rho ** 2, 0.0)) * rng.normal()
    return y

def add_noise(flux: np.ndarray, time: np.ndarray, noise: Noise, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Add white and red noise to a model light curve."""
    white = noise.white_ppm * 1e-06
    y = flux + rng.normal(0.0, white, size=flux.size)
    if noise.red_ppm > 0:
        y = y + red_noise_ar1(time, noise.red_ppm, noise.red_timescale_hr, rng)
    sigma = np.full_like(flux, np.sqrt(noise.white_ppm ** 2 + noise.red_ppm ** 2) * 1e-06)
    return (y, sigma)

def make_moon(cfg: Config, rng: np.random.Generator, radius_re: float, a_rp: float, random_phase: bool=True) -> Dict[str, float]:
    """Create one moon realization from the configured grid."""
    planet_r = cfg.planet.radius_rj * RJUP
    a_pm = a_rp * planet_r
    return {'radius_re': float(radius_re), 'a_rp': float(a_rp), 'a_pm_m': float(a_pm), 'period_days': float(period_from_semi_major_axis(a_pm, cfg.planet.mass_mj * MJUP)), 'phase0': float(rng.uniform(0.0, 2.0 * pi) if random_phase else pi), 'mutual_inc_deg': 0.0 if cfg.moon_grid.force_coplanar else float(rng.uniform(0.0, cfg.moon_grid.mutual_inc_deg_max)), 'ecc': float(rng.uniform(cfg.moon_grid.ecc_min, cfg.moon_grid.ecc_max))}

def simulate_case(cfg: Config, rng: np.random.Generator, moon: Optional[Dict[str, float]], truth_backend: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
    """Simulate one noisy injected-moon or no-moon case."""
    time = make_time_grid(cfg)
    moon_out = None if moon is None else dict(moon)
    if moon_out is not None and truth_backend == 'fast':
        _, n_seen = fast_moon_delta(time, cfg, moon_out)
        moon_out['n_moon_transits'] = int(n_seen)
    elif moon_out is not None:
        moon_out['n_moon_transits'] = int(cfg.n_transits)
    clean = model_flux(time, cfg, moon=moon_out, backend=truth_backend)
    flux, sigma = add_noise(clean, time, cfg.noise, rng)
    return (time, flux, sigma, moon_out)

def _box_improvement(resid: np.ndarray, sigma: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    """Fit a positive transit depth and return its chi-square gain and SNR."""
    if mask.sum() < 2:
        return (0.0, 0.0, 0.0)
    w = 1.0 / sigma ** 2
    depth = -np.sum(w[mask] * resid[mask]) / np.sum(w[mask])
    if depth <= 0:
        return (0.0, 0.0, 0.0)
    chi0 = np.sum((resid / sigma) ** 2)
    chi1 = np.sum(((resid + depth * mask.astype(float)) / sigma) ** 2)
    snr = depth * np.sqrt(np.sum(w[mask]))
    return (float(max(chi0 - chi1, 0.0)), float(depth), float(snr))

def recover_moon_fast(time: np.ndarray, flux: np.ndarray, sigma: np.ndarray, cfg: Config, planet_baseline_backend: str='fast') -> Dict[str, Any]:
    """Search separation, phase, and duration for the strongest moon-like residual."""
    planet_baseline_backend = str(planet_baseline_backend).lower()
    if planet_baseline_backend == 'fast':
        planet_baseline = fast_planet_flux(time, cfg)
    elif planet_baseline_backend == 'pandora':
        planet_baseline = model_flux(time, cfg, moon=None, backend='pandora')
    else:
        raise ValueError("planet_baseline_backend must be 'fast' or 'pandora'")
    resid = flux - planet_baseline
    a0, a1 = separation_bounds_rp(cfg)
    a_grid = np.linspace(a0, a1, 20)
    phase_grid = np.linspace(0.0, 2.0 * pi, 24, endpoint=False)
    r_mid = 0.5 * (cfg.moon_grid.radius_re_min + cfg.moon_grid.radius_re_max) * REARTH
    dur_mid = transit_duration_days(cfg.star.radius_rsun * RSUN, r_mid, planet_semi_major_m(cfg), cfg.planet.period_days, cfg.planet.impact_b)
    duration_grid = np.array([0.6, 1.0, 1.7]) * dur_mid
    planet_r = cfg.planet.radius_rj * RJUP
    v_p = orbital_speed(planet_semi_major_m(cfg), cfg.planet.period_days)
    best = dict(delta_chi2=0.0, depth=0.0, snr=0.0, a_rp=np.nan, phase=np.nan, duration_days=np.nan)
    for a_rp in a_grid:
        a_m = a_rp * planet_r
        p_m = period_from_semi_major_axis(a_m, cfg.planet.mass_mj * MJUP)
        for ph0 in phase_grid:
            centers = []
            for i in range(int(cfg.n_transits)):
                tc_p = cfg.planet.t0_days + i * cfg.planet.period_days
                ph = ph0 + 2.0 * pi * ((tc_p - cfg.planet.t0_days) / p_m)
                centers.append(tc_p + a_m * np.cos(ph) / v_p / DAY)
            for dur in duration_grid:
                mask = np.zeros_like(time, dtype=bool)
                for c in centers:
                    mask |= np.abs(time - c) <= 0.5 * dur
                dchi2, depth, snr = _box_improvement(resid, sigma, mask)
                if dchi2 > best['delta_chi2']:
                    best.update(delta_chi2=dchi2, depth=depth, snr=snr, a_rp=float(a_rp), phase=float(ph0), duration_days=float(dur))
    n_trials = len(a_grid) * len(phase_grid) * len(duration_grid)
    delta_bic = best['delta_chi2'] - 4.0 * np.log(max(time.size, 2))
    logk_proxy = 0.5 * delta_bic - np.log(n_trials)
    best['delta_bic_proxy'] = float(delta_bic)
    best['logk_proxy'] = float(logk_proxy)
    best['equiv_radius_re'] = float(np.sqrt(max(best['depth'], 0.0)) * cfg.star.radius_rsun * RSUN / REARTH)
    best['detected'] = bool(best['snr'] >= cfg.snr_threshold and logk_proxy >= cfg.logk_threshold)
    return best

def default_grids(cfg: Config, n_radius: int=6, n_separation: int=6, n_red: int=5, n_tau: int=5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct the moon and red-noise parameter grids."""
    radius_grid = np.linspace(cfg.moon_grid.radius_re_min, cfg.moon_grid.radius_re_max, int(n_radius))
    a0, a1 = separation_bounds_rp(cfg)
    sep_grid = np.linspace(a0, a1, int(n_separation))
    red_grid = np.linspace(0.0, max(cfg.noise.red_ppm * 2.0, 1.0), int(n_red))
    tau_grid = np.linspace(0.5, max(cfg.noise.red_timescale_hr * 2.0, 1.0), int(n_tau))
    return (radius_grid, sep_grid, red_grid, tau_grid)

def summarise_grid(trials: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Aggregate trial detections into grid-cell probabilities and uncertainties."""
    if trials.empty:
        return pd.DataFrame(columns=[x, y, 'n', 'k_detected', 'p_hat', 'p_uncertainty_width'])
    g = trials.groupby([x, y], dropna=False).agg(n=('detected', 'size'), k_detected=('detected', 'sum'), p_hat=('detected', 'mean')).reset_index()
    g['p_uncertainty_width'] = 2.0 * np.sqrt(g['p_hat'] * (1.0 - g['p_hat']) / g['n'].clip(lower=1))
    return g

def jeffreys_binomial_median(k, n) -> float:
    """Return the Jeffreys-posterior median for a binomial probability."""
    if n <= 0:
        return np.nan
    return float(beta_distribution.ppf(0.5, k + 0.5, n - k + 0.5))

def _trigger_counts(signal_trials: pd.DataFrame, null_trials: pd.DataFrame) -> Tuple[int, int, int, int, float, float]:
    """Return recovery and false-positive counts and raw rates."""
    n_signal = len(signal_trials)
    n_null = len(null_trials)
    k_signal = int(signal_trials['detected'].sum()) if n_signal else 0
    k_null = int(null_trials['detected'].sum()) if n_null else 0
    signal_rate = k_signal / n_signal if n_signal else np.nan
    null_rate = k_null / n_null if n_null else np.nan
    return (k_signal, n_signal, k_null, n_null, signal_rate, null_rate)

def final_detectability_metrics(moon_trials: pd.DataFrame, noise_trials: pd.DataFrame, priors: Sequence[float]=(0.01, 0.05, 0.1, 0.5)) -> pd.DataFrame:
    """Build the final TPR, FPR, and prior-weighted detectability table."""
    k_rec, n_moon, k_fp, n_null, tpr, fpr = _trigger_counts(moon_trials, noise_trials)
    tpr_jeffreys = jeffreys_binomial_median(k_rec, n_moon)
    fpr_jeffreys = jeffreys_binomial_median(k_fp, n_null)
    rows = [{'quantity': 'Recovery probability / TPR', 'meaning': 'Empirical P(trigger | moon injected)', 'value': tpr, 'count': f'{k_rec}/{n_moon}'}, {'quantity': 'False-positive probability / FPR', 'meaning': 'Empirical P(trigger | no moon)', 'value': fpr, 'count': f'{k_fp}/{n_null}'}]
    for prior in priors:
        denom = prior * tpr_jeffreys + (1.0 - prior) * fpr_jeffreys
        ppv = prior * tpr_jeffreys / denom if denom > 0 else np.nan
        rows.append({'quantity': f'Prior-weighted credibility, prior={prior:g}', 'meaning': 'P(moon | trigger), using Jeffreys posterior-median TPR and FPR', 'value': ppv, 'count': 'derived'})
    return pd.DataFrame(rows)

def _fmt_prob(x: float) -> str:
    """Format a probability as a decimal and percentage."""
    try:
        x = float(x)
    except Exception:
        return 'nan'
    if not np.isfinite(x):
        return 'nan'
    return f'{x:.4f} ({100.0 * x:.2f}%)'

def combined_detectability_report(moon_trials: pd.DataFrame, noise_trials: pd.DataFrame, moon_grid: Optional[pd.DataFrame]=None, priors: Sequence[float]=(0.01, 0.05, 0.1, 0.5)) -> str:
    """Build the final plain-text detectability report."""
    k_rec, n_moon, k_fp, n_null, recovery, false_positive = _trigger_counts(moon_trials, noise_trials)
    recovery_jeffreys = jeffreys_binomial_median(k_rec, n_moon)
    false_positive_jeffreys = jeffreys_binomial_median(k_fp, n_null)
    if moon_grid is not None and len(moon_grid) and ('p_hat' in moon_grid.columns):
        mean_cell = float(moon_grid['p_hat'].mean())
    else:
        mean_cell = recovery
    lines = []
    lines.append('Combined exomoon detectability statistic')
    lines.append('=======================================')
    lines.append('')
    lines.append(f'No-moon false positives: {k_fp} / {n_null} = {_fmt_prob(false_positive)}')
    lines.append(f'Injected moons recovered: {k_rec} / {n_moon} = {_fmt_prob(recovery)}')
    lines.append(f'Mean recovery across moon radius–separation grid cells = {_fmt_prob(mean_cell)}')
    lines.append('')
    lines.append('Jeffreys posterior-median point estimates used in D(pi):')
    lines.append(f'TPR_J = {_fmt_prob(recovery_jeffreys)}')
    lines.append(f'FPR_J = {_fmt_prob(false_positive_jeffreys)}')
    lines.append('No credible interval is reported.')
    lines.append('')
    lines.append('For an assumed moon occurrence prior pi, the final credibility statistic is:')
    lines.append('D(pi) = pi * TPR_J / [pi * TPR_J + (1 - pi) * FPR_J]')
    lines.append('')
    for prior in priors:
        prior = float(prior)
        denom_hat = prior * recovery_jeffreys + (1.0 - prior) * false_positive_jeffreys
        d_hat = prior * recovery_jeffreys / denom_hat if denom_hat > 0 else np.nan
        lines.append(f'pi={prior:g}: D = {_fmt_prob(d_hat)}')
    lines.append('')
    lines.append('Interpretation: the raw counts describe the observed grid performance, while the Jeffreys posterior medians provide finite binomial point estimates for D(pi), including when zero false positives are observed.')
    lines.append(f'The injected-moon grid recovered {100.0 * recovery:.1f}% of trials overall. The mean radius–separation cell recovery probability is {100.0 * mean_cell:.1f}%. The no-moon red-noise grid produced false positives in {100.0 * false_positive:.1f}% of trials.')
    return '\n'.join(lines)

def report_from_results(results: Mapping[str, Any]) -> str:
    """Build the final report from a results dictionary."""
    return combined_detectability_report(results['moon_trials'], results['noise_trials'], moon_grid=results.get('moon_grid'))

def run_grid_suite(cfg: Config, n_radius: int=6, n_separation: int=6, n_red: int=5, n_tau: int=5, n_per_cell: int=3, truth_backend: str='auto', snr_threshold: Optional[float]=None, logk_threshold: Optional[float]=None, progress: bool=True) -> Dict[str, Any]:
    """Run the injected-moon recovery and no-moon false-positive grids."""
    if snr_threshold is not None or logk_threshold is not None:
        cfg = replace(cfg, snr_threshold=cfg.snr_threshold if snr_threshold is None else float(snr_threshold), logk_threshold=cfg.logk_threshold if logk_threshold is None else float(logk_threshold))
    if truth_backend == 'auto':
        truth_backend = choose_truth_backend(cfg)
    truth_backend = str(truth_backend).lower()
    planet_baseline_backend = truth_backend
    radius_grid, sep_grid, red_grid, tau_grid = default_grids(cfg, n_radius, n_separation, n_red, n_tau)
    rng = np.random.default_rng(cfg.rng_seed)
    moon_rows: List[Dict[str, Any]] = []
    total = len(radius_grid) * len(sep_grid)
    c = 0
    for r in radius_grid:
        for a in sep_grid:
            c += 1
            if progress:
                print(f'moon grid {c}/{total}: Rm={r:.3g} Re, a={a:.3g} Rp')
            for rep in range(int(n_per_cell)):
                moon = make_moon(cfg, rng, r, a)
                t, f, s, inj = simulate_case(cfg, rng, moon, truth_backend)
                rec = recover_moon_fast(t, f, s, cfg, planet_baseline_backend=planet_baseline_backend)
                row = {'grid_radius_re': float(r), 'grid_a_rp': float(a), 'rep': rep, **rec}
                row.update({'inj_' + k: v for k, v in inj.items() if np.isscalar(v)})
                moon_rows.append(row)
    moon_trials = pd.DataFrame(moon_rows)
    moon_trials['detected'] = moon_trials['detected'].astype(bool)
    moon_grid = summarise_grid(moon_trials, 'grid_a_rp', 'grid_radius_re')
    noise_rows: List[Dict[str, Any]] = []
    total = len(red_grid) * len(tau_grid)
    c = 0
    for red in red_grid:
        for tau in tau_grid:
            c += 1
            if progress:
                print(f'noise grid {c}/{total}: red={red:.3g} ppm, tau={tau:.3g} hr')
            cfg_n = replace(cfg, noise=replace(cfg.noise, red_ppm=float(red), red_timescale_hr=float(tau)))
            for rep in range(int(n_per_cell)):
                t, f, s, _ = simulate_case(cfg_n, rng, None, truth_backend)
                rec = recover_moon_fast(t, f, s, cfg_n, planet_baseline_backend=planet_baseline_backend)
                noise_rows.append({'noise_red_ppm': float(red), 'noise_tau_hr': float(tau), 'rep': rep, **rec})
    noise_trials = pd.DataFrame(noise_rows)
    noise_trials['detected'] = noise_trials['detected'].astype(bool)
    noise_grid = summarise_grid(noise_trials, 'noise_red_ppm', 'noise_tau_hr')
    metrics = final_detectability_metrics(moon_trials, noise_trials)
    report = combined_detectability_report(moon_trials, noise_trials, moon_grid=moon_grid)
    return dict(cfg=cfg, truth_backend=truth_backend, recovery_planet_backend=planet_baseline_backend, moon_trials=moon_trials, moon_grid=moon_grid, noise_trials=noise_trials, noise_grid=noise_grid, metrics=metrics, report=report)

def _pivot(grid: pd.DataFrame, x: str, y: str, z: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a grid table to x, y, and z arrays."""
    p = grid.pivot(index=y, columns=x, values=z).sort_index().sort_index(axis=1)
    return (p.columns.to_numpy(float), p.index.to_numpy(float), p.to_numpy(float))

def _heatmap(grid: pd.DataFrame, x: str, y: str, z: str, title: str, xlabel: str, ylabel: str, cbar_label: str) -> Tuple[Any, Any]:
    """Draw the common heatmap format."""
    xs, ys, Z = _pivot(grid, x, y, z)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = 'RdBu_r'
    if Z.shape[0] >= 2 and Z.shape[1] >= 2:
        X, Y = np.meshgrid(xs, ys)
        levels = np.linspace(np.nanmin(Z), np.nanmax(Z) if np.nanmax(Z) > np.nanmin(Z) else np.nanmin(Z) + 1e-09, 16)
        m = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        finite = Z[np.isfinite(Z)]
        if finite.size and np.nanmax(finite) > np.nanmin(finite):
            ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.7, alpha=0.75)
    else:
        m = ax.imshow(Z, origin='lower', aspect='auto', cmap=cmap)
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(cbar_label, fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    cb.ax.tick_params(labelsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()
    return (fig, ax)

def plot_recovery_heatmap(results: Mapping[str, Any]) -> Tuple[Any, Any]:
    """Plot moon recovery probability."""
    cfg = results['cfg']
    return _heatmap(results['moon_grid'], 'grid_a_rp', 'grid_radius_re', 'p_hat', f'{cfg.planet.name}: injected-moon recovery', 'Moon separation a/Rp', 'Moon radius [Re]', 'Recovery probability')

def plot_uncertainty_heatmap(results: Mapping[str, Any]) -> Tuple[Any, Any]:
    """Plot recovery-probability uncertainty width."""
    cfg = results['cfg']
    return _heatmap(results['moon_grid'], 'grid_a_rp', 'grid_radius_re', 'p_uncertainty_width', f'{cfg.planet.name}: recovery uncertainty width', 'Moon separation a/Rp', 'Moon radius [Re]', 'Approx. 1σ interval width')

def plot_false_positive_heatmap(results: Mapping[str, Any]) -> Tuple[Any, Any]:
    """Plot no-moon false-positive probability."""
    cfg = results['cfg']
    return _heatmap(results['noise_grid'], 'noise_red_ppm', 'noise_tau_hr', 'p_hat', f'{cfg.planet.name}: no-moon false positives', 'Red noise [ppm]', 'Red-noise timescale [hr]', 'False-positive probability')

def save_grid_results(results: Mapping[str, Any], output_dir: str | Path='results_v31') -> Dict[str, Path]:
    """Save parameters, tables, report, and heatmaps."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {'parameters_json': out / 'parameters.json', 'moon_trials_csv': out / 'moon_trials.csv', 'moon_grid_csv': out / 'moon_grid.csv', 'noise_trials_csv': out / 'noise_trials.csv', 'noise_grid_csv': out / 'noise_grid.csv', 'metrics_csv': out / 'final_detectability_metrics.csv', 'report_txt': out / 'final_detectability_report.txt', 'recovery_png': out / 'heatmap_1_moon_recovery.png', 'uncertainty_png': out / 'heatmap_2_recovery_uncertainty.png', 'false_positive_png': out / 'heatmap_3_false_positive.png'}
    with paths['parameters_json'].open('w') as f:
        json.dump(asdict(results['cfg']), f, indent=2)
    results['moon_trials'].to_csv(paths['moon_trials_csv'], index=False)
    results['moon_grid'].to_csv(paths['moon_grid_csv'], index=False)
    results['noise_trials'].to_csv(paths['noise_trials_csv'], index=False)
    results['noise_grid'].to_csv(paths['noise_grid_csv'], index=False)
    results['metrics'].to_csv(paths['metrics_csv'], index=False)
    paths['report_txt'].write_text(results.get('report', report_from_results(results)))
    fig, _ = plot_recovery_heatmap(results)
    fig.savefig(paths['recovery_png'], dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, _ = plot_uncertainty_heatmap(results)
    fig.savefig(paths['uncertainty_png'], dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, _ = plot_false_positive_heatmap(results)
    fig.savefig(paths['false_positive_png'], dpi=180, bbox_inches='tight')
    plt.close(fig)
    return paths

def _ultranest_available() -> bool:
    """Return whether UltraNest can be imported."""
    try:
        import ultranest
        return True
    except Exception:
        return False

def nested_validation_case_moons(cfg: Config) -> Dict[str, Dict[str, Any]]:
    """Build deterministic best, worst, and intermediate validation moons."""
    a0, a1 = separation_bounds_rp(cfg)

    def _case(radius_re: float, a_rp: float, phase0: float, role: str, description: str):
        """Build one deterministic validation moon."""
        moon = make_moon(cfg, np.random.default_rng(cfg.rng_seed + 4242), radius_re, a_rp, random_phase=False)
        moon['phase0'] = float(phase0)
        moon['mutual_inc_deg'] = 0.0
        moon['ecc'] = 0.0
        moon['case_role'] = role
        moon['case_description'] = description
        return moon
    r_mid = 0.5 * (float(cfg.moon_grid.radius_re_min) + float(cfg.moon_grid.radius_re_max))
    a_mid = 0.5 * (float(a0) + float(a1))
    return {'best_case': _case(cfg.moon_grid.radius_re_max, a1, 0.0, 'bayes_factor_only', 'Largest configured moon radius, widest Hill-limited separation, separated moon-transit phase. Use for Bayes-factor validation, not corner-plot interpretation.'), 'worst_case': _case(cfg.moon_grid.radius_re_min, a0, 0.5 * pi, 'weak_or_non_detection_diagnostic', 'Smallest configured moon radius, closest separation, planet-overlapping moon-transit phase. Use for weak/non-detection behaviour.'), 'intermediate_case': _case(r_mid, a_mid, 0.25 * pi, 'posterior_shape_corner_diagnostic', 'Intermediate radius/separation and partially separated phase. Use for posterior-shape and corner-plot diagnostics.')}

def _run_sampler(names, loglike, transform, ns_cfg: NestedConfig, log_dir: str) -> Mapping[str, Any]:
    """Configure and run one UltraNest sampler."""
    import ultranest
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    sampler = ultranest.ReactiveNestedSampler(names, loglike, transform, log_dir=log_dir, resume=ns_cfg.resume)
    if getattr(ns_cfg, 'use_stepsampler', True):
        from ultranest import stepsampler
        nsteps = getattr(ns_cfg, 'stepsampler_nsteps', None)
        if nsteps is None:
            nsteps = max(50, 10 * len(names))
        step_kwargs = dict(nsteps=int(nsteps))
        adaptive = getattr(ns_cfg, 'stepsampler_adaptive_nsteps', 'move-distance')
        if adaptive is not None:
            step_kwargs['adaptive_nsteps'] = adaptive
        if hasattr(ns_cfg, 'stepsampler_region_filter'):
            step_kwargs['region_filter'] = bool(getattr(ns_cfg, 'stepsampler_region_filter', True))
        try:
            sampler.stepsampler = stepsampler.RegionSliceSampler(**step_kwargs)
        except (TypeError, ValueError):
            step_kwargs.pop('adaptive_nsteps', None)
            try:
                sampler.stepsampler = stepsampler.RegionSliceSampler(**step_kwargs)
            except (TypeError, ValueError):
                step_kwargs.pop('region_filter', None)
                sampler.stepsampler = stepsampler.RegionSliceSampler(**step_kwargs)
    kwargs = dict(min_num_live_points=int(ns_cfg.min_live_points), dlogz=float(ns_cfg.dlogz))
    if ns_cfg.max_ncalls is not None:
        kwargs['max_ncalls'] = int(ns_cfg.max_ncalls)
    if getattr(ns_cfg, 'frac_remain', None) is not None:
        kwargs['frac_remain'] = float(ns_cfg.frac_remain)
    try:
        return sampler.run(**kwargs)
    except TypeError:
        kwargs.pop('max_ncalls', None)
        kwargs.pop('frac_remain', None)
        return sampler.run(**kwargs)

def _samples(result: Mapping[str, Any]) -> np.ndarray:
    """Extract posterior samples from an UltraNest result."""
    s = result.get('samples', None)
    if s is not None:
        return np.asarray(s, dtype=float)
    w = result.get('weighted_samples', {})
    return np.asarray(w.get('points', []), dtype=float)

def _logz(result: Mapping[str, Any]) -> Tuple[float, float]:
    """Extract log-evidence and uncertainty from an UltraNest result."""
    z = result.get('logz', result.get('logZ', np.nan))
    e = result.get('logzerr', result.get('logZerr', np.nan))
    return (float(z), float(e))

def _bayes_factor_summary_row(case_label: str, injected_moon: Mapping[str, Any], z0: float, e0: float, z1: float, e1: float, t_m0: float, t_m1: float) -> Dict[str, Any]:
    """Build one nested-evidence comparison row."""
    ln_k = float(z1 - z0)
    ln_k_err = float(np.sqrt(e0 ** 2 + e1 ** 2)) if np.isfinite(e0) and np.isfinite(e1) else np.nan
    k = float(np.exp(np.clip(ln_k, -700, 700))) if np.isfinite(ln_k) else np.nan
    if np.isfinite(ln_k) and np.isfinite(ln_k_err):
        k_low = float(np.exp(np.clip(ln_k - ln_k_err, -700, 700)))
        k_high = float(np.exp(np.clip(ln_k + ln_k_err, -700, 700)))
    else:
        k_low = np.nan
        k_high = np.nan
    preference = 'M1 planet+moon' if ln_k > 0 else 'M0 planet-only' if ln_k < 0 else 'neither/tie'
    return {'case': case_label, 'case_role': injected_moon.get('case_role', case_label), 'injected_radius_re': float(injected_moon.get('radius_re', np.nan)), 'injected_a_rp': float(injected_moon.get('a_rp', np.nan)), 'injected_phase0_deg': float(np.degrees(injected_moon.get('phase0', np.nan)) % 360.0), 'logz_m0': z0, 'logzerr_m0': e0, 'logz_m1': z1, 'logzerr_m1': e1, 'ln_bayes_factor_m1_over_m0': ln_k, 'ln_bayes_factor_err': ln_k_err, 'bayes_factor_m1_over_m0': k, 'preferred_model_by_lnK_sign': preference}

def nested_bayes_factor_report(comparison: pd.DataFrame) -> str:
    """Build the nested-evidence text report."""
    lines = []
    lines.append('Nested-sampling Bayesian model comparison')
    lines.append('===========================================')
    lines.append('')
    lines.append('M0 = planet-only model plus baseline offset')
    lines.append('M1 = planet + moon model plus baseline offset')
    lines.append('ln K = ln Z_M1 - ln Z_M0; K = exp(ln K)')
    lines.append('')
    lines.append('Case-role footnote: best_case is for Bayes-factor validation only; worst_case is a weak/non-detection diagnostic; intermediate_case is intended for posterior-shape/corner-plot diagnostics.')
    lines.append('')
    if comparison is None or len(comparison) == 0:
        lines.append('No nested comparison rows are available.')
        return '\n'.join(lines)
    for _, row in comparison.iterrows():
        case = row.get('case', 'case')
        role = row.get('case_role', '')
        ln_k = row.get('ln_bayes_factor_m1_over_m0', np.nan)
        ln_k_err = row.get('ln_bayes_factor_err', np.nan)
        k = row.get('bayes_factor_m1_over_m0', np.nan)
        pref = row.get('preferred_model_by_lnK_sign', '')
        lines.append(f'{case} ({role}):')
        lines.append(f"  injected Rm={row.get('injected_radius_re', np.nan):.4g} Re, a={row.get('injected_a_rp', np.nan):.4g} Rp, phase={row.get('injected_phase0_deg', np.nan):.2f} deg")
        lines.append(f"  lnZ(M0)={row.get('logz_m0', np.nan):.4g} ± {row.get('logzerr_m0', np.nan):.3g}; lnZ(M1)={row.get('logz_m1', np.nan):.4g} ± {row.get('logzerr_m1', np.nan):.3g}")
        lines.append(f'  lnK(M1/M0)={ln_k:.4g} ± {ln_k_err:.3g}; K={k:.4g}; sign preference: {pref}')
        lines.append('')
    lines.append('Important: this is a diagnostic model-comparison statistic. The injection/recovery grid and no-moon grid remain the pipeline false-positive controls.')
    return '\n'.join(lines)

def _run_nested_validation_one_case(cfg: Config, injected_moon: Mapping[str, Any], case_label: str, ns_cfg: NestedConfig, truth_backend: str, model_backend: str, output_dir: Path) -> Dict[str, Any]:
    """Fit one simulated validation case with M0 and M1."""
    rng = np.random.default_rng(cfg.rng_seed + 3000 + sum((ord(ch) for ch in str(case_label))))
    moon_for_sim = {k: v for k, v in dict(injected_moon).items() if k not in {'case_role', 'case_description'}}
    time, flux, sigma, injected_for_output = simulate_case(cfg, rng, moon_for_sim, truth_backend)
    injected_for_output = dict(injected_for_output or {})
    injected_for_output['case_role'] = injected_moon.get('case_role', case_label)
    injected_for_output['case_description'] = injected_moon.get('case_description', '')

    def chi_loglike(model):
        """Evaluate the Gaussian log-likelihood."""
        return float(-0.5 * np.sum(((flux - model) / sigma) ** 2 + np.log(2.0 * pi * sigma ** 2)))
    m0_names = ['baseline']

    def m0_transform(cube):
        """Map a unit-cube sample to the M0 prior."""
        return [-0.005 + cube[0] * 0.01]

    def m0_loglike(theta):
        """Evaluate the planet-only likelihood."""
        model = model_flux(time, cfg, None, model_backend) + theta[0]
        return chi_loglike(model)
    a0, a1 = separation_bounds_rp(cfg)
    m1_names = ['baseline', 'moon_radius_re', 'moon_a_rp', 'moon_phase0']

    def m1_transform(cube):
        """Map a unit-cube sample to the M1 prior."""
        return [-0.005 + cube[0] * 0.01, cfg.moon_grid.radius_re_min + cube[1] * (cfg.moon_grid.radius_re_max - cfg.moon_grid.radius_re_min), a0 + cube[2] * (a1 - a0), cube[3] * 2.0 * pi]

    def theta_to_moon(theta):
        """Convert M1 parameters to a moon dictionary."""
        moon = make_moon(cfg, np.random.default_rng(123), theta[1], theta[2], random_phase=False)
        moon['phase0'] = float(theta[3])
        moon['mutual_inc_deg'] = 0.0
        moon['ecc'] = 0.0
        return moon

    def m1_loglike(theta):
        """Evaluate the planet-plus-moon likelihood."""
        model = model_flux(time, cfg, theta_to_moon(theta), model_backend) + theta[0]
        return chi_loglike(model)
    case_out = output_dir / str(case_label)
    case_out.mkdir(parents=True, exist_ok=True)
    t0 = _time.time()
    m0_result = _run_sampler(m0_names, m0_loglike, m0_transform, ns_cfg, str(case_out / 'M0'))
    t_m0 = _time.time() - t0
    t0 = _time.time()
    m1_result = _run_sampler(m1_names, m1_loglike, m1_transform, ns_cfg, str(case_out / 'M1'))
    t_m1 = _time.time() - t0
    m0_s = _samples(m0_result)
    m1_s = _samples(m1_result)
    m0_med = np.nanmedian(m0_s, axis=0) if m0_s.size else np.zeros(len(m0_names))
    m1_med = np.nanmedian(m1_s, axis=0) if m1_s.size else np.zeros(len(m1_names))
    m0_model = model_flux(time, cfg, None, model_backend) + m0_med[0]
    m1_model = model_flux(time, cfg, theta_to_moon(m1_med), model_backend) + m1_med[0]
    z0, e0 = _logz(m0_result)
    z1, e1 = _logz(m1_result)
    comparison_row = _bayes_factor_summary_row(case_label, injected_for_output, z0, e0, z1, e1, t_m0, t_m1)
    pd.DataFrame([comparison_row]).to_csv(case_out / 'nested_evidence_comparison.csv', index=False)
    pd.DataFrame([injected_for_output]).to_csv(case_out / 'nested_injected_moon.csv', index=False)
    return dict(cfg=cfg, case=case_label, truth_backend=truth_backend, model_backend=model_backend, time=time, flux=flux, sigma=sigma, injected_moon=injected_for_output, m0_names=m0_names, m1_names=m1_names, m0_result=m0_result, m1_result=m1_result, m0_median=m0_med, m1_median=m1_med, m0_model=m0_model, m1_model=m1_model, comparison=pd.DataFrame([comparison_row]), output_dir=str(case_out))

def run_nested_validation(cfg: Config, ns_cfg: NestedConfig=NestedConfig(), truth_backend: str='auto', model_backend: Optional[str]=None, output_dir: str | Path='results_v31_nested', case_moons: Optional[Mapping[str, Mapping[str, Any]]]=None) -> Dict[str, Any]:
    """Run and combine the requested nested-validation cases."""
    if not _ultranest_available():
        raise ImportError('UltraNest is not installed, so nested sampling cannot run.')
    if truth_backend == 'auto':
        truth_backend = choose_truth_backend(cfg)
    model_backend = model_backend or truth_backend
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases_in = case_moons or nested_validation_case_moons(cfg)
    cases: Dict[str, Dict[str, Any]] = {}
    comparison_rows: List[pd.DataFrame] = []
    for case_label, moon in cases_in.items():
        one = _run_nested_validation_one_case(cfg=cfg, injected_moon=moon, case_label=str(case_label), ns_cfg=ns_cfg, truth_backend=str(truth_backend).lower(), model_backend=str(model_backend).lower(), output_dir=out)
        cases[str(case_label)] = one
        comparison_rows.append(one['comparison'])
    comparison = pd.concat(comparison_rows, ignore_index=True) if comparison_rows else pd.DataFrame()
    comparison.to_csv(out / 'nested_evidence_comparison_cases.csv', index=False)
    report = nested_bayes_factor_report(comparison)
    (out / 'nested_bayes_factor_report.txt').write_text(report)
    result = dict(cfg=cfg, truth_backend=str(truth_backend).lower(), model_backend=str(model_backend).lower(), cases=cases, comparison=comparison, report=report, output_dir=str(out))
    default_case = 'best_case' if 'best_case' in cases else next(iter(cases), None)
    if default_case is not None:
        result.update({k: v for k, v in cases[default_case].items() if k not in {'comparison', 'output_dir'}})
    return result

def plot_nested_validation(nested: Mapping[str, Any], focus_window_days: Optional[float]=None, case: Optional[str]=None) -> Tuple[Any, Any]:
    """Plot validation data with median M0 and M1 models."""
    if 'cases' in nested:
        cases = nested['cases']
        if case is None:
            case = 'best_case' if 'best_case' in cases else next(iter(cases))
        nested_case = cases[case]
    else:
        nested_case = nested
        case = nested_case.get('case', None)
    cfg = nested_case['cfg']
    t = np.asarray(nested_case['time'])
    f = np.asarray(nested_case['flux'])
    s = np.asarray(nested_case['sigma'])
    centers = cfg.planet.t0_days + np.arange(int(cfg.n_transits)) * cfg.planet.period_days
    idx = np.argmin(np.abs(t[:, None] - centers[None, :]), axis=1)
    x = t - centers[idx]
    if focus_window_days is None:
        focus_window_days = cfg.local_window_days or max(0.2, 10.0 * cfg.noise.cadence_min / (24.0 * 60.0))
    mask = np.abs(x) <= 0.5 * focus_window_days
    order = np.argsort(x[mask])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x[mask][order], f[mask][order], yerr=s[mask][order], fmt='.', ms=2, alpha=0.45, label='validation data')
    ax.plot(x[mask][order], nested_case['m0_model'][mask][order], lw=2, label='M0 median model')
    ax.plot(x[mask][order], nested_case['m1_model'][mask][order], lw=2, label='M1 median model')
    ax.set_xlabel('Time from nearest planet transit [days]', fontsize=14)
    ax.set_ylabel('Relative flux', fontsize=14)
    case_suffix = f' ({case})' if case else ''
    ax.set_title(f'{cfg.planet.name}: nested-sampling validation{case_suffix}', fontsize=15)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend()
    fig.tight_layout()
    return (fig, ax)

def plot_nested_corner(nested: Mapping[str, Any], case: str='intermediate_case', min_spread: float=1e-12) -> Any:
    """Plot a selected M1 posterior when it has sufficient spread."""
    import corner
    if 'cases' in nested:
        nested_case = nested['cases'][case]
    else:
        nested_case = nested
        case = str(nested_case.get('case', case))
    samples = np.asarray(_samples(nested_case['m1_result']), dtype=float)
    labels = list(nested_case['m1_names'])
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    spread = np.nanstd(samples, axis=0)
    keep = np.isfinite(spread) & (spread > float(min_spread))
    if samples.shape[0] < 20 or keep.sum() < 2:
        raise ValueError(f'{case} does not have enough posterior dynamic range for a meaningful corner plot. Sample shape={samples.shape}, std={spread}. Use intermediate_case or increase the nested run budget.')
    fig = corner.corner(samples[:, keep], labels=[label for label, ok in zip(labels, keep) if ok], show_titles=True, title_fmt='.4g', label_kwargs={'fontsize': 14}, title_kwargs={'fontsize': 15})
    fig.suptitle(f'{case} M1 posterior', fontsize=15)
    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=14)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.title.set_size(15)
    fig.tight_layout()
    return fig
