"""WD 0806-661 B transit injection-recovery pipeline using batman for simulated signals, a fast matched-filter search for recovery, and optional UltraNest validation."""
# Return annotations as type hints that describe what a function is expected to return. () -> float e.g.
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from math import pi, log
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List
import importlib
import json
import time as _time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
G = 6.6743e-11
MJUP = 1.89813e+27
RJUP = 71492000.0
REARTH = 6371000.0
DAY = 86400.0

@dataclass
class Primary:
    """Transited primary used by the injection-recovery model."""
    name: str = 'WD 0806-661 B'
    mass_mj: float = 7.0
    radius_rj: float = 1.0
    limb_dark: str = 'uniform'
    u: Tuple[float, ...] = ()

@dataclass
class BodyGrid:
    """Injected body grid: radius in R_earth, separation in primary radii."""
    radius_re_min: float = 0.2
    radius_re_max: float = 2.5
    a_rprimary_min: float = 2.5
    a_rprimary_max: float = 35.0
    impact_b_max: float = 0.85
    ecc_min: float = 0.0
    ecc_max: float = 0.05
    omega_deg: float = 90.0

@dataclass
class Noise:
    instrument: str = 'placeholder_direct_imaging_not_validated'
    cadence_min: float = 10.0
    white_ppm: float = 2500.0
    red_ppm: float = 500.0
    red_timescale_hr: float = 2.0
    duty_cycle: float = 1.0

@dataclass
class Config:
    primary: Primary
    body_grid: BodyGrid
    noise: Noise
    n_transits: int = 10
    local_window_days: Optional[float] = None
    rng_seed: int = 34
    snr_threshold: float = 7.0
    logk_threshold: float = log(10.0)
    supersample_factor: int = 5
    batman_max_err_ppm: float = 1.0
RUN_SIZE_PRESETS = {'smoke': dict(n_radius=3, n_separation=3, n_red=3, n_tau=3, n_per_cell=2), 'quick': dict(n_radius=6, n_separation=6, n_red=5, n_tau=5, n_per_cell=3), 'standard': dict(n_radius=10, n_separation=10, n_red=8, n_tau=8, n_per_cell=10), 'long': dict(n_radius=30, n_separation=30, n_red=12, n_tau=12, n_per_cell=20)}

def placeholder_direct_imaging_noise() -> Noise:
    """Return the default unvalidated direct-imaging sensitivity assumptions."""
    return Noise(instrument='placeholder_direct_imaging_not_validated', cadence_min=10.0, white_ppm=2500.0, red_ppm=500.0, red_timescale_hr=2.0, duty_cycle=1.0)

def pessimistic_direct_imaging_noise() -> Noise:
    """Return a noisier stress-test sensitivity model."""
    return Noise(instrument='pessimistic_direct_imaging_stress_test', cadence_min=10.0, white_ppm=10000.0, red_ppm=3000.0, red_timescale_hr=2.0, duty_cycle=1.0)

def jwst_like_noise() -> Noise:
    """Return an editable JWST-like comparison model that is not an ETC prediction."""
    return Noise(instrument='jwst_like_comparison_not_etc', cadence_min=5.0, white_ppm=1000.0, red_ppm=300.0, red_timescale_hr=2.0, duty_cycle=1.0)

def make_wd0806_config(seed: int=1, noise: Optional[Noise]=None) -> Config:
    """Create the default WD 0806 configuration with an optional replacement noise model."""
    return Config(primary=Primary(), body_grid=BodyGrid(), noise=placeholder_direct_imaging_noise() if noise is None else noise, rng_seed=int(seed))

def parameter_input_dataframes(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Convert the configuration dataclasses into notebook-ready tables."""
    return {'primary': pd.DataFrame([asdict(cfg.primary)]), 'body_grid': pd.DataFrame([asdict(cfg.body_grid)]), 'noise': pd.DataFrame([asdict(cfg.noise)]), 'experiment': pd.DataFrame([{k: v for k, v in asdict(cfg).items() if k not in {'primary', 'body_grid', 'noise'}}])}

def require_batman() -> Any:
    """Import batman and raise a clear installation error if it is unavailable."""
    try:
        return importlib.import_module('batman')
    except Exception as exc:
        raise ImportError('Install batman first: pip install batman-package') from exc

def period_from_semi_major_axis(a_m: float, m_primary_kg: float) -> float:
    """Calculate orbital period from separation and primary mass using Kepler’s third law."""
    return float(2.0 * pi * np.sqrt(a_m ** 3 / (G * m_primary_kg)) / DAY)

def orbital_speed(a_m: float, period_days: float) -> float:
    """Calculate circular orbital speed from semi-major axis and period."""
    return float(2.0 * pi * a_m / (period_days * DAY))

def inclination_deg_from_b(b: float, a_over_r: float) -> float:
    """Convert impact parameter to inclination for a specified scaled separation."""
    return float(np.degrees(np.arccos(np.clip(float(b) / max(float(a_over_r), 1e-12), -1.0, 1.0))))

def transit_duration_days(primary_r_m: float, body_r_m: float, a_m: float, period_days: float, b: float) -> float:
    """Estimate the chord-crossing transit duration for the adopted geometry."""
    chord = np.sqrt(max((1.0 + body_r_m / primary_r_m) ** 2 - float(b) ** 2, 0.0)) * primary_r_m
    return float(2.0 * chord / orbital_speed(a_m, period_days) / DAY)

def body_period_days(cfg: Config, a_rprimary: float) -> float:
    """Calculate an injected body’s period from its separation in primary radii."""
    return period_from_semi_major_axis(float(a_rprimary) * cfg.primary.radius_rj * RJUP, cfg.primary.mass_mj * MJUP)

def body_duration_days(cfg: Config, radius_re: float, a_rprimary: float, b: float=0.0) -> float:
    """Estimate an injected body’s transit duration around the configured primary."""
    primary_r = cfg.primary.radius_rj * RJUP
    body_r = float(radius_re) * REARTH
    a_m = float(a_rprimary) * primary_r
    return transit_duration_days(primary_r, body_r, a_m, body_period_days(cfg, a_rprimary), b)

def make_body(cfg: Config, rng: np.random.Generator, radius_re: float, a_rprimary: float) -> Dict[str, float]:
    """Create one injected body at fixed grid coordinates with random allowed geometry."""
    rp_over_r = float(radius_re) * REARTH / (cfg.primary.radius_rj * RJUP)
    bmax = min(float(cfg.body_grid.impact_b_max), 1.0 + rp_over_r - 0.0001)
    return {'radius_re': float(radius_re), 'a_rprimary': float(a_rprimary), 'period_days': body_period_days(cfg, a_rprimary), 'impact_b': float(rng.uniform(0.0, max(bmax, 0.0))), 'ecc': float(rng.uniform(cfg.body_grid.ecc_min, cfg.body_grid.ecc_max)), 'omega_deg': float(cfg.body_grid.omega_deg), 't0_days': 0.0}

def batman_flux(time: np.ndarray, cfg: Config, body: Mapping[str, float]) -> np.ndarray:
    """Generate the finite-cadence batman light curve for one injected body."""
    batman = require_batman()
    p = batman.TransitParams()
    p.t0 = float(body.get('t0_days', 0.0))
    p.per = float(body['period_days'])
    p.rp = float(body['radius_re']) * REARTH / (cfg.primary.radius_rj * RJUP)
    p.a = float(body['a_rprimary'])
    p.inc = inclination_deg_from_b(float(body.get('impact_b', 0.0)), p.a)
    p.ecc = float(body.get('ecc', 0.0))
    p.w = float(body.get('omega_deg', cfg.body_grid.omega_deg))
    p.u = list(cfg.primary.u)
    p.limb_dark = cfg.primary.limb_dark
    exp_days = cfg.noise.cadence_min / (24.0 * 60.0)
    model = batman.TransitModel(p, np.asarray(time, dtype=float), max_err=float(cfg.batman_max_err_ppm), supersample_factor=max(1, int(cfg.supersample_factor)), exp_time=exp_days if int(cfg.supersample_factor) > 1 else 0.0)
    return np.asarray(model.light_curve(p), dtype=float)

def make_time_grid(cfg: Config, body: Mapping[str, float]) -> np.ndarray:
    """Build observing windows around the expected transit times."""
    cadence_days = cfg.noise.cadence_min / (24.0 * 60.0)
    dur = body_duration_days(cfg, body['radius_re'], body['a_rprimary'], body.get('impact_b', 0.0))
    window = cfg.local_window_days if cfg.local_window_days is not None else max(8.0 * dur, 12.0 * cadence_days)
    chunks = []
    for n in range(int(cfg.n_transits)):
        tc = float(body.get('t0_days', 0.0)) + n * float(body['period_days'])
        chunks.append(np.arange(tc - window / 2.0, tc + window / 2.0 + cadence_days / 2.0, cadence_days))
    time = np.unique(np.sort(np.concatenate(chunks)))
    if cfg.noise.duty_cycle < 1.0:
        rng = np.random.default_rng(cfg.rng_seed + 17)
        time = time[rng.random(time.size) < cfg.noise.duty_cycle]
    return time

def red_noise_ar1(time: np.ndarray, red_ppm: float, tau_hr: float, rng: np.random.Generator) -> np.ndarray:
    """Generate exponentially correlated Gaussian noise on an irregular time grid."""
    sigma = float(red_ppm) * 1e-06
    if sigma <= 0 or len(time) == 0:
        return np.zeros_like(time, dtype=float)
    tau_days = max(float(tau_hr) / 24.0, 1e-12)
    y = np.zeros_like(time, dtype=float)
    y[0] = rng.normal(0.0, sigma)
    for i in range(1, len(time)):
        rho = np.exp(-max(float(time[i] - time[i - 1]), 0.0) / tau_days)
        y[i] = rho * y[i - 1] + sigma * np.sqrt(max(1.0 - rho ** 2, 0.0)) * rng.normal()
    return y

def add_noise(flux: np.ndarray, time: np.ndarray, noise: Noise, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Add white and correlated noise and return the noisy flux and adopted uncertainty."""
    white = float(noise.white_ppm) * 1e-06
    y = np.asarray(flux, dtype=float) + rng.normal(0.0, white, size=len(flux))
    if noise.red_ppm > 0:
        y = y + red_noise_ar1(time, noise.red_ppm, noise.red_timescale_hr, rng)
    sigma = np.full_like(y, np.sqrt(noise.white_ppm ** 2 + noise.red_ppm ** 2) * 1e-06, dtype=float)
    return (y, sigma)

def simulate_case(cfg: Config, rng: np.random.Generator, body: Optional[Mapping[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
    """Simulate an injected-body or no-object light curve on a matched time grid."""
    body_use = dict(body) if body is not None else make_body(cfg, rng, np.sqrt(cfg.body_grid.radius_re_min * cfg.body_grid.radius_re_max), np.sqrt(cfg.body_grid.a_rprimary_min * cfg.body_grid.a_rprimary_max))
    time = make_time_grid(cfg, body_use)
    clean = np.ones_like(time, dtype=float) if body is None else batman_flux(time, cfg, body_use)
    flux, sigma = add_noise(clean, time, cfg.noise, rng)
    return (time, flux, sigma, None if body is None else body_use)

def box_improvement(resid: np.ndarray, sigma: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    """Fit a positive box-shaped transit depth and return its chi-square improvement and SNR."""
    if mask.sum() < 2:
        return (0.0, 0.0, 0.0)
    w = 1.0 / np.asarray(sigma) ** 2
    depth = -np.sum(w[mask] * resid[mask]) / np.sum(w[mask])
    if depth <= 0:
        return (0.0, 0.0, 0.0)
    chi0 = np.sum((resid / sigma) ** 2)
    chi1 = np.sum(((resid + depth * mask.astype(float)) / sigma) ** 2)
    snr = depth * np.sqrt(np.sum(w[mask]))
    return (float(max(chi0 - chi1, 0.0)), float(depth), float(snr))

def recover_body_fast(time: np.ndarray, flux: np.ndarray, sigma: np.ndarray, cfg: Config, n_sep: int=24, n_t0: int=9) -> Dict[str, Any]:
    """Search separation, epoch, and duration grids for the highest-scoring transit-like signal."""
    resid = np.asarray(flux, dtype=float) - np.nanmedian(flux)
    sep_grid = np.logspace(np.log10(cfg.body_grid.a_rprimary_min), np.log10(cfg.body_grid.a_rprimary_max), int(n_sep))
    r_mid = 0.5 * (cfg.body_grid.radius_re_min + cfg.body_grid.radius_re_max)
    best = dict(delta_chi2=0.0, depth=0.0, snr=0.0, a_rprimary=np.nan, period_days=np.nan, t0_days=np.nan, duration_days=np.nan)
    n_trials = 0
    for sep in sep_grid:
        p_days = body_period_days(cfg, sep)
        dur0 = body_duration_days(cfg, r_mid, sep, 0.0)
        window = cfg.local_window_days if cfg.local_window_days is not None else max(8.0 * dur0, 12.0 * cfg.noise.cadence_min / (24.0 * 60.0))
        for t0 in np.linspace(-0.35 * window, 0.35 * window, int(n_t0)):
            centers = [float(t0) + n * p_days for n in range(int(cfg.n_transits))]
            for dur in np.array([0.6, 1.0, 1.7]) * dur0:
                mask = np.zeros_like(time, dtype=bool)
                for c in centers:
                    mask |= np.abs(time - c) <= 0.5 * dur
                dchi2, depth, snr = box_improvement(resid, sigma, mask)
                n_trials += 1
                if dchi2 > best['delta_chi2']:
                    best.update(delta_chi2=dchi2, depth=depth, snr=snr, a_rprimary=float(sep), period_days=float(p_days), t0_days=float(t0), duration_days=float(dur))
    delta_bic = best['delta_chi2'] - 4.0 * np.log(max(len(time), 2))
    logk_proxy = 0.5 * delta_bic - np.log(max(n_trials, 1))
    best['delta_bic_proxy'] = float(delta_bic)
    best['logk_proxy'] = float(logk_proxy)
    best['equiv_radius_re'] = float(np.sqrt(max(best['depth'], 0.0)) * cfg.primary.radius_rj * RJUP / REARTH)
    best['detected'] = bool(best['snr'] >= cfg.snr_threshold and logk_proxy >= cfg.logk_threshold)
    return best

def default_grids(cfg: Config, n_radius: int=6, n_separation: int=6, n_red: int=5, n_tau: int=5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct the body and red-noise grids used by the experiment."""
    radius_grid = np.logspace(np.log10(cfg.body_grid.radius_re_min), np.log10(cfg.body_grid.radius_re_max), int(n_radius))
    sep_grid = np.logspace(np.log10(cfg.body_grid.a_rprimary_min), np.log10(cfg.body_grid.a_rprimary_max), int(n_separation))
    red_grid = np.linspace(0.0, max(1.0, 2.0 * cfg.noise.red_ppm), int(n_red))
    tau_grid = np.linspace(0.5, max(1.0, 2.0 * cfg.noise.red_timescale_hr), int(n_tau))
    return (radius_grid, sep_grid, red_grid, tau_grid)

def wilson_interval(k: float, n: float, z: float=1.0) -> Tuple[float, float]:
    """Calculate a Wilson score interval for a binomial proportion."""
    k = float(k)
    n = float(n)
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2.0 * n)) / denom
    half = z / denom * np.sqrt(p * (1.0 - p) / n + z ** 2 / (4.0 * n ** 2))
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))

def binomial_probability_interval(k: float, n: float, method: str='wilson', cred: float=0.6827) -> Tuple[float, float]:
    """Calculate the selected binomial interval used for a grid-cell probability."""
    k = float(k)
    n = float(n)
    if n <= 0:
        return (float('nan'), float('nan'))
    method = str(method).lower().replace('-', '_')
    if method == 'wilson':
        z = 1.0 if abs(float(cred) - 0.6827) < 0.05 else 1.96
        return wilson_interval(k, n, z=z)
    if method in {'wald', 'normal'}:
        p_hat = k / n
        z = 1.0 if abs(float(cred) - 0.6827) < 0.05 else 1.96
        half = z * np.sqrt(max(p_hat * (1.0 - p_hat) / n, 0.0))
        return (float(max(0.0, p_hat - half)), float(min(1.0, p_hat + half)))
    raise ValueError('method must be one of: wilson, wald')

def summarise_grid(trials: pd.DataFrame, x: str, y: str, interval_method: str='wilson', interval_cred: float=0.6827) -> pd.DataFrame:
    """Aggregate trial outcomes into exact grid-cell probabilities and interval widths."""
    grouped = trials.groupby([x, y], dropna=False).agg(n=('detected', 'size'), k_detected=('detected', 'sum'), p_hat=('detected', 'mean')).reset_index()
    lows_68, highs_68 = ([], [])
    lows_95, highs_95 = ([], [])
    lows_selected, highs_selected = ([], [])
    for k, n in zip(grouped['k_detected'].astype(float), grouped['n'].astype(float)):
        low_68, high_68 = wilson_interval(k, n, z=1.0)
        low_95, high_95 = wilson_interval(k, n, z=1.96)
        low_selected, high_selected = binomial_probability_interval(k, n, method=interval_method, cred=interval_cred)
        lows_68.append(low_68)
        highs_68.append(high_68)
        lows_95.append(low_95)
        highs_95.append(high_95)
        lows_selected.append(low_selected)
        highs_selected.append(high_selected)
    grouped['p_wilson68_low'] = lows_68
    grouped['p_wilson68_high'] = highs_68
    grouped['p_wilson95_low'] = lows_95
    grouped['p_wilson95_high'] = highs_95
    grouped['p_wilson68_width'] = grouped['p_wilson68_high'] - grouped['p_wilson68_low']
    grouped['p_interval_method'] = interval_method
    grouped['p_interval_cred'] = float(interval_cred)
    grouped['p_interval_low'] = lows_selected
    grouped['p_interval_high'] = highs_selected
    grouped['p_interval_width'] = grouped['p_interval_high'] - grouped['p_interval_low']
    grouped['p_uncertainty_width'] = grouped['p_interval_width']
    return grouped

def _trigger_counts(body_trials: pd.DataFrame, noise_trials: pd.DataFrame) -> Tuple[int, int, int, int, float, float]:
    """Return whole-grid recovery and false-positive counts and rates."""
    n_body = len(body_trials)
    n_null = len(noise_trials)
    k_rec = int(body_trials['detected'].sum()) if n_body else 0
    k_fp = int(noise_trials['detected'].sum()) if n_null else 0
    return {'k_rec': k_rec, 'n_body': n_body, 'recovery': k_rec / n_body if n_body else np.nan, 'k_fp': k_fp, 'n_null': n_null, 'fpr': k_fp / n_null if n_null else np.nan}

def final_detectability_report(body_trials: pd.DataFrame, noise_trials: pd.DataFrame, priors: Sequence[float]=(0.01, 0.05, 0.1, 0.5)) -> str:
    """Create the whole-grid recovery, false-positive, and prior-weighted report."""
    stats = _trigger_counts(body_trials, noise_trials)
    rec = stats['recovery']
    fpr = stats['fpr']
    lines = ['Combined WD 0806-661 B detectability statistic', '===============================================', f"Injected batman transits recovered: {stats['k_rec']}/{stats['n_body']} = {rec:.4f} ({100 * rec:.2f}%)", f"No-object false positives: {stats['k_fp']}/{stats['n_null']} = {fpr:.4f} ({100 * fpr:.2f}%)", '', 'D(pi) = pi * recovery / [pi * recovery + (1 - pi) * false_positive]']
    for prior in priors:
        denominator = prior * rec + (1.0 - prior) * fpr
        value = prior * rec / denominator if denominator > 0 else np.nan
        lines.append(f'pi={prior:g}: D={value:.4f}')
    return '\n'.join(lines)

def final_metrics(body_trials: pd.DataFrame, noise_trials: pd.DataFrame) -> pd.DataFrame:
    """Return a compact table of whole-grid recovery and false-positive rates."""
    stats = _trigger_counts(body_trials, noise_trials)
    return pd.DataFrame([{'quantity': 'Recovery probability / TPR', 'value': stats['recovery'], 'count': f"{stats['k_rec']}/{stats['n_body']}"}, {'quantity': 'False-positive probability / FPR', 'value': stats['fpr'], 'count': f"{stats['k_fp']}/{stats['n_null']}"}])

def run_grid_suite(cfg: Config, n_radius: int=6, n_separation: int=6, n_red: int=5, n_tau: int=5, n_per_cell: int=3, progress: bool=True) -> Dict[str, Any]:
    """Run the injected-body recovery grid and matched no-object false-positive grid."""
    radius_grid, sep_grid, red_grid, tau_grid = default_grids(cfg, n_radius, n_separation, n_red, n_tau)
    rng = np.random.default_rng(cfg.rng_seed)
    body_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(radius_grid, 1):
        for j, a in enumerate(sep_grid, 1):
            if progress:
                print(f'body grid {i}/{len(radius_grid)}, {j}/{len(sep_grid)}: R={r:.3g} Re, a={a:.3g} Rprimary')
            for rep in range(int(n_per_cell)):
                body = make_body(cfg, rng, r, a)
                t, f, s, inj = simulate_case(cfg, rng, body)
                rec = recover_body_fast(t, f, s, cfg)
                body_rows.append({'grid_radius_re': float(r), 'grid_a_rprimary': float(a), 'rep': rep, **rec, **{f'inj_{k}': v for k, v in inj.items()}})
    body_trials = pd.DataFrame(body_rows)
    body_trials['detected'] = body_trials['detected'].astype(bool)
    body_grid = summarise_grid(body_trials, 'grid_a_rprimary', 'grid_radius_re')
    noise_rows: List[Dict[str, Any]] = []
    for red in red_grid:
        for tau in tau_grid:
            if progress:
                print(f'noise grid: red={red:.3g} ppm, tau={tau:.3g} hr')
            cfg_n = replace(cfg, noise=replace(cfg.noise, red_ppm=float(red), red_timescale_hr=float(tau)))
            for rep in range(int(n_per_cell)):
                t, f, s, _ = simulate_case(cfg_n, rng, None)
                rec = recover_body_fast(t, f, s, cfg_n)
                noise_rows.append({'noise_red_ppm': float(red), 'noise_tau_hr': float(tau), 'rep': rep, **rec})
    noise_trials = pd.DataFrame(noise_rows)
    noise_trials['detected'] = noise_trials['detected'].astype(bool)
    noise_grid = summarise_grid(noise_trials, 'noise_red_ppm', 'noise_tau_hr')
    report = final_detectability_report(body_trials, noise_trials)
    return dict(cfg=cfg, body_trials=body_trials, body_grid=body_grid, noise_trials=noise_trials, noise_grid=noise_grid, metrics=final_metrics(body_trials, noise_trials), report=report)

def threshold_sensitivity_summary(body_trials: pd.DataFrame, noise_trials: pd.DataFrame, snr_cuts: Sequence[float]=(5.0, 7.0, 8.0, 10.0), logk_cuts: Sequence[float]=(log(10.0),)) -> pd.DataFrame:
    """Recalculate recovery and false-positive rates for alternative trigger thresholds."""
    rows: List[Dict[str, Any]] = []
    for snr_cut in snr_cuts:
        for logk_cut in logk_cuts:
            bdet = (body_trials['snr'] >= float(snr_cut)) & (body_trials['logk_proxy'] >= float(logk_cut))
            ndet = (noise_trials['snr'] >= float(snr_cut)) & (noise_trials['logk_proxy'] >= float(logk_cut))
            rows.append({'snr_cut': float(snr_cut), 'logk_cut': float(logk_cut), 'body_recovered': int(bdet.sum()), 'body_total': int(len(body_trials)), 'recovery_probability': float(bdet.mean()) if len(body_trials) else np.nan, 'noise_false_positives': int(ndet.sum()), 'noise_total': int(len(noise_trials)), 'false_positive_probability': float(ndet.mean()) if len(noise_trials) else np.nan})
    return pd.DataFrame(rows)

def run_wd0806_batman_pipeline(cfg: Optional[Config]=None, run_size: str='quick', seed: int=1, progress: bool=True) -> Dict[str, Any]:
    """Apply a named run-size preset and execute the complete grid workflow."""
    cfg_use = make_wd0806_config(seed=seed) if cfg is None else replace(cfg, rng_seed=int(seed))
    params = dict(RUN_SIZE_PRESETS[str(run_size).lower()])
    out = run_grid_suite(cfg_use, progress=progress, **params)
    out['run_size'] = run_size
    out['run_size_parameters'] = params
    out['threshold_sensitivity'] = threshold_sensitivity_summary(out['body_trials'], out['noise_trials'])
    return out

def _pivot(grid: pd.DataFrame, x: str, y: str, z: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape a long-form grid table into sorted arrays for heatmap plotting."""
    p = grid.pivot(index=y, columns=x, values=z).sort_index().sort_index(axis=1)
    return (p.columns.to_numpy(float), p.index.to_numpy(float), p.to_numpy(float))

def _format_positive_log_axis(axis, values: np.ndarray) -> None:
    """Place readable numeric ticks on a positive logarithmic axis."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        return
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        return
    ticks: List[float] = []
    lo_exp = int(np.floor(np.log10(vmin))) - 1
    hi_exp = int(np.ceil(np.log10(vmax))) + 1
    for exponent in range(lo_exp, hi_exp + 1):
        scale = 10.0 ** exponent
        for sub in (1.0, 2.0, 5.0):
            value = sub * scale
            if vmin <= value <= vmax:
                ticks.append(float(value))
    if not ticks:
        ticks = [vmin, vmax]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}' if value > 0 else ''))
    axis.set_minor_locator(LogLocator(base=10.0, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=12))
    axis.set_minor_formatter(NullFormatter())

def _label_with_log_note(label: str, is_log: bool) -> str:
    """Mark an axis label as logarithmic without duplicating the note."""
    if not is_log:
        return label
    return label if 'log' in str(label).lower() else f'{label} (log scale)'

def _heatmap(grid: pd.DataFrame, x: str, y: str, z: str, title: str, xlabel: str, ylabel: str, cbar_label: str, contour_levels: Sequence[float]=(0.5, 0.9)) -> Tuple[Any, Any]:
    """Plot a gridded statistic with optional contours and logarithmic axes."""
    xs, ys, Z = _pivot(grid, x, y, z)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    X, Y = np.meshgrid(xs, ys)
    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        m = ax.imshow(Z, origin='lower', aspect='auto', cmap='RdYlBu')
    elif Z.shape[0] >= 2 and Z.shape[1] >= 2:
        zmin = float(np.nanmin(finite))
        zmax = float(np.nanmax(finite))
        if zmax <= zmin:
            zmax = zmin + 1e-09
        m = ax.contourf(X, Y, Z, levels=np.linspace(zmin, zmax, 16), cmap='RdYlBu')
        valid_levels = [lev for lev in contour_levels if zmin < float(lev) < zmax]
        if valid_levels:
            ax.contour(X, Y, Z, levels=valid_levels, colors='black', linewidths=1.4)
    else:
        m = ax.imshow(Z, origin='lower', aspect='auto', cmap='RdYlBu')
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(cbar_label, fontsize=14)
    cb.ax.tick_params(labelsize=14)
    x_is_log = bool(np.nanmin(xs) > 0)
    y_is_log = bool(np.nanmin(ys) > 0)
    if x_is_log:
        ax.set_xscale('log')
        ax.set_xlim(float(np.nanmin(xs)), float(np.nanmax(xs)))
        _format_positive_log_axis(ax.xaxis, xs)
    if y_is_log:
        ax.set_yscale('log')
        ax.set_ylim(float(np.nanmin(ys)), float(np.nanmax(ys)))
        _format_positive_log_axis(ax.yaxis, ys)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(_label_with_log_note(xlabel, x_is_log), fontsize=14)
    ax.set_ylabel(_label_with_log_note(ylabel, y_is_log), fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()
    return (fig, ax)

def plot_recovery_heatmap(results: Mapping[str, Any]) -> Tuple[Any, Any]:
    """Plot injected-body recovery probability over radius and separation."""
    return _heatmap(results['body_grid'], 'grid_a_rprimary', 'grid_radius_re', 'p_hat', 'WD 0806-661 B: batman-injected transit recovery', 'Separation a/Rprimary', 'Body radius [Re]', 'Recovery probability', contour_levels=(0.5, 0.9))

def plot_uncertainty_heatmap(results: Mapping[str, Any]) -> Tuple[Any, Any]:
    """Plot binomial uncertainty width over the injected-body grid."""
    return _heatmap(results['body_grid'], 'grid_a_rprimary', 'grid_radius_re', 'p_uncertainty_width', 'WD 0806-661 B: recovery uncertainty width', 'Separation a/Rprimary', 'Body radius [Re]', 'Binomial interval width', contour_levels=())

def plot_false_positive_heatmap(results: Mapping[str, Any]) -> Tuple[Any, Any]:
    """Plot no-object false-positive probability over red-noise parameters."""
    return _heatmap(results['noise_grid'], 'noise_red_ppm', 'noise_tau_hr', 'p_hat', 'WD 0806-661 B: no-object false positives', 'Red noise [ppm]', 'Red-noise timescale [hr]', 'False-positive probability', contour_levels=(0.01, 0.05, 0.1))

def save_grid_results(results: Mapping[str, Any], output_dir: str | Path='result_wd0806_v15') -> Dict[str, Path]:
    """Save parameters, tables, report text, and the three heatmaps."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {'parameters_json': out / 'parameters.json', 'body_trials_csv': out / 'body_trials.csv', 'body_grid_csv': out / 'body_grid.csv', 'noise_trials_csv': out / 'noise_trials.csv', 'noise_grid_csv': out / 'noise_grid.csv', 'metrics_csv': out / 'final_detectability_metrics.csv', 'report_txt': out / 'final_detectability_report.txt', 'recovery_png': out / 'heatmap_1_body_recovery.png', 'uncertainty_png': out / 'heatmap_2_recovery_uncertainty.png', 'false_positive_png': out / 'heatmap_3_false_positive.png'}
    with paths['parameters_json'].open('w') as f:
        json.dump(asdict(results['cfg']), f, indent=2)
    results['body_trials'].to_csv(paths['body_trials_csv'], index=False)
    results['body_grid'].to_csv(paths['body_grid_csv'], index=False)
    results['noise_trials'].to_csv(paths['noise_trials_csv'], index=False)
    results['noise_grid'].to_csv(paths['noise_grid_csv'], index=False)
    results['metrics'].to_csv(paths['metrics_csv'], index=False)
    paths['report_txt'].write_text(results['report'])
    for key, plotter in [('recovery_png', plot_recovery_heatmap), ('uncertainty_png', plot_uncertainty_heatmap), ('false_positive_png', plot_false_positive_heatmap)]:
        fig, _ = plotter(results)
        fig.savefig(paths[key], dpi=180, bbox_inches='tight')
        plt.close(fig)
    return paths

@dataclass
class NestedConfig:
    """UltraNest controls for the optional diagnostic model comparison."""
    min_live_points: int = 80
    dlogz: float = 0.5
    max_ncalls: Optional[int] = None
    best_case_max_ncalls: Optional[int] = 100_000
    resume: str = 'overwrite'
    use_stepsampler: bool = True
    stepsampler_nsteps: Optional[int] = 100
    stepsampler_adaptive_nsteps: Any = 'move-distance'
    stepsampler_region_filter: bool = True
    frac_remain: Optional[float] = 0.5
    storage_backend: str = 'hdf5'

def nested_validation_case_bodies(cfg: Config, include_intermediate: bool=False) -> Dict[str, Dict[str, Any]]:
    """Construct deterministic best, worst, and optional intermediate validation cases."""
    r_best = float(cfg.body_grid.radius_re_max)
    a_best = float(cfg.body_grid.a_rprimary_max)
    r_worst = float(cfg.body_grid.radius_re_min)
    a_worst = float(cfg.body_grid.a_rprimary_min)
    r_mid = float(np.sqrt(cfg.body_grid.radius_re_min * cfg.body_grid.radius_re_max))
    a_mid = float(np.sqrt(cfg.body_grid.a_rprimary_min * cfg.body_grid.a_rprimary_max))

    def bmax_for_radius(radius_re: float):
        """Calculate the largest permitted impact parameter for a supplied body radius."""
        rp_over_r = float(radius_re) * REARTH / (cfg.primary.radius_rj * RJUP)
        return float(min(cfg.body_grid.impact_b_max, 1.0 + rp_over_r - 0.0001))
    best = {'radius_re': r_best, 'a_rprimary': a_best, 'period_days': body_period_days(cfg, a_best), 'impact_b': 0.0, 'ecc': 0.0, 'omega_deg': float(cfg.body_grid.omega_deg), 't0_days': 0.0, 'case_role': 'best_case', 'case_description': 'Largest configured radius, widest configured separation, central transit, circular orbit.'}
    worst = {'radius_re': r_worst, 'a_rprimary': a_worst, 'period_days': body_period_days(cfg, a_worst), 'impact_b': max(0.0, bmax_for_radius(r_worst)), 'ecc': 0.0, 'omega_deg': float(cfg.body_grid.omega_deg), 't0_days': 0.0, 'case_role': 'worst_case', 'case_description': 'Smallest configured radius, closest configured separation, near-grazing transit, circular orbit.'}
    intermediate = {'radius_re': r_mid, 'a_rprimary': a_mid, 'period_days': body_period_days(cfg, a_mid), 'impact_b': min(0.3, max(0.0, bmax_for_radius(r_mid))), 'ecc': 0.0, 'omega_deg': float(cfg.body_grid.omega_deg), 't0_days': 0.0, 'case_role': 'posterior_shape_corner_diagnostic', 'case_description': 'Geometric-midpoint radius/separation and non-central transit. Run separately for posterior-shape diagnostics.'}
    cases = {'best_case': best, 'worst_case': worst}
    if include_intermediate:
        cases['intermediate_case'] = intermediate
    return cases

def _run_sampler(names, loglike, transform, ns_cfg: NestedConfig, log_dir: str) -> Mapping[str, Any]:
    """Run UltraNest while preserving any requested likelihood-call cutoff."""
    import inspect
    import ultranest

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    sampler = ultranest.ReactiveNestedSampler(
        names,
        loglike,
        transform,
        log_dir=log_dir,
        resume=ns_cfg.resume,
        storage_backend=str(ns_cfg.storage_backend),
    )

    if ns_cfg.use_stepsampler:
        from ultranest import stepsampler

        nsteps = ns_cfg.stepsampler_nsteps
        if nsteps is None:
            nsteps = max(50, 10 * len(names))

        step_kwargs = {"nsteps": int(nsteps)}
        if ns_cfg.stepsampler_adaptive_nsteps is not None:
            step_kwargs["adaptive_nsteps"] = ns_cfg.stepsampler_adaptive_nsteps
        step_kwargs["region_filter"] = bool(ns_cfg.stepsampler_region_filter)

        try:
            sampler.stepsampler = stepsampler.RegionSliceSampler(**step_kwargs)
        except (TypeError, ValueError):
            step_kwargs.pop("adaptive_nsteps", None)
            try:
                sampler.stepsampler = stepsampler.RegionSliceSampler(**step_kwargs)
            except (TypeError, ValueError):
                step_kwargs.pop("region_filter", None)
                sampler.stepsampler = stepsampler.RegionSliceSampler(**step_kwargs)

    run_kwargs = {
        "min_num_live_points": int(ns_cfg.min_live_points),
        "dlogz": float(ns_cfg.dlogz),
    }
    if ns_cfg.max_ncalls is not None:
        run_kwargs["max_ncalls"] = int(ns_cfg.max_ncalls)
    if ns_cfg.frac_remain is not None:
        run_kwargs["frac_remain"] = float(ns_cfg.frac_remain)

    # Remove only unsupported optional keywords. A requested max_ncalls cutoff is
    # never discarded silently because that would allow the best case to run on.
    try:
        parameters = inspect.signature(sampler.run).parameters
    except (TypeError, ValueError):
        parameters = {}
    if parameters:
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        unsupported = [
            key for key in run_kwargs
            if key not in parameters and not accepts_kwargs
        ]
        if "max_ncalls" in unsupported:
            raise RuntimeError(
                "This UltraNest version does not expose max_ncalls, so the "
                "requested nested-sampling cutoff cannot be guaranteed."
            )
        for key in unsupported:
            run_kwargs.pop(key)

    return sampler.run(**run_kwargs)

def posterior_samples(result: Mapping[str, Any]) -> np.ndarray:
    """Extract posterior samples from either standard or weighted UltraNest output."""
    s = result.get('samples', None)
    if s is not None:
        return np.asarray(s, dtype=float)
    w = result.get('weighted_samples', {})
    return np.asarray(w.get('points', []), dtype=float)
_samples = posterior_samples

def _logz(result: Mapping[str, Any]) -> Tuple[float, float]:
    """Extract log-evidence and its uncertainty from an UltraNest result."""
    z = result.get('logz', result.get('logZ', np.nan))
    e = result.get('logzerr', result.get('logZerr', np.nan))
    return (float(z), float(e))

def _nested_t0_prior_half_width_days(cfg: Config) -> float:
    """Set the symmetric transit-epoch prior width from the observing window."""
    r_ref = float(np.sqrt(cfg.body_grid.radius_re_min * cfg.body_grid.radius_re_max))
    a_ref = float(np.sqrt(cfg.body_grid.a_rprimary_min * cfg.body_grid.a_rprimary_max))
    dur = body_duration_days(cfg, r_ref, a_ref, b=0.0)
    window = cfg.local_window_days if cfg.local_window_days is not None else max(8.0 * dur, 12.0 * cfg.noise.cadence_min / (24.0 * 60.0))
    return float(0.5 * window)

def _bayes_factor_summary_row(case_label: str, injected_body: Mapping[str, Any], z0: float, e0: float, z1: float, e1: float, t_m0: float, t_m1: float) -> Dict[str, Any]:
    """Convert two evidence results into one model-comparison summary row."""
    ln_k = float(z1 - z0)
    ln_k_err = float(np.sqrt(e0 ** 2 + e1 ** 2)) if np.isfinite(e0) and np.isfinite(e1) else np.nan
    k = float(np.exp(np.clip(ln_k, -700, 700)))
    if np.isfinite(ln_k_err):
        k_low = float(np.exp(np.clip(ln_k - ln_k_err, -700, 700)))
        k_high = float(np.exp(np.clip(ln_k + ln_k_err, -700, 700)))
    else:
        k_low = np.nan
        k_high = np.nan
    preference = 'M1 body-transit' if ln_k > 0 else 'M0 no-object' if ln_k < 0 else 'neither/tie'
    return {'case': case_label, 'case_role': injected_body.get('case_role', case_label), 'case_description': injected_body.get('case_description', ''), 'injected_radius_re': float(injected_body.get('radius_re', np.nan)), 'injected_a_rprimary': float(injected_body.get('a_rprimary', np.nan)), 'injected_impact_b': float(injected_body.get('impact_b', np.nan)), 'injected_t0_days': float(injected_body.get('t0_days', np.nan)), 'injected_period_days': float(injected_body.get('period_days', np.nan)), 'logz_m0': z0, 'logzerr_m0': e0, 'logz_m1': z1, 'logzerr_m1': e1, 'ln_bayes_factor_m1_over_m0': ln_k, 'ln_bayes_factor_err': ln_k_err, 'bayes_factor_m1_over_m0': k, 'bayes_factor_1sigma_low': k_low, 'bayes_factor_1sigma_high': k_high, 'preferred_model_by_lnK_sign': preference, 'runtime_m0_s': t_m0, 'runtime_m1_s': t_m1}

def _nested_model_setup(cfg: Config, time: np.ndarray, flux: np.ndarray, sigma: np.ndarray) -> Tuple[Any, ...]:
    """Build the shared M0 and M1 transforms and likelihoods for nested validation."""
    t0_half_width = _nested_t0_prior_half_width_days(cfg)
    log_r0 = float(np.log(cfg.body_grid.radius_re_min))
    log_r1 = float(np.log(cfg.body_grid.radius_re_max))
    log_a0 = float(np.log(cfg.body_grid.a_rprimary_min))
    log_a1 = float(np.log(cfg.body_grid.a_rprimary_max))

    def chi_loglike(model):
        """Evaluate the Gaussian log-likelihood for a supplied model light curve."""
        return float(-0.5 * np.sum(((flux - model) / sigma) ** 2 + np.log(2.0 * pi * sigma ** 2)))
    m0_names = ['baseline']

    def m0_transform(cube):
        """Map a unit-cube sample to the M0 baseline prior."""
        return [-0.005 + cube[0] * 0.01]

    def m0_loglike(theta):
        """Evaluate the no-object likelihood for an M0 parameter vector."""
        model = np.ones_like(time, dtype=float) + float(theta[0])
        return chi_loglike(model)
    m1_names = ['baseline', 'body_radius_re', 'body_a_rprimary', 'body_impact_b', 'body_t0_days']

    def m1_transform(cube):
        """Map a unit-cube sample to the M1 transit-model priors."""
        radius_re = float(np.exp(log_r0 + cube[1] * (log_r1 - log_r0)))
        a_rprimary = float(np.exp(log_a0 + cube[2] * (log_a1 - log_a0)))
        rp_over_r = radius_re * REARTH / (cfg.primary.radius_rj * RJUP)
        b_max = float(min(cfg.body_grid.impact_b_max, 1.0 + rp_over_r - 0.0001))
        return [-0.005 + cube[0] * 0.01, radius_re, a_rprimary, cube[3] * max(b_max, 0.0), -t0_half_width + cube[4] * (2.0 * t0_half_width)]

    def theta_to_body(theta):
        """Convert an M1 parameter vector into a batman body dictionary."""
        radius_re = float(theta[1])
        a_rprimary = float(theta[2])
        return {'radius_re': radius_re, 'a_rprimary': a_rprimary, 'period_days': body_period_days(cfg, a_rprimary), 'impact_b': float(theta[3]), 'ecc': 0.0, 'omega_deg': float(cfg.body_grid.omega_deg), 't0_days': float(theta[4])}

    def m1_loglike(theta):
        """Evaluate the batman transit likelihood for an M1 parameter vector."""
        model = batman_flux(time, cfg, theta_to_body(theta)) + float(theta[0])
        return chi_loglike(model)
    return (m0_names, m0_transform, m0_loglike, m1_names, m1_transform, m1_loglike, theta_to_body)

def nested_bayes_factor_report(comparison: pd.DataFrame) -> str:
    """Format the nested evidence comparisons as a plain-text report."""
    lines = ['Nested-sampling Bayesian model comparison', '===========================================', '', 'M0 = no-object flat model plus baseline offset', 'M1 = batman transiting-body model plus baseline offset', 'ln K = ln Z_M1 - ln Z_M0; K = exp(ln K)', '']
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
        lines.append(f"  injected R={row.get('injected_radius_re', np.nan):.4g} Re, a={row.get('injected_a_rprimary', np.nan):.4g} Rprimary, b={row.get('injected_impact_b', np.nan):.4g}")
        lines.append(f"  lnZ(M0)={row.get('logz_m0', np.nan):.4g} ± {row.get('logzerr_m0', np.nan):.3g}; lnZ(M1)={row.get('logz_m1', np.nan):.4g} ± {row.get('logzerr_m1', np.nan):.3g}")
        lines.append(f'  lnK(M1/M0)={ln_k:.4g} ± {ln_k_err:.3g}; K={k:.4g}; sign preference: {pref}')
        lines.append('')
    lines.append('Important: this is a diagnostic model-comparison statistic. The grid recovery and no-object grid remain the empirical false-positive controls.')
    return '\n'.join(lines)

def _run_nested_validation_one_case(cfg: Config, injected_body: Mapping[str, Any], case_label: str, ns_cfg: NestedConfig, output_dir: Path) -> Dict[str, Any]:
    """Fit one simulated validation case with M0 and M1 and save its summaries."""
    rng = np.random.default_rng(cfg.rng_seed + 3000 + sum((ord(ch) for ch in str(case_label))))
    body_for_sim = {k: v for k, v in dict(injected_body).items() if k not in {'case_role', 'case_description'}}
    time, flux, sigma, injected_for_output = simulate_case(cfg, rng, body_for_sim)
    injected_for_output = dict(injected_for_output or {})
    injected_for_output['case_role'] = injected_body.get('case_role', case_label)
    injected_for_output['case_description'] = injected_body.get('case_description', '')
    m0_names, m0_transform, m0_loglike, m1_names, m1_transform, m1_loglike, theta_to_body = _nested_model_setup(cfg, time, flux, sigma)
    case_out = output_dir / str(case_label)
    case_out.mkdir(parents=True, exist_ok=True)
    t0 = _time.time()
    m0_result = _run_sampler(m0_names, m0_loglike, m0_transform, ns_cfg, str(case_out / 'M0'))
    t_m0 = _time.time() - t0
    t0 = _time.time()
    m1_result = _run_sampler(m1_names, m1_loglike, m1_transform, ns_cfg, str(case_out / 'M1'))
    t_m1 = _time.time() - t0
    m0_s = posterior_samples(m0_result)
    m1_s = posterior_samples(m1_result)
    m0_med = np.nanmedian(m0_s, axis=0) if m0_s.size else np.zeros(len(m0_names))
    m1_med = np.nanmedian(m1_s, axis=0) if m1_s.size else np.zeros(len(m1_names))
    m0_model = np.ones_like(time, dtype=float) + float(m0_med[0])
    m1_model = batman_flux(time, cfg, theta_to_body(m1_med)) + float(m1_med[0])
    z0, e0 = _logz(m0_result)
    z1, e1 = _logz(m1_result)
    comparison_row = _bayes_factor_summary_row(case_label, injected_for_output, z0, e0, z1, e1, t_m0, t_m1)
    pd.DataFrame([comparison_row]).to_csv(case_out / 'nested_evidence_comparison.csv', index=False)
    pd.DataFrame([injected_for_output]).to_csv(case_out / 'nested_injected_body.csv', index=False)
    return dict(cfg=cfg, case=case_label, time=time, flux=flux, sigma=sigma, injected_body=injected_for_output, m0_names=m0_names, m1_names=m1_names, m0_result=m0_result, m1_result=m1_result, m0_median=m0_med, m1_median=m1_med, m0_model=m0_model, m1_model=m1_model, comparison=pd.DataFrame([comparison_row]), output_dir=str(case_out))

def run_nested_validation(
    cfg: Config,
    ns_cfg: Optional[NestedConfig] = None,
    case_bodies: Optional[Mapping[str, Mapping[str, Any]]] = None,
    output_dir: str | Path = "result_wd0806_v15_nested",
    progress: bool = True,
) -> Dict[str, Any]:
    """Run diagnostic M0/M1 evidence comparisons for the requested cases."""
    ns_cfg_use = NestedConfig() if ns_cfg is None else ns_cfg
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases_in = nested_validation_case_bodies(cfg) if case_bodies is None else dict(case_bodies)
    if not cases_in:
        raise ValueError("case_bodies must contain at least one validation case.")

    cases: Dict[str, Any] = {}
    comparison_rows: List[Dict[str, Any]] = []
    for case_label, body in cases_in.items():
        case_ns_cfg = ns_cfg_use
        if case_label == "best_case" and ns_cfg_use.best_case_max_ncalls is not None:
            case_ns_cfg = replace(
                ns_cfg_use,
                max_ncalls=int(ns_cfg_use.best_case_max_ncalls),
            )

        if progress:
            cutoff = (
                "none" if case_ns_cfg.max_ncalls is None
                else f"{int(case_ns_cfg.max_ncalls):,} likelihood calls per model"
            )
            print(f"nested validation case: {case_label} (cutoff: {cutoff})")

        one = _run_nested_validation_one_case(
            cfg,
            body,
            str(case_label),
            case_ns_cfg,
            out,
        )
        one["nested_config"] = case_ns_cfg
        cases[str(case_label)] = one
        comparison_rows.append(one["comparison"].iloc[0].to_dict())

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out / "nested_evidence_comparison.csv", index=False)
    report = nested_bayes_factor_report(comparison)
    (out / "nested_bayes_factor_report.txt").write_text(report)

    result = {
        "cfg": cfg,
        "cases": cases,
        "comparison": comparison,
        "report": report,
        "output_dir": str(out),
    }
    default_case = "best_case" if "best_case" in cases else next(iter(cases))
    result.update(cases[default_case])
    return result

def plot_nested_validation(nested: Mapping[str, Any], focus_window_days: Optional[float]=None, case: Optional[str]=None) -> Tuple[Any, Any]:
    """Plot one validation light curve with the median M0 and M1 models."""
    if 'cases' in nested:
        cases = nested['cases']
        case = case or ('best_case' if 'best_case' in cases else next(iter(cases)))
        nested_case = cases[case]
    else:
        nested_case = nested
        case = nested_case.get('case', None)
    cfg = nested_case['cfg']
    t = np.asarray(nested_case['time'])
    f = np.asarray(nested_case['flux'])
    s = np.asarray(nested_case['sigma'])
    body = nested_case.get('injected_body', {})
    period = float(body.get('period_days', body_period_days(cfg, np.sqrt(cfg.body_grid.a_rprimary_min * cfg.body_grid.a_rprimary_max))))
    t0 = float(body.get('t0_days', 0.0))
    centers = t0 + np.arange(int(cfg.n_transits)) * period
    idx = np.argmin(np.abs(t[:, None] - centers[None, :]), axis=1)
    x = t - centers[idx]
    if focus_window_days is None:
        if cfg.local_window_days is not None:
            focus_window_days = cfg.local_window_days
        elif body:
            focus_window_days = max(0.2, 8.0 * body_duration_days(cfg, body.get('radius_re', cfg.body_grid.radius_re_min), body.get('a_rprimary', cfg.body_grid.a_rprimary_min), body.get('impact_b', 0.0)))
        else:
            focus_window_days = max(0.2, 10.0 * cfg.noise.cadence_min / (24.0 * 60.0))
    mask = np.abs(x) <= 0.5 * float(focus_window_days)
    order = np.argsort(x[mask])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x[mask][order], f[mask][order], yerr=s[mask][order], fmt='.', ms=2, alpha=0.45, label='validation data')
    ax.plot(x[mask][order], nested_case['m0_model'][mask][order], lw=2, label='M0 median model')
    ax.plot(x[mask][order], nested_case['m1_model'][mask][order], lw=2, label='M1 median model')
    ax.set_xlabel('Time from nearest injected-body transit [days]', fontsize=14)
    ax.set_ylabel('Relative flux', fontsize=14)
    ax.set_title(f'WD 0806-661 B: nested-sampling validation' + (f' ({case})' if case else ''), fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend()
    fig.tight_layout()
    return (fig, ax)


def plot_nested_corner(
    nested: Mapping[str, Any],
    case: str = "intermediate_case",
    min_spread: float = 1e-12,
) -> Any:
    """Plot only posterior dimensions with enough spread for interpretation."""
    import corner

    if "cases" in nested:
        if case not in nested["cases"]:
            available = ", ".join(nested["cases"].keys())
            raise KeyError(f"Case {case!r} is unavailable. Available cases: {available}")
        nested_case = nested["cases"][case]
    else:
        nested_case = nested
        case = str(nested_case.get("case", case))

    samples = np.asarray(posterior_samples(nested_case["m1_result"]), dtype=float)
    labels = list(nested_case["m1_names"])
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2 or samples.shape[1] != len(labels):
        raise ValueError(
            f"Unexpected posterior sample shape {samples.shape} for {len(labels)} parameters."
        )

    finite_rows = np.all(np.isfinite(samples), axis=1)
    samples = samples[finite_rows]
    spread = np.nanstd(samples, axis=0) if samples.size else np.array([])
    keep = np.isfinite(spread) & (spread > float(min_spread))
    if samples.shape[0] < 20 or keep.sum() < 2:
        raise ValueError(
            f"{case} does not have enough posterior dynamic range for a meaningful "
            f"corner plot. Sample shape={samples.shape}, std={spread}."
        )

    fig = corner.corner(
        samples[:, keep],
        labels=[label for label, ok in zip(labels, keep) if ok],
        show_titles=True,
        title_fmt=".4g",
        label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 15},
    )
    fig.suptitle(f"{case} M1 posterior", fontsize=15)
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=9)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.title.set_size(15)
    fig.tight_layout()
    return fig
