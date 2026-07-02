"""
Extra target configuration blocks for the v24 exomoon detectability notebooks.

"""


def make_kepler1625_config(exo):
    """Kepler-1625 b config."""
    return exo.Config(
        target="kepler1625",
        star=exo.Star(
            name="Kepler-1625",
            mass_msun=1.079,
            radius_rsun=1.793,
            limb_darkening_u1=0.6196773353931867,
            limb_darkening_u2=-0.15491933384829665,
        ),
        planet=exo.Planet(
            name="Kepler-1625 b",
            period_days=287.38,
            radius_rj=1.04,
            mass_mj=10.0,
            semi_major_au=0.875,
            impact_b=0.04,
        ),
        moon_grid=exo.MoonGrid(
            radius_re_min=0.25,
            radius_re_max=5.0,
            a_rp_min=5.0,
            hill_fraction_max=0.15,
            mutual_inc_deg_max=5.0,
            ecc_min=0.0,
            ecc_max=0.10,
        ),
        noise=exo.telescope_noise_preset(
            instrument="kepler",
            target_mag=15.8,
            cadence_min=29.4,
            red_ppm=50.0,
            red_timescale_hr=6.0,
            duty_cycle=0.92,
        ),
        n_transits=3,
        local_window_days=None,
        rng_seed=1,
        snr_threshold=7.0,
        logk_threshold=exo.log(10.0),
    )


def make_helix_config(exo):
    """Hypothetical Helix Nebula transiting companion config."""
    return exo.Config(
        target="helix",
        star=exo.Star(
            name="WD 2226-210 / Helix",
            mass_msun=0.60,
            radius_rsun=0.020,
            limb_darkening_u1=0.0,
            limb_darkening_u2=0.0,
        ),
        planet=exo.Planet(
            name="Helix Nebula b candidate",
            period_days=2.79,
            radius_rj=0.021 * exo.RSUN / exo.RJUP,
            mass_mj=0.054,
            semi_major_au=6.9 * exo.RSUN / exo.AU,
            impact_b=0.0,
        ),
        moon_grid=exo.MoonGrid(
            radius_re_min=0.05,
            radius_re_max=0.5,
            a_rp_min=3.0,
            hill_fraction_max=0.30,
            mutual_inc_deg_max=3.0,
            ecc_min=0.0,
            ecc_max=0.10,
        ),
        noise=exo.telescope_noise_preset(
            instrument="tess",
            target_mag=13.5,
            cadence_min=2.0,
            red_ppm=250.0,
            red_timescale_hr=3.0,
            duty_cycle=0.90,
        ),
        n_transits=8,
        local_window_days=None,
        rng_seed=1,
        snr_threshold=7.0,
        logk_threshold=exo.log(10.0),
    )


def make_wd1856_config(exo):
    """WD 1856+534 b compact white-dwarf giant-planet stress-test config."""
    return exo.Config(
        target="wd1856",
        star=exo.Star(
            name="WD 1856+534",
            mass_msun=0.518,
            radius_rsun=0.0131,
            limb_darkening_u1=0.0,
            limb_darkening_u2=0.0,
        ),
        planet=exo.Planet(
            name="WD 1856+534 b",
            period_days=1.40794,
            radius_rj=0.92782585,
            mass_mj=13.8,
            semi_major_au=0.0204,
            impact_b=0.35,
        ),
        moon_grid=exo.MoonGrid(
            radius_re_min=0.05,
            radius_re_max=0.5,
            a_rp_min=3.0,
            hill_fraction_max=0.30,
            mutual_inc_deg_max=2.0,
            ecc_min=0.0,
            ecc_max=0.02,
        ),
        noise=exo.telescope_noise_preset(
            instrument="tess",
            target_mag=15.0,
            cadence_min=2.0,
            red_ppm=250.0,
            red_timescale_hr=3.0,
            duty_cycle=0.90,
        ),
        n_transits=8,
        local_window_days=0.35,
        rng_seed=1,
        snr_threshold=7.0,
        logk_threshold=exo.log(10.0),
    )

### Ideal HZ exomoon config ###

# Nauenberg-derived WD radius for M_WD = 0.60 Msun
R_WD_RSUN = 0.01245

# Keplerian orbital period for:
# a = 0.0358 AU, M_WD = 0.60 Msun, M_p = 1.00 Mjup
P_PLANET_DAYS = 3.1915

# Whole-grid trigger thresholds
SNR_THRESHOLD = 7.0
LOGK_THRESHOLD = log(10.0)

truth_backend = "auto"

cfg = exo.Config(
    target="ideal_wd",

    star=exo.Star(
        name="Ideal HZ WD",
        mass_msun=0.60,
        radius_rsun=R_WD_RSUN,
        limb_darkening_u1=0.4,
        limb_darkening_u2=0.25,
    ),

    planet=exo.Planet(
        name="Ideal HZ giant planet",
        period_days=P_PLANET_DAYS,
        radius_rj=1.00,
        mass_mj=1.00,
        semi_major_au=0.0358,
        t0_days=0.0,
        impact_b=0.0,
        ecc=0.0,
        omega_deg=90.0,
    ),

    moon_grid=exo.MoonGrid(
        radius_re_min=0.05,
        radius_re_max=1.00,

        # For this system:
        # R_Hill / R_p ≈ 6.06.
        # With hill_fraction_max=0.49, the upper search bound is ≈ 2.97 R_p.
        # Therefore a_rp_min must be below this.
        a_rp_min=1.20,
        hill_fraction_max=0.30,

        mutual_inc_deg_max=0.0,
        ecc_min=0.0,
        ecc_max=0.0,
        force_coplanar=True,
    ),

    noise=exo.Noise(
        instrument="ideal",
        cadence_min=2.0,
        white_ppm=300.0,
        red_ppm=100.0,
        red_timescale_hr=3.0,
        duty_cycle=1.0,
    ),

    n_transits=10,
    local_window_days=None,
    rng_seed=34,
    snr_threshold=SNR_THRESHOLD,
    logk_threshold=LOGK_THRESHOLD,
)


# Replace the inputs and some new target can be examined.
