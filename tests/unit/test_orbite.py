import numpy as np
import pytest
import astropy.units as u
import astropy.constants as const
from starships.orbite import rv_theo_t, Kp_theo, t2trueanom, trueanom2t


class TestRvTheoT:

    def test_zero_at_conjunction(self):
        """La RV est nulle à t = t_ref (conjonction, orbite circulaire e=0)."""
        K_star = 100.0
        t_ref  = 0.0 * u.day
        P      = 1.5 * u.day
        rv = rv_theo_t(K_star, t_ref, t_ref, P, plnt=True)
        np.testing.assert_allclose(rv.to(u.km / u.s).value, 0.0, atol=1e-12)

    def test_max_rv_at_quarter_period(self):
        """La RV de la planète est maximale (= Kp) à t_ref + P/4 (orbite circulaire)."""
        K_star = 100.0
        P      = 2.0 * u.day
        t_ref  = 0.0 * u.day
        t      = t_ref + P / 4
        rv = rv_theo_t(K_star, t, t_ref, P, plnt=True)
        np.testing.assert_allclose(rv.to(u.km / u.s).value, K_star, rtol=1e-10)

    def test_star_planet_opposite_sign(self):
        """L'étoile et la planète ont des RV opposées (conservation de la quantité de mouvement)."""
        K_star = 80.0
        P      = 3.0 * u.day
        t_ref  = 0.0 * u.day
        t      = t_ref + 0.3 * u.day
        rv_star   = rv_theo_t(K_star, t, t_ref, P, plnt=False)
        rv_planet = rv_theo_t(K_star, t, t_ref, P, plnt=True)
        np.testing.assert_allclose(
            rv_star.to(u.km / u.s).value,
            -rv_planet.to(u.km / u.s).value,
            rtol=1e-10
        )

    def test_antisymmetry_around_t_ref(self):
        """rv(t_ref + dt) = -rv(t_ref - dt) pour orbite circulaire."""
        K_star = 150.0
        P      = 2.5 * u.day
        t_ref  = 0.0 * u.day
        dt     = 0.3 * u.day
        rv_plus  = rv_theo_t(K_star, t_ref + dt, t_ref, P, plnt=True)
        rv_minus = rv_theo_t(K_star, t_ref - dt, t_ref, P, plnt=True)
        np.testing.assert_allclose(
            rv_plus.to(u.km / u.s).value,
            -rv_minus.to(u.km / u.s).value,
            rtol=1e-10
        )

    def test_periodicity(self):
        """rv(t) = rv(t + P) pour toute valeur de t."""
        K_star = 120.0
        P      = 1.8 * u.day
        t_ref  = 0.0 * u.day
        t      = 0.7 * u.day
        rv1 = rv_theo_t(K_star, t,     t_ref, P, plnt=True)
        rv2 = rv_theo_t(K_star, t + P, t_ref, P, plnt=True)
        np.testing.assert_allclose(
            rv1.to(u.km / u.s).value,
            rv2.to(u.km / u.s).value,
            rtol=1e-10
        )


class TestKpTheo:

    def test_momentum_conservation(self):
        """M_star * K_star = M_pl * Kp (conservation de la quantité de mouvement)."""
        K_star = 0.1 * u.km / u.s
        M_star = const.M_sun
        M_pl   = const.M_jup
        Kp = Kp_theo(K_star, M_star, M_pl)
        np.testing.assert_allclose(
            (M_star * K_star).decompose().value,
            (M_pl   * Kp).decompose().value,
            rtol=1e-10
        )

    def test_kp_much_larger_than_kstar(self):
        """Kp >> K_star pour M_star >> M_pl (cas typique exoplanète)."""
        K_star = 0.1 * u.km / u.s
        Kp = Kp_theo(K_star, const.M_sun, const.M_jup)
        ratio = Kp.to(u.km / u.s).value / K_star.value
        assert ratio > 100, f"Ratio Kp/K_star trop faible : {ratio:.1f}"

    def test_scales_linearly_with_kstar(self):
        """Kp doit être proportionnelle à K_star."""
        M_star = const.M_sun
        M_pl   = const.M_jup
        K1 = 0.05 * u.km / u.s
        K2 = 0.10 * u.km / u.s
        Kp1 = Kp_theo(K1, M_star, M_pl)
        Kp2 = Kp_theo(K2, M_star, M_pl)
        np.testing.assert_allclose(Kp2.value / Kp1.value, 2.0, rtol=1e-10)


class TestT2TrueAnom:

    def test_roundtrip(self):
        """t2trueanom(trueanom2t(nu)) == nu pour une orbite circulaire.

        trueanom2t retourne np.array(t) qui perd les unités ; les valeurs numériques
        sont dans la même unité de temps que P (jours ici), donc on réattache u.day.
        """
        P  = 2.5 * u.day
        nu = np.array([0.1, 0.5, 1.0, 2.0, 3.0])  # floats en radians, pas Quantity
        t  = trueanom2t(P, nu) * u.day
        nu_recovered = t2trueanom(P, t, t0=0 * u.day)
        np.testing.assert_allclose(nu_recovered, nu, atol=1e-6)

    def test_nu_zero_at_t_ref(self):
        """L'anomalie vraie est 0 à t = t0 (périastre) pour orbite circulaire."""
        P  = 1.0 * u.day
        t0 = 0.0 * u.s
        nu = t2trueanom(P, t0, t0=t0)
        np.testing.assert_allclose(nu, 0.0, atol=1e-10)
