import numpy as np
import pytest
from starships.correlation import calc_logl_BL_ord


class TestCalcLogLBLOrd:

    def test_known_value(self):
        """Valeur exacte vérifiable à la main.

        flux=[1,2,3], model=[1.5,1.5,1.5], alpha=1, N=3
        R    = 1*1.5 + 2*1.5 + 3*1.5 = 9.0
        s2f  = 1 + 4 + 9             = 14.0
        s2g  = 2.25 * 3              = 6.75
        chi2 = 14 - 2*9 + 6.75      = 2.75
        logL = -3/2 * log(2.75/3)
        """
        flux  = np.array([1.0, 2.0, 3.0])
        model = np.array([1.5, 1.5, 1.5])
        N     = 3
        expected = -N / 2 * np.log(2.75 / N)
        result   = calc_logl_BL_ord(flux, model, N, alpha=1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_nolog_returns_chi2(self):
        """nolog=True doit retourner chi2 directement, pas la logL."""
        flux  = np.array([1.0, 2.0, 3.0])
        model = np.array([1.5, 1.5, 1.5])
        N     = 3
        chi2 = calc_logl_BL_ord(flux, model, N, alpha=1.0, nolog=True)
        np.testing.assert_allclose(chi2, 2.75, rtol=1e-12)

    def test_alpha_optimum_maximizes_logl(self):
        """La logL est maximale à alpha_opt = R/s2g (Brogi & Line eq. 6).

        Pour des données bruitées, le alpha qui maximise la logL est celui
        qui minimise le chi2, soit alpha_opt = sum(f*g) / sum(g²).
        """
        rng   = np.random.default_rng(42)
        flux  = rng.normal(0, 1, 300)
        model = rng.normal(0, 0.7, 300)
        N     = 300
        R     = np.sum(flux * model)
        s2g   = np.sum(model ** 2)
        alpha_opt = R / s2g

        logl_opt = calc_logl_BL_ord(flux, model, N, alpha=alpha_opt)
        for delta in [-0.5, -0.2, 0.2, 0.5]:
            logl_other = calc_logl_BL_ord(flux, model, N, alpha=alpha_opt + delta)
            assert logl_opt > logl_other, (
                f"logL non-maximale à alpha_opt (delta={delta:+.1f}): "
                f"{logl_opt:.4f} <= {logl_other:.4f}"
            )

    def test_consistency_with_chi2map_formulation(self):
        """calc_logl_BL_ord doit donner le même résultat que la formulation get_logl de chi2_map.

        Dans chi2_map.py, get_logl() reçoit des termes pré-calculés :
          f_x_g = sum(f * g)   (cross_terms)
          s2g   = sum(g²)      (squared_terms)
          s2f   = sum(f²)
          N
        La logL BL est : -N/2 * log(chi2/N) où chi2 = s2f - 2α*f_x_g + α²*s2g
        """
        rng   = np.random.default_rng(0)
        flux  = rng.normal(0, 1, 50)
        model = rng.normal(0, 0.8, 50)
        N     = 50
        alpha = 1.0

        logl_correlation = calc_logl_BL_ord(flux, model, N, alpha=alpha)

        f_x_g = np.sum(flux * model)
        s2g   = np.sum(model ** 2)
        s2f   = np.sum(flux ** 2)
        chi2  = s2f - 2 * alpha * f_x_g + alpha ** 2 * s2g
        logl_chi2map = -N / 2 * np.log(chi2 / N)

        np.testing.assert_allclose(logl_correlation, logl_chi2map, rtol=1e-12)

    def test_output_shape_2d(self):
        """Pour un tableau (n_spec, n_pix), la sortie doit avoir la forme (n_spec,)."""
        rng   = np.random.default_rng(1)
        flux  = rng.normal(0, 1, (7, 100))
        model = rng.normal(0, 0.5, (7, 100))
        N     = np.full(7, 100)
        result = calc_logl_BL_ord(flux, model, N, axis=-1)
        assert result.shape == (7,)

    def test_s2f_precomputed_vs_computed(self):
        """Passer s2f pré-calculé doit donner le même résultat que le calcul interne."""
        rng   = np.random.default_rng(5)
        flux  = rng.normal(0, 1, 80)
        model = rng.normal(0, 1, 80)
        N     = 80
        s2f_precomputed = np.sum(flux ** 2)
        logl_auto = calc_logl_BL_ord(flux, model, N)
        logl_pre  = calc_logl_BL_ord(flux, model, N, s2f=s2f_precomputed)
        np.testing.assert_allclose(logl_auto, logl_pre, rtol=1e-12)
