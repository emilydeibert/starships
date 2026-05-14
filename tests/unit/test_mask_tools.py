import numpy as np
import pytest
from starships.mask_tools import split_mask, where_not_masked, interp1d_masked


class TestWhereNotMasked:

    def test_three_segments(self):
        """Trois segments non-masqués dans un tableau avec deux gaps."""
        mask = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0])
        i1, i2 = where_not_masked(mask)
        np.testing.assert_array_equal(i1, [0, 4, 7])
        np.testing.assert_array_equal(i2, [2, 6, 9])

    def test_no_mask(self):
        """Sans masque, un seul segment couvrant tout le tableau."""
        mask = np.zeros(8, dtype=int)
        i1, i2 = where_not_masked(mask)
        assert len(i1) == 1
        assert i1[0] == 0
        assert i2[0] == 8

    def test_fully_masked(self):
        """Un tableau entièrement masqué ne doit produire aucun segment."""
        mask = np.ones(6, dtype=int)
        i1, i2 = where_not_masked(mask)
        assert len(i1) == 0

    def test_single_unmasked_value(self):
        """Un seul pixel non-masqué = un segment de longueur 1."""
        mask = np.array([1, 1, 0, 1, 1])
        i1, i2 = where_not_masked(mask)
        assert len(i1) == 1
        assert i2[0] - i1[0] == 1


class TestSplitMask:

    def test_three_segments_values(self):
        """split_mask doit retourner les valeurs correctes pour trois segments."""
        data = np.ma.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            mask=[0,   0,   1,   1,   0,   0,   1,   0,   0]
        )
        segments = split_mask(data)
        assert len(segments) == 3
        np.testing.assert_array_equal(segments[0], [1.0, 2.0])
        np.testing.assert_array_equal(segments[1], [5.0, 6.0])
        np.testing.assert_array_equal(segments[2], [8.0, 9.0])

    def test_external_mask(self):
        """split_mask doit accepter un masque passé séparément."""
        x    = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mask = np.array([0,   0,   1,   0,   0])
        y    = np.ma.array(np.sin(x), mask=mask)
        segs_x = split_mask(x, mask)
        segs_y = split_mask(y)
        assert len(segs_x) == 2
        assert len(segs_y) == 2

    def test_no_mask_returns_full_array(self):
        """Sans masque, split_mask doit retourner le tableau complet en un segment."""
        data = np.ma.array([1.0, 2.0, 3.0, 4.0])
        segments = split_mask(data)
        assert len(segments) == 1
        np.testing.assert_array_equal(segments[0], [1.0, 2.0, 3.0, 4.0])


class TestInterp1dMasked:

    def test_gap_returns_masked(self):
        """Les points dans le gap masqué doivent être masqués en sortie."""
        x = np.arange(10.0)
        y = np.ma.array(np.sin(x), mask=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        f = interp1d_masked(x, y, kind='linear')
        result = f(np.array([3.5, 4.0, 4.5]))
        assert result.mask.all(), "Les points dans le gap doivent être masqués"

    def test_outside_gap_interpolates_correctly(self):
        """En dehors du gap, l'interpolation doit être correcte."""
        x      = np.linspace(0, 2 * np.pi, 50)
        y_true = np.sin(x)
        mask   = np.zeros(50, dtype=int)
        mask[20:30] = 1
        y = np.ma.array(y_true, mask=mask)
        f = interp1d_masked(x, y, kind='cubic')

        x_test = np.array([0.5, 1.0, 5.0])
        result = f(x_test)
        expected = np.sin(x_test)
        np.testing.assert_allclose(result.data[~result.mask], expected, atol=0.01)

    def test_unmasked_input_behaves_like_standard_interp1d(self):
        """Sans tableau masqué, interp1d_masked doit se comporter comme interp1d standard."""
        from scipy.interpolate import interp1d
        x = np.linspace(0, 5, 20)
        y = np.cos(x)
        x_new = np.array([1.1, 2.3, 3.7])

        f_standard = interp1d(x, y, kind='linear')
        f_masked   = interp1d_masked(x, y, kind='linear')

        np.testing.assert_allclose(f_standard(x_new), f_masked(x_new), rtol=1e-10)

    def test_two_chunks_no_overlap(self):
        """Deux segments non-masqués doivent produire des interpolants disjoints sans conflit."""
        x    = np.arange(12.0)
        mask = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        y    = np.ma.array(x ** 2, mask=mask)
        f    = interp1d_masked(x, y, kind='linear')

        # Premier segment
        r1 = f(np.array([1.0, 2.5]))
        assert not r1.mask.any()

        # Second segment
        r2 = f(np.array([9.0, 10.5]))
        assert not r2.mask.any()

        # Gap
        r_gap = f(np.array([5.0, 6.0]))
        assert r_gap.mask.all()
