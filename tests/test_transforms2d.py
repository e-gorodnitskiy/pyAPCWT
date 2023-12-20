import unittest
import numpy as np

import pyapcwt.transforms2d as transforms2d


class TestTranslation2d(unittest.TestCase):
    def test_neutral_translation(self):
        vec2d = np.array([1, 4])
        zero_translation = transforms2d.Translation2D(np.array([0, 0]))
        neutral_result = zero_translation(vec2d)
        self.assertTrue((vec2d.reshape(2, -1) == neutral_result).all())

    def test_compose(self):
        vec2d = np.array([1.5, -0.625]).reshape(2, -1)

        t1 = transforms2d.Translation2D(np.array([1, 1]))
        t2 = transforms2d.Translation2D(np.array([-1, -1]))

        t = t1 * t2
        neutral_result = t(vec2d)
        self.assertTrue((vec2d == neutral_result).all())

        t2 = transforms2d.Translation2D(np.array([-1.25, 1.625]))

        vl = (t1 * t2).apply(vec2d)
        vr = (t2 * t1).apply(vec2d)
        v_expected = np.array([1.25, 2]).reshape(2, -1)

        self.assertTrue((vl == vr).all())
        self.assertTrue((v_expected == vl).all())

    def test_inv(self):
        vecs2d = np.array([[2.5, -1.125], [1.125, 2.5, ]])

        t = transforms2d.Translation2D(np.array([-2, 4]))
        t_inv = t.inv()

        i2t = (t_inv * t).apply(vecs2d)
        t2i = (t * t_inv).apply(vecs2d)
        self.assertTrue((i2t == t2i).all())
        self.assertTrue((i2t == vecs2d).all())

    def test_apply(self):
        vec2d = np.array([2.5, -1.125]).reshape((2,1))
        vec2d_vectorized = np.array([[2.5, -1.125], [1.125, 2], [-5, 2]])

        t = transforms2d.Translation2D(np.array([2.5, -1.125]))

        vec2d_expected = np.array([5, -2.25]).reshape((2, 1))
        vec2d_vectorized_expected = np.array([[5, -2.25], [3.625, 0.875], [-2.5, 0.875]]).transpose()

        vec2d_res = t.apply(vec2d)
        vec2d_vectorized_res = t.apply(vec2d_vectorized.transpose())

        self.assertTrue((vec2d_expected == vec2d_res).all())
        self.assertTrue((vec2d_vectorized_expected == vec2d_vectorized_res).all())


if __name__ == '__main__':
    unittest.main()
