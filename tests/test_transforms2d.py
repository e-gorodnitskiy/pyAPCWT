import unittest
import numpy as np

import pyapcwt.transforms2d as transforms2d

EPSILON_FOR_EXACT_FP64 = 10 * np.finfo(np.float64).eps


class TestTranslation2d(unittest.TestCase):
    def test_neutral(self):
        vec2d = np.array([1, 4]).reshape((2, 1))
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
        vec2d = np.array([2.5, -1.125]).reshape((2, 1))
        vec2d_vectorized = np.array([[2.5, -1.125], [1.125, 2], [-5, 2]])

        t = transforms2d.Translation2D(np.array([2.5, -1.125]))

        vec2d_expected = np.array([5, -2.25]).reshape((2, 1))
        vec2d_vectorized_expected = np.array([[5, -2.25], [3.625, 0.875], [-2.5, 0.875]]).transpose()

        vec2d_res = t.apply(vec2d)
        vec2d_vectorized_res = t.apply(vec2d_vectorized.transpose())

        self.assertTrue((vec2d_expected == vec2d_res).all())
        self.assertTrue((vec2d_vectorized_expected == vec2d_vectorized_res).all())


class TestRotation2D(unittest.TestCase):
    def test_neutral(self):
        vec2d = np.array([1, 4]).reshape((2, 1))
        zero_rotation = transforms2d.Rotation2D(0)
        neutral_result = zero_rotation(vec2d)
        self.assertTrue((vec2d == neutral_result).all())

        zero_rotation = transforms2d.Rotation2D(np.pi * 4)
        neutral_result = zero_rotation(vec2d)
        self.assertTrue(np.allclose(vec2d, neutral_result, EPSILON_FOR_EXACT_FP64, EPSILON_FOR_EXACT_FP64))

    def test_compose(self):
        vec2d = np.array([1, 1]).reshape(2, -1)

        t1 = transforms2d.Rotation2D(np.pi / 3)
        t2 = transforms2d.Rotation2D(-np.pi / 3)

        t = t1 * t2
        neutral_result = t(vec2d)
        self.assertTrue((vec2d == neutral_result).all())

        t2 = transforms2d.Rotation2D(2 * np.pi / 3)

        vl = (t1 * t2).apply(vec2d)
        vr = (t2 * t1).apply(vec2d)

        v_expected = np.array([-1, -1]).reshape(2, -1)

        self.assertTrue((vl == vr).all())
        self.assertTrue(np.allclose(v_expected, vl, EPSILON_FOR_EXACT_FP64, EPSILON_FOR_EXACT_FP64))

    def test_inv(self):
        vecs2d = np.array([[2.5, -1.125], [1.125, 2.5, ]])

        t = transforms2d.Rotation2D(0.3351)
        t_inv = t.inv()

        i2t = (t_inv * t).apply(vecs2d)
        t2i = (t * t_inv).apply(vecs2d)
        self.assertTrue((i2t == t2i).all())
        self.assertTrue(np.allclose(i2t, vecs2d, EPSILON_FOR_EXACT_FP64, EPSILON_FOR_EXACT_FP64))

    def test_apply(self):
        vec2d = np.array([2.5, -1.125]).reshape((2, 1))
        vec2d_vectorized = np.array([[2.5, -1.125], [1.125, 2.5], [-5, 2]]).transpose()

        angle = 3.5
        r = transforms2d.Rotation2D(angle)

        vec2d_res = r.apply(vec2d)
        vec2d_vectorized_res = r.apply(vec2d_vectorized)

        angle_res = np.arccos(np.vdot(vec2d / np.linalg.norm(vec2d), vec2d_res / np.linalg.norm(vec2d_res)))
        cross_z = vec2d[0] * vec2d_res[1] - vec2d[1] * vec2d_res[0]
        # 3.5 > pi
        self.assertTrue(cross_z < 0)
        self.assertTrue(np.allclose(np.linalg.norm(vec2d), np.linalg.norm(vec2d_res), EPSILON_FOR_EXACT_FP64,
                                    EPSILON_FOR_EXACT_FP64))
        # 3.5 > pi => angle_res should be taken with
        self.assertTrue(np.allclose(angle_res, 2 * np.pi - angle, EPSILON_FOR_EXACT_FP64, EPSILON_FOR_EXACT_FP64))

        self.assertTrue(np.allclose(vec2d_res, vec2d_res[:, 0:1], EPSILON_FOR_EXACT_FP64, EPSILON_FOR_EXACT_FP64))
        # keep orthogonality
        self.assertTrue(np.allclose(
            0,
            np.vdot(vec2d_vectorized_res[:, 0:1], vec2d_vectorized_res[:, 1:2]),
            EPSILON_FOR_EXACT_FP64,
            EPSILON_FOR_EXACT_FP64))

        # we compare 1-st vectors, orthogonality of fist & second and I believe it is enough to compare the whole norm
        self.assertTrue(np.allclose(
            np.linalg.norm(vec2d_vectorized),
            np.linalg.norm(vec2d_vectorized_res),
            EPSILON_FOR_EXACT_FP64,
            EPSILON_FOR_EXACT_FP64))


class TestUniformScale2d(unittest.TestCase):
    def test_neutral(self):
        vec2d = np.array([1, 4]).reshape((2, 1))
        unit_scale = transforms2d.UniformScale2D(1)
        neutral_result = unit_scale(vec2d)
        self.assertTrue((vec2d.reshape(2, -1) == neutral_result).all())

    def test_compose(self):
        vec2d = np.array([1.5, -0.625]).reshape(2, -1)

        t1 = transforms2d.UniformScale2D(2.5)
        t2 = transforms2d.UniformScale2D(0.4)

        t = t1 * t2
        neutral_result = t(vec2d)
        self.assertTrue((vec2d == neutral_result).all())

        t2 = transforms2d.UniformScale2D(0.2)

        vl = (t1 * t2).apply(vec2d)
        vr = (t2 * t1).apply(vec2d)
        v_expected = np.array([0.75, -0.3125]).reshape(2, -1)

        self.assertTrue((vl == vr).all())
        self.assertTrue((v_expected == vl).all())

    def test_inv(self):
        vecs2d = np.array([[2.5, -1.125], [1.125, 2.5, ]])

        t = transforms2d.UniformScale2D(4)
        t_inv = t.inv()

        i2t = (t_inv * t).apply(vecs2d)
        t2i = (t * t_inv).apply(vecs2d)
        self.assertTrue((i2t == t2i).all())
        self.assertTrue((i2t == vecs2d).all())

    def test_apply(self):
        vec2d = np.array([2.5, -1.125]).reshape((2, 1))
        vec2d_vectorized = np.array([[2.5, -1.125], [1.125, 2], [-5, 2]])

        t = transforms2d.UniformScale2D(2)

        vec2d_expected = np.array([5, -2.25]).reshape((2, 1))
        vec2d_vectorized_expected = np.array([[5, -2.25], [2.25, 4], [-10, 4]]).transpose()

        vec2d_res = t.apply(vec2d)
        vec2d_vectorized_res = t.apply(vec2d_vectorized.transpose())

        self.assertTrue((vec2d_expected == vec2d_res).all())
        self.assertTrue((vec2d_vectorized_expected == vec2d_vectorized_res).all())


class TestLorentz2d(unittest.TestCase):
    def test_neutral(self):
        vec2d = np.array([1, 4]).reshape((2, 1))
        neutral_rotation = transforms2d.Lorentz2D(0)
        neutral_result = neutral_rotation(vec2d)
        self.assertTrue((vec2d == neutral_result).all())

    def test_compose(self):
        vec2d = np.array([1.5, -0.625]).reshape(2, -1)

        t1 = transforms2d.Lorentz2D(2.5)
        t2 = transforms2d.Lorentz2D(-2.5)

        t = t1 * t2
        neutral_result = t(vec2d)
        self.assertTrue((vec2d == neutral_result).all())

        t2 = transforms2d.Lorentz2D(1)

        vl = (t1 * t2).apply(vec2d)
        vr = (t2 * t1).apply(vec2d)

        self.assertTrue((vl == vr).all())

    def test_inv(self):
        # pseudoscalar dot product is t1 * t2 - x1 * x2
        vecs2d = np.array([[2.5, 1.125], [1.125, 2.5, ]])

        t = transforms2d.Lorentz2D(0.5)
        t_inv = t.inv()

        i2t = (t_inv * t).apply(vecs2d)
        t2i = (t * t_inv).apply(vecs2d)
        self.assertTrue((i2t == t2i).all())
        self.assertTrue((i2t == vecs2d).all())

    def test_apply(self):
        vec2d_vectorized = np.array([
            [2.5, 1.125],  # D1
            [-2.5, -1.125],  # D2
            [1.125, 2.5],  # D3
            [-1.125, -2.5]  # D4
        ]).transpose()

        t = transforms2d.Lorentz2D(-1)

        expected_intervals2 = np.square(vec2d_vectorized[0, :]) - np.square(vec2d_vectorized[1, :])

        vec2d_vectorized_res = t.apply(vec2d_vectorized)
        intervals2 = np.square(vec2d_vectorized_res[0, :]) - np.square(vec2d_vectorized_res[1, :])

        self.assertTrue(np.allclose(expected_intervals2, intervals2, EPSILON_FOR_EXACT_FP64, EPSILON_FOR_EXACT_FP64))

        # check for orthogonality & dot product conservation
        for i in range(0, 4):
            for j in range(0, 4):
                v0 = vec2d_vectorized[:, i:i + 1]
                v1 = vec2d_vectorized[:, j:j + 1]
                product_expected = v0[0] * v1[0] - v0[1] * v1[1]

                v0t = vec2d_vectorized_res[:, i:i + 1]
                v1t = vec2d_vectorized_res[:, j:j + 1]

                product_transformed = v0t[0] * v1t[0] - v0t[1] * v1t[1]
                self.assertTrue(np.allclose(product_expected, product_transformed, EPSILON_FOR_EXACT_FP64, EPSILON_FOR_EXACT_FP64))


if __name__ == '__main__':
    unittest.main()
