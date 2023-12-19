import unittest
import numpy as np

import pyapcwt.transforms2d as transforms2d


class TestTranslation2d(unittest.TestCase):
    def test_neutral_translation(self):
        vec2d = np.array([1,4])
        zero_translation = transforms2d.Translation2D(np.array([0,0]))
        neutral_result =  zero_translation(vec2d)
        self.assertTrue( (vec2d.reshape(2,-1) == neutral_result).all())

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



if __name__ == '__main__':
    unittest.main()