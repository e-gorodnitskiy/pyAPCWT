from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.typing as npt

NumpyArray = npt.NDArray[np.float64 | np.float32]


class ITransform2D(ABC):

    def __call__(self, vec2d: NumpyArray) -> NumpyArray:
        assert vec2d.ndim == 2
        assert vec2d.shape[0] == 2

        return self.apply(vec2d.reshape((2, -1)))

    def __mul__(self, other):
        assert isinstance(other, ITransform2D)
        return BasicChainedTransform2D([self, other])

    @abstractmethod
    def apply(self, vec2d: NumpyArray) -> NumpyArray:
        raise NotImplementedError

    @abstractmethod
    def inv(self):
        raise NotImplementedError


class BasicChainedTransform2D(ITransform2D):
    def __init__(self, transforms: List[ITransform2D]):
        super().__init__()
        self.__transforms = transforms

    def apply(self, vec2d: np.ndarray) -> np.ndarray:
        res = vec2d.copy()
        for transform in self.__transforms:
            res = transform.apply(res)
        return res

    def inv(self):
        inv_transforms = []

        for transform in self.__transforms:
            inv_transforms.append(transform.inv())

        inv_transforms.reverse()

        return BasicChainedTransform2D(inv_transforms)


class LinearTransform2D(ITransform2D):
    def __init__(self, matrix: NumpyArray, translation: NumpyArray):
        """
        :param matrix 3x3 in projective format
         R T  here R - 2x2 rotation matrix, T - translation vector
         0 1 to simple handle rotation/translation transforms.
        in case one needs general 2d transformation just use  M 0
                                                              0 1
        :param matrix:
        """
        assert matrix.shape == (2, 2)
        assert translation.size == 2

        super().__init__()

        self.__translation = np.atleast_2d(translation).reshape((2, 1))
        self.__matrix2d = matrix

    def apply(self, vec2d: np.ndarray) -> np.ndarray:
        assert vec2d.shape[0] == 2
        return np.matmul(self.__matrix2d, vec2d.reshape((2, -1))) + self.__translation

    def inv(self) -> ITransform2D:
        m = np.eye(3)
        m[0:2, 0:2] = self.__matrix2d[0:2, 0:2]
        # use 2: instead of 2 to save last dimension of m[] (2,1) instead of (2,)
        m[0:2, 2:] = self.__translation

        minv = np.linalg.inv(m)
        return LinearTransform2D(minv[0:2, 0:2], minv[0:2, 2])

    def __mul__(self, other) -> ITransform2D:
        if isinstance(other, LinearTransform2D):
            return LinearTransform2D(
                np.matmul(self.__matrix2d, other.__matrix2d),
                np.matmul(self.__matrix2d, other.__translation) + self.__translation)
        else:
            return super().__mul__(other)


def make_rotation_matrix2d(angle: float) -> NumpyArray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


class Rotation2D(ITransform2D):
    def __init__(self, angle: float):
        self.__angle = angle
        self.__matrix = make_rotation_matrix2d(angle)

    def __mul__(self, other) -> ITransform2D:
        if isinstance(other, Rotation2D):
            return Rotation2D(np.fmod(self.__angle + other.__angle, 2 * np.pi))
        else:
            return super().__mul__(other)

    def inv(self):
        return Rotation2D(-self.__angle)

    def apply(self, vec2d: NumpyArray) -> NumpyArray:
        return np.matmul(self.__matrix, vec2d)


class Translation2D(ITransform2D):
    def __init__(self, translation: NumpyArray):
        assert translation.size == 2
        super().__init__()
        self.__translation = np.atleast_2d(translation).reshape((2, 1))

    def __mul__(self, other) -> ITransform2D:
        if isinstance(other, Translation2D):
            return Translation2D(self.__translation + other.__translation)
        else:
            return super().__mul__(other)

    def inv(self) -> ITransform2D:
        return Translation2D(-self.__translation)

    def apply(self, vec2d: NumpyArray) -> NumpyArray:
        assert vec2d.shape[0] == 2
        assert vec2d.ndim == 2
        return vec2d + self.__translation


class DummyTransform2D(ITransform2D):
    def __init__(self, identity: int):
        super().__init__()
        self.__identity = identity

    def apply(self, vec2d: np):
        return self.__identity * vec2d

    def inv(self):
        return DummyTransform2D(-self.__identity)


def main():
    v0 = DummyTransform2D(3)
    v1 = DummyTransform2D(2)
    v = v0 * v1
    # vec2d = np.array([[1, 2, 3], [4, 5, 6]])
    vec2d1 = np.array([[1, 4], [2, 5], [3, 6]])
    res = v.apply(vec2d1)
    print(res)

    m = np.array([[1, 2], [0, 3]])

    vm = np.matmul(m, vec2d1.transpose())  # .transpose()
    print(vm)
    print(m)
    print(vm[:, 2])

    print(make_rotation_matrix2d(np.pi / 4))


if __name__ == "__main__":
    main()
