from abc import ABC, abstractmethod
from typing import List, Callable, Any

import numpy as np
import numpy.typing as npt

NumpyArray = npt.NDArray[np.float64 | np.float32]


# todo: move to some designated utils module
def check_dimension_vec2d(f: Callable[[Any, NumpyArray], Any]):
    def checked(obj: Any, vec2d: NumpyArray):
        assert vec2d.ndim == 2
        assert vec2d.shape[0] == 2

        return f(obj, vec2d)

    return checked


class ITransform2D(ABC):

    def __call__(self, vec2d: NumpyArray) -> NumpyArray:
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

    def __str__(self):
        return str([str(t) for t in self.__transforms])

    @check_dimension_vec2d
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

    @check_dimension_vec2d
    def apply(self, vec2d: np.ndarray) -> np.ndarray:
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


def make_lorentz_rotation_matrix2d(rapidity: float) -> NumpyArray:
    return np.array(
        [[np.cosh(rapidity), np.sinh(rapidity)],
         [np.sinh(rapidity), np.cosh(rapidity)]]
    )


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

    @check_dimension_vec2d
    def apply(self, vec2d: NumpyArray) -> NumpyArray:
        return vec2d + self.__translation


class UniformScale2D(ITransform2D):
    def __init__(self, scale: float):
        super().__init__()
        self.__scale = scale

    def __mul__(self, other) -> ITransform2D:
        if isinstance(other, UniformScale2D):
            return UniformScale2D(self.__scale * other.__scale)
        else:
            return super().__mul__(other)

    def inv(self) -> ITransform2D:
        return UniformScale2D(1.0 / self.__scale)

    @check_dimension_vec2d
    def apply(self, vec2d: NumpyArray) -> NumpyArray:
        return vec2d * self.__scale


class Lorentz2D(ITransform2D):
    def __init__(self, rapidity: float):
        self.__rapidity = rapidity
        self.__matrix = make_lorentz_rotation_matrix2d(rapidity)

    def __mul__(self, other: ITransform2D) -> ITransform2D:
        if isinstance(other, Lorentz2D):
            return Lorentz2D(self.__rapidity + other.__rapidity)
        else:
            return super().__mul__(other)

    def inv(self) -> ITransform2D:
        return Lorentz2D(-self.__rapidity)

    @check_dimension_vec2d
    def apply(self, vec2d: NumpyArray) -> NumpyArray:
        return np.matmul(self.__matrix, vec2d)


class DummyTransform2D(ITransform2D):
    def __init__(self, identity: int):
        super().__init__()
        self.__identity = identity

    def __str__(self):
        return str(self.__identity)

    @check_dimension_vec2d
    def apply(self, vec2d: np):
        return self.__identity * vec2d

    def inv(self):
        return DummyTransform2D(-self.__identity)


def main():
    v0 = DummyTransform2D(1)
    v1 = DummyTransform2D(2)
    v2 = DummyTransform2D(3)
    v = v0 * v1
    print(v0 * v1 * v2)
    print((v0 * v1 * v2).inv().inv())
    print((v0 * v1 * v2).inv())
    print(v2 * v1 * v0)
    print((v2 * v1 * v0).inv())

    # vec2d = np.array([[1, 2, 3], [4, 5, 6]])
    vec2d1 = np.array([[1, 4], [2, 5], [3, 6]])
    # res = v.apply(vec2d1.transpose())
    res = v.apply(vec2d1.transpose())
    print(res)

    m = np.array([[1, 2], [0, 3]])

    vm = np.matmul(m, vec2d1.transpose())  # .transpose()
    print(vm)
    print(m)
    print(vm[:, 2])

    print(make_rotation_matrix2d(np.pi / 4))


if __name__ == "__main__":
    main()
