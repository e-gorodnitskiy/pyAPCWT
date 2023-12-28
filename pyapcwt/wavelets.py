""" Module contains definitions for some mother wavelets.
It is started with just simple functions and highly likely it will be transformed that mother wavelet should define
some interface.(TBD)
"""
import numpy as np

from typing import NamedTuple
from pyapcwt.types import *


class MexicanHatParam(NamedTuple):
    sigma: float


def mexican_hat2d(vec2d: NumpyArray, param: MexicanHatParam) -> NumpyArray:
    """
    Implements real 2d Mexican hat(Riecker) wavelet normed in the L2
    :param vec2d: array of 2d points shaped as (2,n), where n is point numbes
    :param param: parameters of the wavelet. Strictly speaking sigma is excessive as soon as most variant of the
    wavelet transform uses scales, but sometimes it might be convenient.
    :return: array of shape (n,) with wavelet values. (
    """
    assert vec2d.ndim == 2
    assert vec2d.shape[0] == 2
    assert param.sigma > 0

    r2_over_sigma2 = np.sum(np.square(vec2d), axis=0) / param.sigma ** 2
    ps = np.multiply(1.0 - r2_over_sigma2, np.exp(- 0.5 * r2_over_sigma2)) / (param.sigma * np.sqrt(np.pi))
    return np.squeeze(ps)


if __name__ == "__main__":
    import plotly.graph_objects as go


def main():
    param = MexicanHatParam(2)
    v0 = mexican_hat2d(np.array([0, 0]).reshape(2, 1), param)
    print(v0)
    nv = mexican_hat2d(np.array([[0, 0], [1, 1], [2, 1]]).reshape(2, -1), param)
    print(nv)

    xs = np.linspace(-10 * param.sigma, 10 * param.sigma, 201)
    ys = np.linspace(-10 * param.sigma, 10 * param.sigma, 201)

    xv, yv = np.meshgrid(xs, ys)

    vec = np.stack([xv.flatten(), yv.flatten()])
    values = mexican_hat2d(vec, param)
    norm = np.sum(np.square(values)) * (param.sigma / 10) ** 2
    print(f"Norm is {norm:6f}")
    values = np.reshape(values, xv.shape)

    fig = go.Figure(go.Surface(
        contours={
            "x": {"show": True, "start": xs[0], "end": xs[-1], "size": param.sigma / 2, "color": "white"},
            "y": {"show": True, "start": ys[0], "end": ys[-1], "size": param.sigma / 2, "color": "white"},
            "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
        },
        x=xs,
        y=ys,
        z=values
    ))

    fig.update_layout(
        scene={
            "xaxis": {"nticks": 20},
            "zaxis": {"nticks": 4},
            'camera_eye': {"x": 0, "y": -1, "z": 0.5},
            "aspectratio": {"x": 1, "y": 1, "z": 0.2}
        })
    fig.show()


if __name__ == "__main__":
    main()
