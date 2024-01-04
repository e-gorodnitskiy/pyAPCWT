""" Module contains definitions for some mother wavelets.
It is started with just simple functions and highly likely it will be transformed that mother wavelet should define
some interface.(TBD)
"""
import numpy
import numpy as np

from typing import NamedTuple
from pyapcwt.types import *


class MexicanHatParam(NamedTuple):
    sigma: float


def mexican_hat2d(param: MexicanHatParam, vec2d: NumpyArray) -> NumpyArray:
    """
    Implements real 2d Mexican hat(Riecker) wavelet normed in the L2
    :param vec2d: array of 2d points shaped as (2,n), where n is point number
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


class MorletParam(NamedTuple):
    """
    The 2d morlet wavelet parameters -- simplified. The correct formula for the wavelet is
    \frac{ \sqrt{ 2\sqrt {det A}} }{ \sqrt{\pi } } \exp ( i \vec {k_0} \vec r}) \exp( -0.5 (A\vec{r}, \vec{r})
    where vector k_0 defines direction of propagation or is orthogonal to oscillation and A defines the matrix of the
    quadratic form. Below one defines k_0 = (k0, 0) and
    A = ( 1/sigma_par^2 , 0
          0            1/sigma_perp^2)
    If sometimes one will need more sofisticated parameter set as cross-term in A and non-zero second component of the k_0\
    one have to redefine this struct and make appropriate changes in the Morlet wavelets code.
    """
    sigma_par: float
    sigma_perp: float
    k0: float


def morlet2d(param: MorletParam, vec2d: NumpyArray) -> NumpyArray:
    """
    Implements Morlet wavelet w/o correction term to satisfy admissibility condition. From practicl point of view
    in case of sigmas ~ 1 and k0>=5 is wavelet is admissible (at least for rotation, shift, scale group).
    :param param: MorletParam
    :param vec2d: coordinates (x,y), size must be (2, n) 
    :return: array of the shape (n,) of the complex wavelet values  
    """
    assert vec2d.ndim == 2
    assert vec2d.shape[0] == 2

    mx = np.array([
        [0.5 / (param.sigma_par ** 2), 0],
        [0, 0.5 / (param.sigma_perp ** 2)]])

    k_0 = np.array([param.k0, 0]).reshape((2, 1))

    norm_factor = np.sqrt(2 * np.sqrt(np.linalg.det(mx))) / (np.sqrt(np.pi))

    phase_value = np.matmul(k_0.transpose(), vec2d).squeeze()
    form_value = np.sum(numpy.multiply(numpy.matmul(mx, vec2d), vec2d), axis=0)

    ps = np.exp(1j * phase_value - form_value) * norm_factor

    return np.squeeze(ps)


if __name__ == "__main__":
    import plotly.graph_objects as go


def main():
    param = MexicanHatParam(2)
    v0 = mexican_hat2d(param, np.array([0, 0]).reshape(2, 1))
    print(v0)
    nv = mexican_hat2d(param, np.array([[0, 0], [1, 1], [2, 1]]).reshape(2, -1))
    print(nv)

    xs = np.linspace(-10 * param.sigma, 10 * param.sigma, 201)
    ys = np.linspace(-10 * param.sigma, 10 * param.sigma, 201)

    xv, yv = np.meshgrid(xs, ys)

    vec = np.stack([xv.flatten(), yv.flatten()])
    values = mexican_hat2d(param, vec)
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
    fig.show(renderer="firefox")

    paramMorlet = MorletParam(sigma_par=1.25, sigma_perp=.8, k0=5)
    nv = morlet2d(paramMorlet, np.array([[0, 0], [1, 1], [2, 1]]).transpose())
    print(nv)

    morlet_values = morlet2d(paramMorlet, vec)
    morlet_norm = np.sum(morlet_values * np.conjugate(morlet_values)) * (param.sigma / 10) ** 2
    print(f"Morlet norm is {morlet_norm:6f}")
    zv = np.real(morlet_values)
    zv = np.reshape(zv, xv.shape)

    fig2 = go.Figure(go.Surface(
        contours={
            "x": {"show": True, "start": xs[0], "end": xs[-1], "size": param.sigma / 2, "color": "white"},
            "y": {"show": True, "start": ys[0], "end": ys[-1], "size": param.sigma / 2, "color": "white"},
            "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
        },
        x=xs,
        y=ys,
        z=zv
    ))

    fig2.update_layout(
        scene={
            "xaxis": {"nticks": 20},
            "zaxis": {"nticks": 4},
            'camera_eye': {"x": 0, "y": -1, "z": 0.5},
            "aspectratio": {"x": 1, "y": 1, "z": 0.2}
        })
    fig2.show(renderer="firefox")


if __name__ == "__main__":
    main()
