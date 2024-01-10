r"""
This file demonstrates the result of different choice in order of R(otation), T(ranslation) and  D(ilation) transforms.
Traditionally CWT is built with similitude group, that is group operation g is g = T D R combining with norming which
leads to wavelets a^{-1} s(a^-1 R^{-1} (\vec r - \vec r_s)), where a is scale for dilatation.
Elementary transforms act on the vector from L_2 as follows
translation T: s(r) -> s(r-r_s)
dilation   D: s(r) -> a^{-1} s(r/a)
rotation s(r) -> s(R^{-1} r), R \in SO(2).
(TDR)^{-1} = R^{-1} D^{-1} T{-1}
This file builds contour plots for the following variants of all 6 possible combinations for usual and Lorentz rotation
in space and Fourier domains.
"""


import plotly.graph_objects as go
import pyapcwt.transforms2d as t2d
import pyapcwt.wavelets as wav
import numpy as np
from plotly.subplots import make_subplots


def main():
    dilation: t2d.ITransform2D = t2d.UniformScale2D(2)
    rotation: t2d.ITransform2D = t2d.Rotation2D(np.pi / 6)
    translation: t2d.ITransform2D = t2d.Translation2D(np.array([3,0]).reshape((2,1)))

    transforms = [
        ("TDR", translation * dilation * rotation),
        ("RDT", rotation * translation * dilation),
        ("DRT", dilation * rotation * translation),
        ("TRD", translation * rotation * dilation),
        ("DTR", dilation * translation * rotation),
        ("RTD", rotation * translation * dilation),
    ]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=tuple([name for name, _ in transforms]))

    sz = 1;
    xs = np.linspace(-10 * sz, 10 * sz, 201)
    ys = np.linspace(-10 * sz, 10 * sz, 201)

    xv, yv = np.meshgrid(xs, ys)

    vec = np.stack([xv.flatten(), yv.flatten()])

    paramMorlet = wav.MorletParam(sigma_par=1.25, sigma_perp=.8, k0=5)
    psi = lambda v: np.real(wav.morlet2d(paramMorlet, v))

    values0 = np.reshape(psi(vec), xv.shape)

    for idx,(transform_name, transform) in enumerate(transforms):
        act: t2d.ITransform2D = transform.inv()
        vec_t = act.apply(vec)
        values = np.reshape(np.real(psi(vec_t)), xv.shape)

        fig.add_trace(go.Contour(
            x=xs,
            y=ys,
            z=values,
        ), row= idx // 3 + 1, col=idx%3 +1  )


    fig.show(renderer="firefox")


if __name__ == "__main__":
    main()
