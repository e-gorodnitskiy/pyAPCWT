r"""
This file demonstrates the result of different choice in order of R(otation), T(ranslation) and  D(ilation) transforms.
Traditionally CWT is built with similitude group, that is group operation g is g = T D R combining with norming which
leads to wavelets a^{-1} s(a^-1 R^{-1} (\vec r - \vec r_s)), where a is scale for dilatation.
Elementary transforms act on the vector from L_2 as follows
translation T: s(r) -> s(r-r_s)
dilation   D: s(r) -> a^{-1} s(r/a)
rotation s(r) -> s(R^{-1} r), R \in SO(2).
(TDR)^{-1} = R^{-1} D^{-1} T{-1}
This file builds contour plots for the following variants of all 6 possible combinations for usual rotation
in space and Fourier domains.
"""

import plotly.graph_objects as go
import pyapcwt.transforms2d as t2d
import pyapcwt.wavelets as wav
import numpy as np
from plotly.subplots import make_subplots


def demo_similitude_with_morlet() -> None:
    dilation: t2d.ITransform2D = t2d.UniformScale2D(2)
    rotation: t2d.ITransform2D = t2d.Rotation2D(np.pi / 3)
    translation: t2d.ITransform2D = t2d.Translation2D(np.array([5, 0]).reshape((2, 1)))

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

    d = 1
    xmin = -5
    ymin = -5
    sz = 20
    xs = np.linspace(xmin, xmin + sz * d, 201)
    ys = np.linspace(ymin, ymin + sz * d, 201)

    xv, yv = np.meshgrid(xs, ys)

    vec = np.stack([xv.flatten(), yv.flatten()])

    paramMorlet = wav.MorletParam(sigma_par=0.5, sigma_perp=2, k0=5)
    psi = lambda v: np.abs(wav.morlet2d(paramMorlet, v))

    values0 = np.reshape(psi(vec), xv.shape)
    # fig.update_layout(autosize=False, height=600, width=600)

    for idx, (transform_name, transform) in enumerate(transforms):
        act: t2d.ITransform2D = transform.inv()
        vec_t = act.apply(vec)
        values = np.reshape(psi(vec_t), xv.shape) + np.reshape(psi(vec), xv.shape)
        layout_kwargs = {f"yaxis{idx + 1}_scaleanchor": f"x{idx + 1}"}
        fig.update_layout(**layout_kwargs)

        cnt = go.Contour(

            x=xs,
            y=ys,
            z=values,
        )

        fig.add_trace(cnt, row=idx // 3 + 1, col=idx % 3 + 1)

    fig.show(renderer="firefox")


def main():
    demo_similitude_with_morlet()


if __name__ == "__main__":
    main()
