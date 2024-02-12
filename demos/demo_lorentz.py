r"""
This file demonstrates the result of different choice in order of L(orentz) transform, T(ranslation) and  D(ilation) transforms.
Traditionally CWT is built like similitude group, that is group operation g is g = T D L combining with norming which
leads to wavelets a^{-1} s(a^-1 L^{-1} (\vec \chi - \vec \chi_s)), where a is scale for dilatation.
Elementary transforms act on the vector from L_2 as follows
translation T: s(\vec \chi) -> s(\chi-\chi_s)
dilation   D: s(\chi) -> a^{-1} s(\chi/a)
Lorentz s(t, x) -> s(L^{-1} \chi), R \in P(2).
(TDL)^{-1} = L^{-1} D^{-1} T{-1}
This file builds contour plots for the following variants of all 6 possible combinations for Lorentz rotation
in space and Fourier domains.
"""

import plotly.graph_objects as go
import pyapcwt.transforms2d as t2d
import pyapcwt.wavelets as wav
import numpy as np
from plotly.subplots import make_subplots


def demo_lorentz_with_morlet() -> None:
    dilation: t2d.ITransform2D = t2d.UniformScale2D(2)
    lorentz: t2d.ITransform2D = t2d.Lorentz2D(0.71)
    translation: t2d.ITransform2D = t2d.Translation2D(np.array([5, 0]).reshape((2, 1)))

    transforms = [
        ("TDL", translation * dilation * lorentz),
        ("LDT", lorentz * translation * dilation),
        ("DLT", dilation * lorentz * translation),
        ("TLD", translation * lorentz * dilation),
        ("DTL", dilation * translation * lorentz),
        ("LTD", lorentz * translation * dilation),
    ]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=tuple([name for name, _ in transforms]))

    d = 1
    tmin = 0
    xmin = -5
    sz = 20
    ts = np.linspace(tmin, tmin + sz * d, 201)
    xs = np.linspace(xmin, xmin + sz * d, 201)

    xv, tv = np.meshgrid(xs, ts)

    vec = np.stack([tv.flatten(), xv.flatten()])

    paramMorlet = wav.MorletParam(sigma_par=0.5, sigma_perp=2, k0=5)
    chi_0 = np.array([2, 0]).reshape((2, 1))
    psi = lambda v: np.abs(wav.morlet2d(paramMorlet, v - chi_0))

    values0 = np.reshape(psi(vec), tv.shape)
    # fig.update_layout(autosize=False, height=600, width=600)

    for idx, (transform_name, transform) in enumerate(transforms):
        act: t2d.ITransform2D = transform.inv()
        vec_t = act.apply(vec)
        values = np.reshape(psi(vec_t), xv.shape) + np.reshape(psi(vec), xv.shape)
        layout_kwargs = {f"yaxis{idx + 1}_scaleanchor": f"x{idx + 1}"}
        fig.update_layout(**layout_kwargs)

        cnt = go.Contour(

            x=xs,
            y=ts,
            z=values,
        )

        fig.add_trace(cnt, row=idx // 3 + 1, col=idx % 3 + 1)

    fig.show(renderer="firefox")


def main():
    demo_lorentz_with_morlet()


if __name__ == "__main__":
    main()
