"""
Types definition and checking utils for the APCWT
"""
import numpy as np
import numpy.typing as npt
from typing import Callable, Any

NumpyArray = npt.NDArray[np.float64 | np.float32]


# todo: think if this one is suitable plase
def check_dimension_vec2d(f: Callable[[Any, NumpyArray], Any]):
    def checked(obj: Any, vec2d: NumpyArray):
        assert vec2d.ndim == 2
        assert vec2d.shape[0] == 2

        return f(obj, vec2d)

    return checked
