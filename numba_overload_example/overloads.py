import numpy as np
from numba import types
from numba.errors import TypingError
from numba.extending import overload


@overload(np.clip)
def impl_clip(x, a, b):
    # In numba type checking happens at *compile time*. We check the types of
    # the arguments here, and return a proper implementation based on those
    # types (or error accordingly).

    # Check that `a` and `b` are scalars, and at most one of them is None.
    if not isinstance(a, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("a must be a scalar int/float")
    if not isinstance(b, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("b must be a scalar int/float")
    if isinstance(a, types.NoneType) and isinstance(b, types.NoneType):
        raise TypingError("a and b can't both be None")

    if isinstance(x, (types.Integer, types.Float)):
        # x is a scalar with a valid type
        if isinstance(a, types.NoneType):
            # a is None
            def impl(x, a, b):
                return min(x, b)
        elif isinstance(b, types.NoneType):
            # b is None
            def impl(x, a, b):
                return max(x, a)
        else:
            # neither a or b are None
            def impl(x, a, b):
                return min(max(x, a), b)
    elif (
        isinstance(x, types.Array) and
        x.ndim == 1 and
        isinstance(x.dtype, (types.Integer, types.Float))
    ):
        # x is a 1D array of the proper type
        def impl(x, a, b):
            # Allocate an output array using standard numpy functions
            out = np.empty_like(x)
            # Iterate over x, calling `np.clip` on every element
            for i in range(x.size):
                # This will dispatch to the proper scalar implementation (as
                # defined above) at *compile time*. There should have no
                # overhead at runtime.
                out[i] = np.clip(x[i], a, b)
            return out
    else:
        raise TypingError("x must be an int/float or a 1D array of ints/floats")

    # The call to `np.clip` has arguments with valid types, return our
    # numba-compatible implementation
    return impl
