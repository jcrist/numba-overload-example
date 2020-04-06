from setuptools import setup

setup(
    name="numba_overload_example",
    version="0.1.0",
    license="BSD",
    description="Example of extending Numba using numba.extending.overload",
    packages=["numba_overload_example"],
    install_requires=["numba"],
    # Register our numba extensions.
    # Numba will automatically load our module and run the `init` function,
    # registering any overloads we defined in this module.
    entry_points={
        "numba_extensions": [
            "init = numba_overload_example:init",
        ]
    },
)
