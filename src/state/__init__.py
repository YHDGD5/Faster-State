"""State: perturbation-response transition models.

Importing the package installs the data-pipeline fast paths in :mod:`state.tx.data.fast_fetch` (currently a
bitwise-identical fast densify of a cell's raw gene-count row). It is done here, at package import, because
that is the earliest point common to both `state tx train` and the numerics adapter, and because the data
loader workers are forked later and must inherit the patched method.
"""
from .tx.data import fast_fetch as _fast_fetch  # noqa: F401  (installs the patch on import)
