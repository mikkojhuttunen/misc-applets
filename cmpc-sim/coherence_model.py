"""coherence_model — interference (speckle / etalon) noise of a multipass cell from its discrete paths.
Thin re-export of math-engines/engines/path_coherence; see that module for the model and its assumptions."""
import _engines  # noqa: F401
from engines.path_coherence.engine import (C, Paths, autocovariance, contrast, correlation_width, prepare,  # noqa: F401
                                           simulate_spectrum, speckle_noise_A, thermal_shift_hz)
