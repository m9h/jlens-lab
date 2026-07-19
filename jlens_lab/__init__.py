"""jlens-lab -- convergence fitting, architecture layouts, and the controls the
Jacobian lens needs before anyone should believe a figure made with it.

Companion to Anthropic's ``jlens`` (Apache-2.0, "not maintained and not accepting
contributions"). This does not fork it; it depends on it.

    from jlens_lab import from_hf, fit_converged, controls, geometry

    model = from_hf(hf, tok)                      # knows Mamba/SSM hybrids too
    lens, report = fit_converged(model, prompts)  # stops when J stops moving
    assert report.converged                       # under-fitting is silent otherwise
"""

from .fitting import FitReport, fit_converged, wikitext
from .layouts import LAYOUTS, describe, from_hf, layout_for, register
from . import artifacts, controls, geometry

__all__ = [
    "fit_converged", "FitReport", "wikitext",
    "from_hf", "layout_for", "register", "describe", "LAYOUTS",
    "artifacts", "controls", "geometry",
]
__version__ = "0.1.0"
