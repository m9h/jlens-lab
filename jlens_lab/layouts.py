"""Architecture layouts, including the hybrids ``jlens`` cannot auto-detect.

``jlens.from_hf`` sniffs six known transformer shapes and raises otherwise:

    ValueError: could not locate the text decoder inside NemotronHForCausalLM
                (tried 6 known layouts); pass layout= explicitly

That is not an incompatibility -- it is a naming mismatch. Nemotron-H calls its
embedding ``embeddings`` and its final norm ``norm_f``; five fields fix it.

This matters more than a papercut, because the workspace phenomenon is clearest in
exactly the models jlens cannot load:

  * Qwen3.5-27B -- 48 of 64 layers are *linear attention*, only 16 are softmax.
    This is the model where the ASCII-face "nose" readout emerges (rank 2, vs 164
    at Qwen3-14B) and where CKA block structure first exceeds a distance-only null.
  * Nemotron-H  -- 21 Mamba-2 mixers, 17 MLPs, 4 attention layers.

If a J-space appears in these, the "global workspace" is not an artifact of softmax
attention -- which is the standing objection to the whole research programme. You
cannot test that with a library that refuses to load them.
"""

from __future__ import annotations

from jlens import Layout

# Registry: HF ``model_type`` -> Layout.
LAYOUTS: dict[str, Layout] = {
    # NVIDIA Nemotron-H (Mamba-2 hybrid). Verified on
    # nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16: children are
    # model.{embeddings, layers, norm_f} + lm_head.
    "nemotron_h": Layout(
        path="model",
        layers="layers",
        norm="norm_f",
        embed="embeddings",
        lm_head="lm_head",
    ),
}


def register(model_type: str, layout: Layout) -> None:
    """Add a layout for an architecture jlens cannot auto-detect."""
    LAYOUTS[model_type] = layout


def layout_for(hf_model) -> Layout | None:
    """Return a known Layout for ``hf_model``, or None to let jlens auto-detect.

    Falls back to the model's ``text_config.model_type`` for multimodal wrappers
    (e.g. Qwen3.5, whose causal-LM class is *ForConditionalGeneration).
    """
    cfg = hf_model.config
    mt = getattr(cfg, "model_type", None)
    if mt in LAYOUTS:
        return LAYOUTS[mt]
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        mt = getattr(text_cfg, "model_type", None)
        if mt in LAYOUTS:
            return LAYOUTS[mt]
    return None


def from_hf(hf_model, tokenizer, **kwargs):
    """``jlens.from_hf`` with the extra layouts wired in.

    Drop-in: uses the registry when it recognises the architecture, otherwise
    defers to jlens's own auto-detection.
    """
    import jlens

    if "layout" not in kwargs:
        lay = layout_for(hf_model)
        if lay is not None:
            kwargs["layout"] = lay
    return jlens.from_hf(hf_model, tokenizer, **kwargs)


def describe(hf_model) -> dict:
    """Summarise a model's block composition -- how much of it is actually attention.

    Returns counts of mixer types, so a hybrid announces itself:

        {'Mamba2Mixer': 21, 'MLP': 17, 'Attention': 4}
    """
    from collections import Counter

    inner = getattr(hf_model, "model", None) or getattr(hf_model, "backbone", None)
    blocks = getattr(inner, "layers", None)
    if blocks is None:
        return {}
    kinds = Counter(
        type(getattr(b, "mixer", b)).__name__ for b in blocks
    )
    return dict(kinds)
