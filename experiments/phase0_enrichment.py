"""Phase 0: discourse-marker enrichment across ungated models, with a permutation null.

Turns the n=1 hand-picked-layer content observation into a band-wide, null-tested,
cross-family result -- or kills it. Caches per model so a restart never re-downloads.
"""
import json, re, pathlib, sys
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
from jlens_lab import artifacts, workspace_content as wc

OUT = pathlib.Path("results/phase0"); OUT.mkdir(parents=True, exist_ok=True)
api = HfApi(); FILES = api.list_repo_files("neuronpedia/jacobian-lens")

UNGATED = ["qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen2.5-7b-it",
           "olmo-3-1025-7b", "pythia-70m-deduped", "gpt2-small"]


def cfg_hf(npid):
    c = [f for f in FILES if f.startswith(npid + "/") and f.endswith("config.yaml")][0]
    t = open(hf_hub_download("neuronpedia/jacobian-lens", c)).read()
    return re.search(r'hf_model_name:\s*"([^"]+)"', t).group(1)


def load_lens(npid):
    p = [f for f in FILES if f.startswith(npid + "/") and f.endswith(".pt")][0]
    return artifacts.load_lens(hf_hub_download("neuronpedia/jacobian-lens", p))


def load_WU(hf):
    try:
        idx = json.load(open(hf_hub_download(hf, "model.safetensors.index.json")))
        wm = idx["weight_map"]
        k = "lm_head.weight" if "lm_head.weight" in wm else "model.embed_tokens.weight"
        with safe_open(hf_hub_download(hf, wm[k]), "pt") as f:
            return f.get_tensor(k).float()
    except Exception:
        with safe_open(hf_hub_download(hf, "model.safetensors"), "pt") as f:
            for k in ("lm_head.weight", "model.embed_tokens.weight",
                      "embed_out.weight", "wte.weight"):
                if k in f.keys():
                    return f.get_tensor(k).float()
            raise KeyError("no unembed tensor found")


for npid in UNGATED:
    cache = OUT / f"{npid}.json"
    if cache.exists():
        print(f"[{npid}] cached: {json.loads(cache.read_text())}", flush=True)
        continue
    try:
        hf = cfg_hf(npid)
        lens = load_lens(npid)
        L = sorted(lens.jacobians); n = len(L)
        if n < 6:
            print(f"[{npid}] too few layers ({n})", flush=True); continue
        ws = L[int(0.35 * n):int(0.75 * n)]          # workspace band = mid 40%
        mo = L[-max(2, n // 10):]                     # motor band = last 10%
        W_U = load_WU(hf)
        tok = AutoTokenizer.from_pretrained(hf)
        r = wc.band_ratio(lens, W_U, workspace_layers=ws, motor_layers=mo)
        e = wc.marker_enrichment(r, tok, top_frac=0.05, n_perm=1000)
        e["hf_model"] = hf; e["n_layers"] = n
        cache.write_text(json.dumps(e, indent=2))
        print(f"[{npid}] enrich={e.get('enrichment', '?'):.2f}x p={e.get('p_value','?')} "
              f"({e.get('observed_in_top')}/{e.get('null_mean'):.1f} markers in top)", flush=True)
        del W_U
    except Exception as ex:
        print(f"[{npid}] FAILED {type(ex).__name__}: {ex}", flush=True)

# summary
rows = {p.stem: json.loads(p.read_text()) for p in OUT.glob("*.json")}
rows = {k: v for k, v in rows.items() if "p_value" in v}
sig = sum(v["p_value"] < 0.05 for v in rows.values())
print(f"\n=== PHASE 0 GATE: {sig}/{len(rows)} models significant (p<0.05) ===")
for k, v in sorted(rows.items()):
    print(f"  {k:20s} enrich={v['enrichment']:.2f}x  p={v['p_value']:.3f}")
print("gate: pass if majority significant -> the workspace-holds-discourse-markers "
      "result is real and worth grounding in data.")
