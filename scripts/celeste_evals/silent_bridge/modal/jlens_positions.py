"""J-lens diagnostic on Modal (HF, stock Qwen3.6-27B): full-vocab rank of each labelled intermediate
(neuronpedia n=1000, cosine readout) at the last K positions, layers 36-48, plus the top-10 at L42.
    D=/vol_c/celeste_evals/silent; python silent_jlens.py --pool $D/pool.json --out $D/jlens.json"""

import argparse
import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "0"
import huggingface_hub.constants as _hfc

_hfc.HF_HUB_OFFLINE = False
import torch  # noqa: E402  # must follow the HF offline toggle above
from huggingface_hub import hf_hub_download  # noqa: E402  # same: download needs HF online

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="36,38,40,42,44,46,48")
ap.add_argument("--format", default="raw", choices=["raw", "chat"])
ap.add_argument(
    "--last-k", type=int, default=1, help="read the last K positions (ranks per position)"
)
ap.add_argument("--repo", default="neuronpedia/jacobian-lens")
ap.add_argument(
    "--filename", default="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
)
a = ap.parse_args()
t0 = time.time()
dev = "cuda"
LAYERS = [int(x) for x in a.layers.split(",")]
obj = torch.load(
    hf_hub_download(a.repo, a.filename, cache_dir="/tmp/hf_dl"),
    map_location="cpu",
    weights_only=False,
)
jac = obj["J"] if "J" in obj else obj["jacobians"]
J = {
    L: ((jac[L] if L in jac else jac[str(L)]) if isinstance(jac, dict) else jac[L]).float().to(dev)
    for L in LAYERS
}
del obj, jac
_hfc.HF_HUB_OFFLINE = True
os.environ["HF_HUB_OFFLINE"] = "1"
from common import backbone, load_base, load_tokenizer  # noqa: E402  # after HF goes offline again

tok = load_tokenizer()
model = load_base(dev)
W_U = model.get_output_embeddings().weight.float()
blocks = backbone(model).layers
denom = {}
with torch.no_grad():
    for L in LAYERS:
        d = torch.empty(W_U.shape[0], device=dev)
        for i in range(0, W_U.shape[0], 16384):
            d[i : i + 16384] = (W_U[i : i + 16384] @ J[L]).norm(dim=1)
        denom[L] = d.clamp_min(1e-9)
store = {}


def mk(layer):
    def f(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        store[layer] = h[0, -a.last_k :].float()

    return f


for L in LAYERS:
    blocks[L].register_forward_hook(mk(L))


def ids_of(word):
    out = set()
    for v in (" " + word, word, " " + word.capitalize(), word.capitalize(), " " + word.lower()):
        t = tok(v, add_special_tokens=False)["input_ids"]
        if t:
            out.add(t[0])
    return sorted(out)


CHAT = "Complete this sentence with only the missing word or words, nothing else.\n\n{body} ..."


def render(prompt, user=None):
    if user is not None:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    if a.format == "raw":
        return tok(prompt, add_special_tokens=False)["input_ids"]
    body = prompt[len("Fact: ") :] if prompt.startswith("Fact: ") else prompt
    ids = tok.apply_chat_template(
        [{"role": "user", "content": CHAT.format(body=body)}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


with open(a.pool) as fh:
    pool = json.load(fh)
res = []
with torch.no_grad():
    for p in pool:
        ids = render(p["prompt"], p.get("user"))
        model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1)
        K = min(a.last_k, len(ids))
        pos0 = len(ids) - K
        r = {
            "name": p["name"],
            "n_tokens": len(ids),
            "positions": list(range(pos0, len(ids))),
            "tokens": [tok.decode([i]) for i in ids[pos0:]],
            "ranks": {},
            "top10_L42": [],
        }
        for L in LAYERS:
            S = ((store[L][-K:] @ J[L].T) @ W_U.T) / denom[L]  # [K, V]
            for w in p["intermediates"]:
                t = ids_of(w)
                sc = S[:, t].max(dim=1).values  # [K]
                rk = (sc[:, None] < S).sum(dim=1).tolist()
                r["ranks"].setdefault(w, {})[str(L)] = [int(x) for x in rk]
            if L == 42:
                r["top10_L42"] = [
                    [tok.decode([i]) for i in row]
                    for row in torch.topk(S, 10, dim=1).indices.tolist()
                ]
        res.append(r)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as fh:
    json.dump({"layers": LAYERS, "items": res}, fh, ensure_ascii=False)
print(f"[silent-jlens] {len(res)} prompts | {(time.time() - t0) / 60:.1f} min", flush=True)
