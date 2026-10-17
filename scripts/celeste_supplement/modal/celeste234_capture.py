# ruff: noqa
"""Celeste supplement tasks 2-4: block-42 output of the frozen BASE Qwen3.6-27B at every cell (same as capture_wsb_release.py: one forward per
item, cells gathered from it; cache format identical so celeste234_vllm_readout.py reads it), PLUS the logit lens at the same cells
(final RMSNorm + lm_head on the raw block-42 residual, top-10 vocab tokens with logits) written in the readout contract as a token arm.
    python celeste234_capture.py --cells /vol_c/tasks234/cells --out /vol_c/tasks234/evalcache --logit-lens-out /vol_c/tasks234/results/logit_lens"""

import argparse, glob, json, os, time
import torch
from common import Layer42Hook, load_base, backbone, lm_head_weight, load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="/vol_c/tasks234/cells")
ap.add_argument("--out", default="/vol_c/tasks234/evalcache")
ap.add_argument("--logit-lens-out", default="/vol_c/tasks234/results/logit_lens")
ap.add_argument("--families", default=None)
ap.add_argument("--topk", type=int, default=10)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
os.makedirs(a.logit_lens_out, exist_ok=True)
t0 = time.time()
dev = "cuda"
model = load_base(dev)
hook = Layer42Hook(model)
tok = load_tokenizer()
norm = backbone(model).norm
W = lm_head_weight(model)


def fwd(ids):
    x = torch.tensor([ids], device=dev)
    with torch.no_grad():
        hook.capture()
        model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
        H = hook.captured[0]
        hook.off()
    return H


def logit_lens(h):  # h [n, d] raw residual after block 42
    with torch.no_grad():
        z = norm(h.to(W.dtype))
        logits = z @ W.T
        top = torch.topk(logits.float(), a.topk, dim=-1)
    return top.indices.cpu().tolist(), top.values.cpu().tolist()


files = sorted(glob.glob(f"{a.cells}/*.jsonl"))
fams = set(a.families.split(",")) if a.families else None
for f in files:
    fam = os.path.basename(f)[:-6]
    if fams and fam not in fams:
        continue
    if os.path.exists(f"{a.out}/{fam}.pt"):
        print(f"[c234-cap] {fam}: exists, skip", flush=True)
        continue
    ids_, pos_, tok_, acts = [], [], [], []
    t1 = time.time()
    ll = open(f"{a.logit_lens_out}/{fam}.jsonl", "w")
    for line in open(f):
        r = json.loads(line)
        if not r["cells"]:
            continue
        H = fwd(r["ids"])
        idx = torch.tensor([c["abs"] for c in r["cells"]], device=dev)
        hc = H[idx]
        acts.append(hc.to(torch.float16).cpu())
        ti, tv = logit_lens(hc)
        for c, ii, vv in zip(r["cells"], ti, tv):
            ids_.append(r["id"])
            pos_.append(c["pos"])
            tok_.append(c["token"])
            ll.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "layer": 42,
                        "pos": int(c["pos"]),
                        "token": c["token"],
                        "tokens": tok.convert_ids_to_tokens(ii),
                        "scores": [round(v, 3) for v in vv],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    ll.close()
    A = torch.cat(acts) if acts else torch.zeros(0, 5120, dtype=torch.float16)
    torch.save({"id": ids_, "pos": pos_, "token": tok_, "acts": A}, f"{a.out}/{fam}.pt.tmp")
    os.replace(f"{a.out}/{fam}.pt.tmp", f"{a.out}/{fam}.pt")
    print(
        f"[c234-cap] {fam}: {len(ids_)} cells ({(time.time() - t1) / 60:.1f} min, total {(time.time() - t0) / 60:.1f} min)",
        flush=True,
    )
json.dump(
    {
        "created": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "elapsed_min": (time.time() - t0) / 60,
    },
    open(f"{a.out}/CAPTURE_DONE.json", "w"),
)
print(f"C234_CAPTURE_DONE {(time.time() - t0) / 60:.1f} min", flush=True)
