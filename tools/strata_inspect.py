"""Which Puddle model a GGUF is - and whether Puddle can run it - from its header alone.

Reads only the files' headers (a few MB each, by HTTP range requests for a remote file) with quantscope, prints the
real storage of every weight group (bits per weight, quantization types), then compares the routed experts and the
sizes with the files setup installs (data/gguf_fingerprints.json):

    python tools/strata_inspect.py hf:unsloth/Laguna-S-2.1-GGUF                           # list the variants
    python tools/strata_inspect.py hf:unsloth/Laguna-S-2.1-GGUF UD-Q3_K_XL                # one variant
    python tools/strata_inspect.py D:/models/my-copy-00001-of-00003.gguf                  # a local file or folder
    python tools/strata_inspect.py https://example.com/x.gguf                             # one URL

hf: is Hugging Face (HF_ENDPOINT is honoured), ms: ModelScope.  setup.py --inspect runs the same.
    python tools/strata_inspect.py --fingerprints data/gguf_fingerprints.json             # remake the table
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import quantscope as qs  # noqa: E402

FINGERPRINTS = ROOT / "data" / "gguf_fingerprints.json"
# the architectures strata-poly runs for Puddle's families -> FAMILIES[family]["title"] in setup.py (resolve_custom_gguf
# maps a "family" verdict back to --family / --model by that title).  Laguna-S and Laguna-XS share one architecture:
# their layer count tells them apart, as it tells Nemotron 3.5 Super (88 layers and its MTP block) from Bolt (88).
POLY_ARCHS = {"laguna": {48: "Laguna S 2.1", 40: "Laguna XS 2.1"}, "nemotron_h_moe": {89: "Nemotron 3.5 Super"},
              "gemma4": "Gemma 4 26B-A4B"}
# the architectures of setup's llama-server families ("engine": "llama-server"): setup installs their own file only.
# A file of one of them that a family above has (Super) is that family's.
LLAMA_ARCHS = {"nemotron_h_moe": "Nemotron 3.5 Bolt"}


def known_sentence(exact: list) -> str:
    """The verdict for a file setup installs: each family/size that installs it (Bolt's file: on llama-server, and on
    strata-poly as nemotron-bolt-poly)."""
    first, *more = [k.split("/") for k in exact]
    return (f"this is setup's --family {first[0]} --model {first[1]}"
            + "".join(f" (or --family {f} --model {m})" for f, m in more) + " (give its folder with --gguf-dir)")


def family_title(fp: dict) -> str:
    """The title of the family a GGUF's architecture (and layer count) belongs to, "" for none."""
    t = POLY_ARCHS.get(fp.get("arch"), "")
    return t.get(fp.get("layers"), "") if isinstance(t, dict) else t


def known_files():
    """The files setup installs: (family, model, [Source of each shard at its pinned revision]) - for --fingerprints."""
    sys.path.insert(0, str(ROOT))
    import setup as S
    for fam, d in S.FAMILIES.items():
        for model, size in d["models"].items():
            base = size.get("hf", d["hf"])
            sha = {**d["sha256"], **size.get("sha256", {})}
            names = [S.model_file(d, model, i) for i in range(1, S.model_shards(d, model) + 1)]
            yield fam, model, [qs.Source(n, sha[n.split("/")[-1]][0], url=base + n) for n in names]


def read_headers(files: list) -> tuple[str | None, list[dict]]:
    """The architecture and every tensor of a model's shards (headers only)."""
    arch, tensors = None, []
    for f in files:
        meta, ts = qs.read_gguf(f)
        arch = arch or meta.get("general.architecture")
        tensors += ts
    return arch, tensors


def layers(tensors: list[dict]) -> int | None:
    """How many blocks the tensors are of (blk.0 ... blk.N-1; a draft block after them counts too)."""
    ids = {int(m.group(1)) for t in tensors if (m := re.match(r"^blk\.(\d+)\.", t["name"]))}
    return max(ids) + 1 if ids else None


def role(name: str) -> str:
    """A tensor's name without its layer number: blk.7.ffn_down_exps.weight -> blk.N.ffn_down_exps.weight."""
    return re.sub(r"^blk\.\d+\.", "blk.N.", name)


def roles(tensors: list[dict]) -> dict[str, list[str]]:
    out: dict[str, set] = defaultdict(set)
    for t in tensors:
        out[role(t["name"])].add(t["type"])
    return {k: sorted(v) for k, v in sorted(out.items())}


def fingerprint(arch: str | None, tensors: list[dict]) -> dict:
    """What the comparison needs from a model's shards: architecture, tensor count, bytes, routed experts' bytes per
    quantization type and their bits per weight."""
    n, total_b, total_e, exp_e = len(tensors), 0, 0, 0
    experts: dict[str, int] = defaultdict(int)
    for t in tensors:
        total_b += t["bytes"]
        total_e += t["elements"]
        if qs.category(t["name"]) == "routed experts":
            experts[t["type"]] += t["bytes"]
            exp_e += t["elements"]
    return {"arch": arch, "layers": layers(tensors), "tensors": n, "bytes": total_b,
            "expert_bytes": dict(sorted(experts.items())),
            "experts_bpw": round(8 * sum(experts.values()) / exp_e, 3) if exp_e else None,
            "bpw": round(8 * total_b / total_e, 3) if total_e else None, "roles": roles(tensors)}


def load_fingerprints(path: Path = FINGERPRINTS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verdict(fp: dict, table: dict) -> tuple[str, str]:
    """(kind, sentence): kind is "known" (a file setup installs, under any name), "layout" (the same routed-expert
    layout as one: a fine-tune made with the same recipe, untested), "family" (another GGUF of an architecture a
    family of Puddle's has: it runs as that family, its estimates are that family's), "no" (Puddle cannot run it)."""
    known = {k: v for k, v in table.items() if not k.startswith("_")}
    title = family_title(fp)
    exact = [k for k, v in known.items() if v["expert_bytes"] == fp["expert_bytes"] and v["bytes"] == fp["bytes"]
             and v["tensors"] == fp["tensors"] and v.get("arch") == fp["arch"]]
    if fp["arch"] in LLAMA_ARCHS and not title:       # llama-server's: setup's own file, or nothing
        if exact:
            return "known", known_sentence(exact)
        return "no", (f"a {fp['arch']} model setup does not offer: Puddle runs {LLAMA_ARCHS[fp['arch']]} from "
                      "NVIDIA's own file on llama-server, and sets up no other file of it")
    if fp["arch"] not in POLY_ARCHS:
        return "no", (f"not a model Puddle runs (architecture {fp['arch']!r}): its strata-poly engine runs "
                      + ", ".join(POLY_ARCHS) + " (Laguna, Nemotron 3.5 Super, Gemma 4), and llama-server "
                      + ", ".join(LLAMA_ARCHS) + " (" + ", ".join(LLAMA_ARCHS.values()) + ")")
    if not title:
        return "no", (f"a {fp['arch']} model of {fp.get('layers')} layers: Puddle has no family of that size "
                      "(Laguna-S has 48, Laguna-XS 40)")
    if exact:
        return "known", known_sentence(exact)
    same = [k for k, v in known.items() if v["expert_bytes"] == fp["expert_bytes"]]
    if same:
        return "layout", ("the routed experts are stored exactly as in " + " / ".join(same) +
                          ", but the file is not one setup knows (a fine-tune with the same recipe?): it may run as "
                          "that size; not tested")
    return "family", (f"{title} (architecture {fp['arch']!r}): Puddle runs it on the strata-poly engine as that "
                      "family, a file setup does not offer: not tested; give its folder with --gguf-dir")


def make_fingerprints(out: Path) -> None:
    table = {}
    for fam, model, files in known_files():
        table[f"{fam}/{model}"] = {**fingerprint(*read_headers(files)), "files": [f.name for f in files],
                                   "url": files[0].url.rsplit("/", 1)[0] + "/"}
        print(f"{fam}/{model}: experts {table[f'{fam}/{model}']['experts_bpw']} bits/weight", file=sys.stderr)
    union: dict[str, set] = defaultdict(set)
    for v in table.values():
        for r, ts in v.pop("roles").items():
            union[r] |= set(ts)
    table["_roles"] = {r: sorted(ts) for r, ts in sorted(union.items())}   # every quantization each tensor has
    out.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="ms:owner/repo, hf:owner/repo, a URL, a .gguf file or a folder")
    ap.add_argument("variant", nargs="?", help="a variant name or glob (e.g. IQ3_S, UD-Q2_K_XL)")
    ap.add_argument("--fingerprints", metavar="OUT", help="read the files setup installs and write the table")
    a = ap.parse_args()
    if a.fingerprints:
        make_fingerprints(Path(a.fingerprints))
        return
    if not a.source:
        ap.error("a source is needed")
    files = [f for f in qs.resolve(a.source) if f.name.endswith(".gguf") and "mmproj" not in f.name.lower()]
    variants: dict[str, list] = defaultdict(list)
    for f in files:
        variants[qs.variant_of(f.name)].append(f)
    if not variants:
        sys.exit("no model .gguf files there")
    chosen = {k: v for k, v in variants.items() if not a.variant or a.variant in k or fnmatch.fnmatch(k, a.variant)}
    if len(chosen) > 1:
        print("variants (pass one as the second argument):")
        for name, fs in sorted(chosen.items()):
            print(f"  {name:70s} {sum(f.size for f in fs) / 1e9:8.2f} GB  {len(fs)} file(s)")
        return
    if not chosen:
        sys.exit(f"no variant matches {a.variant!r}")
    name, fs = next(iter(chosen.items()))
    fs = sorted(fs, key=lambda f: f.name)
    arch, tensors = read_headers(fs)
    qs.summarize(name, tensors)
    fp = fingerprint(arch, tensors)
    kind, sentence = verdict(fp, load_fingerprints())
    print(f"\nPuddle: {sentence}")
    sys.exit(0 if kind in ("known", "layout") else 2 if kind == "no" else 0)


if __name__ == "__main__":
    main()
