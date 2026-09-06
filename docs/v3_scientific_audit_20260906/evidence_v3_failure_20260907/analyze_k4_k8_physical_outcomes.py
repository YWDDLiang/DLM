"""K4/K8/reference outcomes and actual R-terminal contact geometry, CPU only."""
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "k4k8_physics"
OUT.mkdir(exist_ok=True)


def load(name, sha):
    raw = (BASE / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha
    return json.loads(raw)


OLD_SHA = "f2f07653920cc2b1304a0e13f40f53ceeaae93ee0089701e2a2dc0ce86f98a82"
LABEL_SHA = "1dd63a7812db564c147d5565ac6c4a05aa8bed1d0b7e20ebdb13ed26155be28a"
old = load("old_cases.json", OLD_SHA)
saved_labels = load("k4k8_labels_compact.json", LABEL_SHA)
radii_bytes = (OUT / "covalent_radii.json").read_bytes()
RADII_SHA = hashlib.sha256(radii_bytes).hexdigest()
assert RADII_SHA == "74bf9289545246bead1fa2b2a70bd5cbd73498ae3cef74f5d7ec24007c78bec7"
RADII = json.loads(radii_bytes)
rows = {name: {int(r["sample_idx"]): r for r in e["rows"]} for name, e in old["endpoints"].items()}
labels = {}
for name, records in saved_labels["endpoints"].items():
    by_id = {r["trajectory_id"]: r for r in records}
    assert len(by_id) == 256
    labels[name] = {}
    for i, row in rows[name].items():
        label = by_id[row["trajectory_id"]]
        assert label["group_id"] == row["group_id"]
        assert label["status"] == row["attempt"]["terminal_status"]
        assert label["verified"] == row["attempt"]["terminal_verified"]
        assert label["terminal_energy"] == row["attempt"]["terminal_energy_eV_atom"]
        labels[name][i] = label


def describe(values):
    a = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    return {"n": len(a), **({"median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
                            "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max())}
                           if len(a) else {})}


cache = {}


def feature(structure):
    if structure is None:
        return None
    key = hashlib.sha256(json.dumps(structure, sort_keys=True).encode()).hexdigest()
    if key in cache:
        return cache[key]
    lattice = structure["lattice"]
    L = np.asarray(lattice["matrix"] if isinstance(lattice, dict) else lattice, dtype=float)
    F = np.asarray([s["abc"] for s in structure["sites"]], dtype=float) % 1.
    symbols = [s["species"][0]["element"] for s in structure["sites"]]
    radii = np.asarray([RADII[s] for s in symbols])
    n = len(F)
    sums = radii[:, None] + radii[None, :]
    soft = np.maximum(.5, .5 * sums)
    delta = F[:, None, :] - F[None, :, :]

    def distances(radius):
        offsets = np.asarray(list(itertools.product(*(range(-r, r + 1) for r in radius))), dtype=float)
        values = np.linalg.norm((delta[..., None, :] + offsets) @ L, axis=-1)
        zero = int(np.flatnonzero((offsets == 0).all(-1))[0])
        values[np.arange(n), np.arange(n), zero] = np.inf
        return values, offsets

    d2, offsets2 = distances((2, 2, 2))
    md2 = d2.min(-1)
    off_diagonal = ~np.eye(n, dtype=bool)
    min_ratio2 = float(np.min(md2 / sums))
    nn2 = md2.min(-1)
    # Complete for every vector needed by normalized shortest contact and
    # the 1.1*nearest-neighbour shell, not a fixed cell-image assumption.
    cutoff = max(float(nn2.max()) * 1.1, min_ratio2 * 2 * float(radii.max()), float(soft.max()))
    radius = np.ceil(1 + cutoff * np.linalg.norm(np.linalg.inv(L), axis=0)).astype(int)
    images = math.prod(int(2 * r + 1) for r in radius)
    certified = n * n * images <= 4_000_000 and images <= 200_000
    d, offsets = distances(radius) if certified and np.any(radius > 2) else (d2, offsets2)
    md = d.min(-1)
    nn = md.min(-1)
    shell = d <= nn[:, None, None] * 1.1 + 1e-10
    cn = shell.sum(axis=(1, 2))
    pair_ratio = md / sums
    worst = np.unravel_index(int(np.argmin(pair_ratio)), pair_ratio.shape)
    static2 = bool(((md2 < soft) & off_diagonal).any())
    static_exact = bool(((md < soft) & off_diagonal).any())
    penalty2 = np.minimum(1., np.maximum(0., np.log(soft / md2)))
    penalty2[~off_diagonal] = 0.
    site_stats = []
    for i, symbol in enumerate(symbols):
        counter = Counter()
        for j, other in enumerate(symbols):
            counter[other] += int(shell[i, j].sum())
        site_stats.append({"site": i, "element": symbol, "nearest_A": float(nn[i]),
                           "first_shell_1p1_nn": int(cn[i]), "neighbour_elements": dict(counter)})
    by_element = {}
    for symbol in sorted(set(symbols)):
        selected = [s for s in site_stats if s["element"] == symbol]
        by_element[symbol] = {"sites": len(selected), "nearest_A": describe(s["nearest_A"] for s in selected),
            "first_shell_counts": [s["first_shell_1p1_nn"] for s in selected],
            "neighbour_elements_total": dict(sum((Counter(s["neighbour_elements"]) for s in selected), Counter()))}
    r = {"N": n, "volume_A3": float(abs(np.linalg.det(L))),
         "vpa_A3": float(abs(np.linalg.det(L)) / n), "lattice_condition": float(np.linalg.cond(L)),
         "lengths_A": np.linalg.norm(L, axis=-1).tolist(),
         "min_distance_A": float(md.min()), "min_covalent_radius_ratio": float(pair_ratio.min()),
         "worst_pair": {"sites": list(map(int, worst)), "elements": [symbols[k] for k in worst],
                         "distance_A": float(md[worst]), "radii_sum_A": float(sums[worst])},
         "nearest_A": describe(nn), "coordination_1p1_nn": describe(cn),
         "per_element": by_element, "site_stats": site_stats,
         "static_pair_threshold_125": static2, "static_pair_threshold_complete": static_exact,
         "static_max_penalty_125": float(penalty2.max()),
         "certified": certified, "complete_box_images": images}
    cache[key] = r
    return r


geometries = {name: {i: {"input": feature(row.get("structure")),
                       "R_terminal": feature(labels[name][i].get("final_structure")) if name in labels else None}
                    for i, row in rr.items()} for name, rr in rows.items()}


def ids(name, predicate):
    return {i for i, row in rows[name].items() if predicate(row["attempt"])}


def true_high_energy(a, threshold=.1):
    return (a["terminal_verified"] and a["official_hull_status"] == "known" and
            a["e_above_hull_eV_atom"] is not None and a["e_above_hull_eV_atom"] > threshold)


outcomes = {}
for endpoint in ("native", "tau"):
    a, b = "k4_" + endpoint, "k8_" + endpoint
    result = {}
    for threshold in ("strict", "meta"):
        for criterion in ("stable", "sun"):
            flag = threshold + "_" + criterion
            aa = ids(a, lambda r: r["terminal_verified"] and r[flag])
            bb = ids(b, lambda r: r["terminal_verified"] and r[flag])
            result["verified_" + flag] = {"both": sorted(aa & bb), "k4_only": sorted(aa - bb),
                                         "k8_only": sorted(bb - aa), "neither": sorted(set(range(256)) - aa - bb)}
        t = 0. if threshold == "strict" else .1
        for good, bad in ((a, b), (b, a)):
            result[good + "_true_energy_win_" + threshold] = sorted(i for i in range(256)
                if rows[good][i]["attempt"]["terminal_verified"] and rows[good][i]["attempt"][threshold + "_stable"]
                and true_high_energy(rows[bad][i]["attempt"], t))
    result["common_verified_above_meta"] = sorted(ids(a, true_high_energy) & ids(b, true_high_energy))
    result["either_unknown"] = sorted(ids(a, lambda r: r["official_hull_status"].startswith("official_cache")) |
                                      ids(b, lambda r: r["official_hull_status"].startswith("official_cache")))
    result["either_generation_or_parse_failure"] = sorted(ids(a, lambda r: not r["reconstructed"]) |
                                                          ids(b, lambda r: not r["reconstructed"]))
    result["either_invalid_raw"] = sorted(ids(a, lambda r: r["terminal_status"] == "invalid_raw") |
                                           ids(b, lambda r: r["terminal_status"] == "invalid_raw"))
    result["either_invalid_terminal"] = sorted(ids(a, lambda r: r["terminal_status"] == "invalid_terminal") |
                                                ids(b, lambda r: r["terminal_status"] == "invalid_terminal"))
    result["both_energy_stable_but_either_NU_lost"] = {
        t: sorted(i for i in range(256) if all(rows[n][i]["attempt"][t + "_stable"] for n in (a, b))
                  and not all(rows[n][i]["attempt"]["novel_unique"] for n in (a, b))) for t in ("strict", "meta")}
    outcomes[endpoint] = result

def loss_reason(other, threshold):
    if not other["reconstructed"]:
        return "generation_or_parse_failure"
    if other["official_hull_status"] != "known":
        return "unknown_hull"
    if other["terminal_status"] in ("invalid_raw", "invalid_terminal"):
        return other["terminal_status"]
    if not other["terminal_verified"]:
        return "not_verified:" + other["terminal_status"]
    if not other[threshold + "_stable"]:
        return "both_verified_energy_threshold_flip"
    if not other["novel_unique"]:
        return "both_verified_energy_stable_NU_loss"
    return "unexpected"

one_sided_verified_reasons = {}
for endpoint, table in outcomes.items():
    for threshold in ("strict", "meta"):
        for winner, loser in (("k4", "k8"), ("k8", "k4")):
            selected = table["verified_" + threshold + "_sun"][winner + "_only"]
            one_sided_verified_reasons[endpoint + ":" + threshold + ":" + winner + "_only"] = {
                "ids": selected, "loser_reasons": dict(Counter(loss_reason(rows[loser + "_" + endpoint][i]["attempt"], threshold)
                                                               for i in selected))}


def counts_for(name, chosen):
    selected = [rows[name][i]["attempt"] for i in chosen]
    return {"n": len(selected), **{f: sum(r[f] for r in selected) for f in
        ("strict_stable", "meta_stable", "strict_sun", "meta_sun", "terminal_verified", "verified_strict_sun", "verified_meta_sun")},
        "statuses": dict(Counter(r["terminal_status"] for r in selected))}


contact_association = {}
for name in rows:
    flagged2 = {i for i in rows[name] if geometries[name][i]["input"] is not None and
                geometries[name][i]["input"]["static_pair_threshold_125"]}
    flagged_exact = {i for i in rows[name] if geometries[name][i]["input"] is not None and
                     geometries[name][i]["input"]["static_pair_threshold_complete"]}
    original = counts_for(name, flagged2)
    item = {"input_flagged": original, "input_not_flagged": counts_for(name, set(range(256)) - flagged2),
            "static_125_ids": sorted(flagged2), "static_complete_ids": sorted(flagged_exact),
            "static_125_vs_complete_changed_ids": sorted(flagged2 ^ flagged_exact),
            "not_actual_prefix_coverage": True}
    if name.endswith("native"):
        item["same_ids_after_tau"] = counts_for(name.replace("native", "tau"), flagged2)
    if name in labels:
        for group, predicate in {"verified_meta_success": lambda a: a["terminal_verified"] and a["meta_stable"],
                                  "verified_above_meta": true_high_energy,
                                  "invalid_terminal": lambda a: a["terminal_status"] == "invalid_terminal"}.items():
            selected = [geometries[name][i]["R_terminal"] for i in ids(name, predicate)]
            selected = [f for f in selected if f is not None]
            item[group + "_R_geometry"] = {"n": len(selected),
                "short_pair_trigger": sum(f["static_pair_threshold_125"] for f in selected),
                "min_radius_ratio": describe(f["min_covalent_radius_ratio"] for f in selected),
                "vpa_A3": describe(f["vpa_A3"] for f in selected),
                "nearest_A": describe(f["nearest_A"]["median"] for f in selected),
                "first_shell_CN": describe(f["coordination_1p1_nn"]["median"] for f in selected)}
    contact_association[name] = item


stop = {}
for name in labels:
    stop[name] = dict(Counter((l["status"], str(l["optimizer_converged"]),
        "force_pass" if (rows[name][i]["attempt"].get("terminal") or {}).get("force_max_eV_A", math.inf) <= .1 else "force_fail",
        "stress_pass" if (rows[name][i]["attempt"].get("terminal") or {}).get("stress_max_GPa", math.inf) <= .5 else "stress_fail")
        for i, l in labels[name].items()))
    stop[name] = {"|".join(k): v for k, v in stop[name].items()}


case_records = []
for i in range(256):
    plan = rows["k4_native"][i]["plan_state"]
    case_records.append({"sample_idx": i, "group_id": rows["k4_native"][i]["group_id"],
        "formula": "".join(s + (str(n) if n != 1 else "") for s, n in zip(plan["elements"], plan["counts"])),
        "plan": plan, "P": rows["k4_native"][i]["species_program"],
        "endpoints": {name: {"attempt": rows[name][i]["attempt"], "geometry": geometries[name][i],
                             "label_details": {k: labels[name][i].get(k) for k in
                                 ("status", "optimizer_converged", "actual_steps", "raw_min_distance_A",
                                  "terminal_min_distance_A", "error", "endpoint_cache_key")}
                             if name in labels else None} for name in rows}})

summary = {"input_sha256": {"old_cases.json": OLD_SHA, "k4k8_labels_compact.json": LABEL_SHA},
           "radii_sha256": RADII_SHA, "source_receipts": {**old["files"], **saved_labels["source_receipts"]},
           "requests": 256, "outcomes": outcomes, "one_sided_verified_reasons": one_sided_verified_reasons,
           "contact_association": contact_association,
           "optimizer_status_cross": stop, "geometry_cache_size": len(cache),
           "uncertified_geometries": sum(not f["certified"] for f in cache.values()),
           "missing_radii": sorted({s for e in rows.values() for r in e.values() for s in r["plan_state"]["elements"]} - set(RADII)),
           "coordination_definition": "count of all periodic neighbour images with distance <=1.1*the atom nearest distance; geometric first shell, not valence/bond-order assignment",
           "interpretation": "all stored outcomes retained; no new energy calculation; a static completed structure is not an actual scalar prefix"}
(OUT / "K4_K8_ANALYSIS.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
(OUT / "K4_K8_CASES.json").write_text(json.dumps(case_records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
with (OUT / "K4_K8_OUTCOME_GROUPS.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, ["endpoint", "criterion", "group", "count", "sample_indices"])
    writer.writeheader()
    for endpoint, table in outcomes.items():
        for criterion, value in table.items():
            groups = value if isinstance(value, dict) else {"all": value}
            for group, indices in groups.items():
                writer.writerow({"endpoint": endpoint, "criterion": criterion, "group": group,
                                 "count": len(indices), "sample_indices": ";".join(map(str, indices))})
with (OUT / "K4_K8_OUTCOME_MATRIX.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    fields = ["sample_idx", "formula"] + [n + ":" + f for n in rows for f in
        ("terminal_status", "terminal_verified", "official_hull_status", "e_above_hull_eV_atom", "strict_stable",
         "meta_stable", "strict_sun", "meta_sun", "verified_strict_sun", "verified_meta_sun")]
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for c in case_records:
        r = {"sample_idx": c["sample_idx"], "formula": c["formula"]}
        r.update({n + ":" + f: c["endpoints"][n]["attempt"][f] for n in rows for f in
            ("terminal_status", "terminal_verified", "official_hull_status", "e_above_hull_eV_atom", "strict_stable",
             "meta_stable", "strict_sun", "meta_sun", "verified_strict_sun", "verified_meta_sun")})
        writer.writerow(r)
print(json.dumps({"outcomes": {e: {k: {kk: len(vv) for kk, vv in v.items()} if isinstance(v, dict)
    else len(v) for k, v in r.items()} for e, r in outcomes.items()},
    "contacts": {n: {"static125": len(v["static_125_ids"]), "difference_to_complete": v["static_125_vs_complete_changed_ids"],
                     "flagged_verified_strict_sun": v["input_flagged"]["verified_strict_sun"]}
                  for n, v in contact_association.items()}, "uncertified": summary["uncertified_geometries"],
                  "missing_radii": summary["missing_radii"]}, allow_nan=False))
