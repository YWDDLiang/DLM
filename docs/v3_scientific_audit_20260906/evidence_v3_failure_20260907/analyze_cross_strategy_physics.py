"""CPU-only descriptive analysis of the frozen nine-endpoint development ledger.

The primary matrix has reference/K4/K8/G x native/tau columns; G quantization is
an auxiliary diagnostic. No model, potential, new relaxation or cohort selection.
"""
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np
from pymatgen.core.periodic_table import Element

BASE = Path(__file__).resolve().parent
OUT = BASE / "cross_strategy_physics"
OUT.mkdir(exist_ok=True)
OLD_SHA = "f2f07653920cc2b1304a0e13f40f53ceeaae93ee0089701e2a2dc0ce86f98a82"
V3_SHA = "4c7ef1d35943fb331c38a039e1833519f83b90ce77130849f1818b235913d84b"


def load(name, expected):
    raw = (BASE / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    return json.loads(raw)


old = load("old_cases.json", OLD_SHA)
v3 = load("v3_cases.json", V3_SHA)
endpoints = dict(old["endpoints"])
endpoints.update(g_native=v3["endpoints"]["native"], g_tau=v3["endpoints"]["tau800"],
                 g_quantized=v3["endpoints"]["quantized"])
PRIMARY = [f"{s}_{e}" for s in ("reference", "k4", "k8", "g") for e in ("native", "tau")]
FLAGS = ("reconstructed", "novel", "unique_representative", "novel_unique", "terminal_verified",
         "strict_stable", "meta_stable", "strict_sun", "meta_sun", "verified_strict_sun", "verified_meta_sun")
rows = {}
for name, endpoint in endpoints.items():
    indexed = {int(row["sample_idx"]): row for row in endpoint["rows"]}
    assert len(indexed) == 256 and set(indexed) == set(range(256))
    assert len({r["group_id"] for r in indexed.values()}) == 256
    assert not endpoint.get("condition_mismatches", [])
    for i, row in indexed.items():
        reference = endpoints["g_native"]["rows"][i]
        assert reference["sample_idx"] == i
        for key in ("group_id", "plan_state", "species_program", "sampling_batch_size", "sampling_seed"):
            assert row[key] == reference[key], (name, i, key)
        assert row["attempt"]["group_id"] == row["group_id"]
        if row.get("structure") is not None:
            sites = row["structure"]["sites"]
            assert len(sites) == row["plan_state"]["N"]
            assert all(len(s["species"]) == 1 and s["species"][0]["occu"] == 1 for s in sites)
            assert Counter(s["species"][0]["element"] for s in sites) == Counter(
                dict(zip(row["plan_state"]["elements"], row["plan_state"]["counts"])))
    for flag in FLAGS:
        assert sum(bool(r["attempt"][flag]) for r in indexed.values()) == endpoint["report"]["counts"][flag]
    rows[name] = indexed


def finite(x):
    return isinstance(x, (float, int)) and math.isfinite(x)


def describe(values):
    values = np.asarray([float(x) for x in values if finite(x)], dtype=float)
    return {"n": len(values), **({"mean": float(values.mean()), "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
        "min": float(values.min()), "max": float(values.max())} if len(values) else {})}


def count(rows_to_count):
    selected = list(rows_to_count)
    result = {"n": len(selected), **{f: sum(bool(r["attempt"][f]) for r in selected) for f in FLAGS}}
    result["known_hull"] = sum(r["attempt"]["official_hull_status"] == "known" for r in selected)
    result["known_finite_terminal"] = sum(r["attempt"]["official_hull_status"] == "known" and
                                         finite(r["attempt"]["terminal_energy_eV_atom"]) for r in selected)
    result["statuses"] = dict(Counter(r["attempt"]["terminal_status"] for r in selected))
    result["hull_statuses"] = dict(Counter(r["attempt"]["official_hull_status"] for r in selected))
    for threshold in ("strict", "meta"):
        result[threshold + "_stable_not_sun"] = sum(r["attempt"][threshold + "_stable"] and
                                                     not r["attempt"][threshold + "_sun"] for r in selected)
    return result


geometry_cache = {}


def geometry(structure):
    if structure is None:
        return None
    key = hashlib.sha256(json.dumps(structure, sort_keys=True).encode()).hexdigest()
    if key in geometry_cache:
        return geometry_cache[key]
    L = np.asarray(structure["lattice"], dtype=float)
    if L.shape != (3, 3):
        L = np.asarray(structure["lattice"]["matrix"], dtype=float)
    F = np.asarray([site["abc"] for site in structure["sites"]], dtype=float) % 1
    symbols = [site["species"][0]["element"] for site in structure["sites"]]
    N = len(F)
    volume = abs(float(np.linalg.det(L)))
    assert N > 0 and np.isfinite(L).all() and np.isfinite(F).all() and volume > 0
    i, j = np.triu_indices(N, 1)
    delta = np.vstack((F[i] - F[j], np.zeros((1, 3))))
    # The last pair is a single lattice self-image; its zero image is excluded.
    def distances(ranges):
        offsets = np.asarray(list(itertools.product(*(range(-r, r + 1) for r in ranges))), dtype=float)
        d = np.linalg.norm((delta[:, None, :] + offsets[None, :, :]) @ L, axis=-1)
        d[-1, np.flatnonzero((offsets == 0).all(-1))] = np.inf
        return d.min(-1)
    first = distances((2, 2, 2))
    best = float(first.min())
    # If a Cartesian vector has length < best, each fractional component is
    # bounded by best * norm(column_j(L^-1)); delta components lie in (-1,1).
    radius = np.ceil(1 + best * np.linalg.norm(np.linalg.inv(L), axis=0)).astype(int)
    image_count = int(np.prod(2 * radius + 1))
    certified = image_count <= 200000
    minima = distances(radius) if certified and np.any(radius > 2) else first
    mass = sum(float(Element(s).atomic_mass) for s in symbols)
    result = {"num_atoms": N, "arity": len(set(symbols)), "volume_A3": volume,
              "volume_per_atom_A3": volume / N, "mass_density_g_cm3": mass * 1.66053906660 / volume,
              "lattice_condition": float(np.linalg.cond(L)), "min_distance_A": float(minima.min()),
              "min_different_atom_distance_A": float(minima[:-1].min()) if N > 1 else None,
              "min_self_image_A": float(minima[-1]),
              "short_pairs_lt_0p5_A": int((minima[:-1] < .5).sum()),
              "short_pairs_lt_1_A": int((minima[:-1] < 1.).sum()),
              "distance_certified": certified, "complete_box_images": image_count,
              "radius2_min_distance_A": best}
    geometry_cache[key] = result
    return result


features = {name: {i: geometry(row.get("structure")) for i, row in indexed.items()}
            for name, indexed in rows.items()}


def field_set(names, flag, predicate=all):
    return [i for i in range(256) if predicate(bool(rows[name][i]["attempt"][flag]) for name in names)]


def formula(plan):
    return "".join(s + (str(n) if n != 1 else "") for s, n in zip(plan["elements"], plan["counts"]))


def case(i):
    return {"sample_idx": i, "group_id": rows["g_native"][i]["group_id"],
            "formula": formula(rows["g_native"][i]["plan_state"]),
            "N": rows["g_native"][i]["plan_state"]["N"],
            "endpoints": {name: {"status": rows[name][i]["attempt"]["terminal_status"],
                "hull_status": rows[name][i]["attempt"]["official_hull_status"],
                "e_hull": rows[name][i]["attempt"]["e_above_hull_eV_atom"],
                **{f: rows[name][i]["attempt"][f] for f in FLAGS},
                "geometry": features[name][i]} for name in PRIMARY}}


persistence = {}
for group, names in {
    "native_four": [s + "_native" for s in ("reference", "k4", "k8", "g")],
    "tau_four": [s + "_tau" for s in ("reference", "k4", "k8", "g")],
    "all_eight": PRIMARY,
    "old_native_three": [s + "_native" for s in ("reference", "k4", "k8")],
    "old_tau_three": [s + "_tau" for s in ("reference", "k4", "k8")],
}.items():
    persistence[group] = {}
    for flag in ("strict_stable", "meta_stable", "strict_sun", "meta_sun", "verified_strict_sun", "verified_meta_sun"):
        hits = field_set(names, flag)
        no_hits = [i for i in range(256) if not any(rows[n][i]["attempt"][flag] for n in names)]
        fully_observed = [i for i in no_hits if all(rows[n][i]["attempt"]["reconstructed"] and
            rows[n][i]["attempt"]["official_hull_status"] == "known" and
            finite(rows[n][i]["attempt"]["terminal_energy_eV_atom"]) for n in names)]
        persistence[group][flag] = {"all_hit_n": len(hits), "all_hit_ids": hits,
            "none_hit_n": len(no_hits), "none_hit_ids": no_hits,
            "none_hit_all_finite_known_n": len(fully_observed), "none_hit_all_finite_known_ids": fully_observed}


transitions = {}
for left, right in ([(s + "_" + e, "g_" + e) for e in ("native", "tau") for s in ("reference", "k4", "k8")]
    + [(s + "_native", s + "_tau") for s in ("reference", "k4", "k8", "g")]
    + [("g_native", "g_quantized")]):
    key = left + "__to__" + right
    transitions[key] = {}
    for flag in FLAGS:
        cells = {"both": [], "lost": [], "gained": [], "neither": []}
        for i in range(256):
            a, b = (bool(rows[n][i]["attempt"][flag]) for n in (left, right))
            cells["both" if a and b else "lost" if a else "gained" if b else "neither"].append(i)
        transitions[key][flag] = {k + "_n": len(v) for k, v in cells.items()} | {k + "_ids": v for k, v in cells.items()}


bands = {
    "N": [("2-4", 0, 5), ("5-8", 5, 9), ("9-12", 9, 13), ("13-20", 13, 21)],
    "arity": [("1-2", 0, 3), ("3", 3, 4), ("4+", 4, 99)],
    "volume_per_atom_A3": [("<5", 0, 5), ("5-10", 5, 10), ("10-20", 10, 20), ("20-40", 20, 40), (">=40", 40, math.inf)],
    "min_distance_A": [("<0.5", 0, .5), ("0.5-1", .5, 1), ("1-1.5", 1, 1.5), ("1.5-2", 1.5, 2), (">=2", 2, math.inf)],
    "mass_density_g_cm3": [("<2", 0, 2), ("2-5", 2, 5), ("5-10", 5, 10), (">=10", 10, math.inf)],
    "lattice_condition": [("<2", 0, 2), ("2-4", 2, 4), (">=4", 4, math.inf)],
}
strata = {}
for name, indexed in rows.items():
    strata[name] = {}
    for feature, bins in bands.items():
        strata[name][feature] = {}
        for label, lower, upper in bins:
            ids = []
            for i, row in indexed.items():
                value = (int(row["plan_state"]["N"]) if feature == "N" else len(row["plan_state"]["elements"])
                         if feature == "arity" else None if features[name][i] is None else features[name][i][feature])
                if value is not None and lower <= value < upper:
                    ids.append(i)
            strata[name][feature][label] = count(indexed[i] for i in ids)


E = np.zeros((6, 3, 3))
E[0] = np.diag([1., -1., 0.]) / np.sqrt(2)
E[1] = np.diag([1., 1., -2.]) / np.sqrt(6)
for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 2):
    E[k, i, j] = E[k, j, i] = 1 / np.sqrt(2)
E[5] = np.eye(3) / np.sqrt(3)
normalizer = v3["training_config"]["normalizer"]
mu, sd = np.asarray(normalizer["mean"]), np.asarray(normalizer["std"])


def prior_structure(row):
    prior = row["initial_geometry_prior"]
    coefficients = mu + sd * np.asarray(prior["z"])
    coefficients[5] += math.log(row["plan_state"]["N"]) / np.sqrt(3)
    S = np.einsum("k,kij->ij", coefficients, E)
    eig, vectors = np.linalg.eigh(S)
    L = np.linalg.cholesky((vectors * np.exp(2 * eig)) @ vectors.T)
    return {"lattice": L.tolist(), "sites": [{"species": site["species"], "abc": f}
        for site, f in zip(row["structure"]["sites"], prior["fractional"])]}


movement = []
for i, row in rows["g_native"].items():
    prior = prior_structure(row)
    f0 = np.asarray([s["abc"] for s in prior["sites"]])
    f1 = np.asarray([s["abc"] for s in row["structure"]["sites"]])
    L1 = np.asarray(row["structure"]["lattice"])
    delta = f1 - f0
    delta -= np.round(delta)
    centered = delta - delta.mean(0)
    initial, final = geometry(prior), features["g_native"][i]
    lattice_only = geometry({**prior, "lattice": row["structure"]["lattice"]})
    coordinates_only = geometry({**row["structure"], "lattice": prior["lattice"]})
    relaxed = geometry(row["relaxed_structure"])
    eig, vectors = np.linalg.eigh(L1 @ L1.T)
    S = (vectors * (.5 * np.log(eig))) @ vectors.T
    c = np.einsum("kij,ij->k", E, S)
    c[5] -= math.log(len(f1)) / np.sqrt(3)
    z_final = (c - mu) / sd
    movement.append({"sample_idx": i, "group_id": row["group_id"],
        "fractional_rms": float(np.sqrt(np.mean(delta**2))),
        "centered_fractional_rms": float(np.sqrt(np.mean(centered**2))),
        "centered_cartesian_atom_rms_A": float(np.sqrt(np.mean(np.sum((centered @ L1)**2, axis=-1)))),
        "common_translation_fractional_norm": float(np.linalg.norm(delta.mean(0))),
        "z_change_norm": float(np.linalg.norm(z_final - np.asarray(row["initial_geometry_prior"]["z"]))),
        "volume_ratio_final_over_prior": final["volume_A3"] / initial["volume_A3"],
        "initial_geometry": initial, "final_geometry": final, "relaxed_geometry": relaxed,
        "lattice_only_geometry_diagnostic": lattice_only,
        "coordinates_only_geometry_diagnostic": coordinates_only})


prior_failure = {
    "prior_short_distance_lt_0p5": [m["sample_idx"] for m in movement if m["initial_geometry"]["min_distance_A"] < .5],
    "final_short_distance_lt_0p5": [m["sample_idx"] for m in movement if m["final_geometry"]["min_distance_A"] < .5],
}
initial_bad, final_bad = map(set, prior_failure.values())
prior_failure.update(both_ids=sorted(initial_bad & final_bad), repaired_ids=sorted(initial_bad - final_bad),
                     introduced_ids=sorted(final_bad - initial_bad))
prior_failure["lattice_only_short_ids"] = [m["sample_idx"] for m in movement
    if m["lattice_only_geometry_diagnostic"]["min_distance_A"] < .5]
prior_failure["coordinates_only_short_ids"] = [m["sample_idx"] for m in movement
    if m["coordinates_only_geometry_diagnostic"]["min_distance_A"] < .5]
g_failure_groups = {}
for label, ids in {"initially_short": sorted(initial_bad), "initially_not_short": sorted(set(range(256)) - initial_bad),
                   "finally_short": sorted(final_bad), "finally_not_short": sorted(set(range(256)) - final_bad)}.items():
    g_failure_groups[label] = {name: count(rows[name][i] for i in ids) for name in PRIMARY}


def exclusive_failure(row, threshold):
    a = row["attempt"]
    if not a["reconstructed"]:
        return "generation_or_parse_failure"
    if a["official_hull_status"] != "known":
        return "unknown_hull"
    if not finite(a["terminal_energy_eV_atom"]):
        return "no_finite_terminal_energy:" + a["terminal_status"]
    if not a[threshold + "_stable"]:
        return "above_energy_threshold"
    if not a["novel_unique"]:
        return "stable_but_NU_failed"
    return "SUN"


summary = {name: count(indexed.values()) for name, indexed in rows.items()}
for name, indexed in rows.items():
    summary[name]["geometry"] = {key: describe(f[key] for f in features[name].values() if f is not None)
        for key in ("volume_per_atom_A3", "mass_density_g_cm3", "min_distance_A", "lattice_condition")}
    summary[name]["failure_decomposition"] = {t: dict(Counter(exclusive_failure(r, t) for r in indexed.values()))
                                               for t in ("strict", "meta")}

old_consensus_g = {}
for endpoint in ("native", "tau"):
    for threshold in ("strict", "meta"):
        for suffix in ("stable", "sun"):
            flag = threshold + "_" + suffix
            old_ids = set(field_set([s + "_" + endpoint for s in ("reference", "k4", "k8")], flag))
            g_ids = set(field_set(["g_" + endpoint], flag))
            any_old = set(field_set([s + "_" + endpoint for s in ("reference", "k4", "k8")], flag, any))
            old_consensus_g[endpoint + "_" + flag] = {
                "old_consensus": sorted(old_ids), "g_lost_from_old_consensus": sorted(old_ids - g_ids),
                "g_retained_old_consensus": sorted(old_ids & g_ids),
                "g_new_beyond_any_old": sorted(g_ids - any_old)}

validation = []
for epoch, event in enumerate(v3["validation"], 1):
    validation.append({"epoch": epoch, "bands": [{"band": band,
        "coordinate_mse": event[f"t_band_{band}_geometry_coordinates_sum"] / event[f"t_band_{band}_geometry_weight_sum"],
        "lattice_mse": event[f"t_band_{band}_geometry_lattice_sum"] / event[f"t_band_{band}_geometry_weight_sum"]}
        for band in range(6)]})

result = {"source_sha256": {"old_cases.json": OLD_SHA, "v3_cases.json": V3_SHA},
    "source_receipts": {**old["files"], **v3["files"]}, "requests": 256, "condition_mismatches": 0,
    "matrix_order": PRIMARY, "summary": summary, "persistence": persistence, "transitions": transitions,
    "strata": strata, "old_consensus_g": old_consensus_g,
    "g_prior_movement": {k: describe(m[k] for m in movement) for k in (
        "fractional_rms", "centered_fractional_rms", "centered_cartesian_atom_rms_A",
        "common_translation_fractional_norm", "z_change_norm", "volume_ratio_final_over_prior")},
    "g_prior_collision_transition": prior_failure, "g_collision_groups": g_failure_groups,
    "validation_by_band": validation,
    "geometry_computations": len(geometry_cache),
    "uncertified_geometry_checks": sum(not x["distance_certified"] for x in geometry_cache.values()),
    "interpretation": "observed strategy outcomes, not independent trials or composition impossibility; U remains the original full-cohort flag"}
(OUT / "ANALYSIS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
(OUT / "G_PRIOR_MOVEMENT.json").write_text(json.dumps(movement, indent=2, allow_nan=False) + "\n", encoding="utf-8")
(OUT / "CASES.json").write_text(json.dumps([case(i) for i in range(256)], indent=2, allow_nan=False) + "\n", encoding="utf-8")
with (OUT / "MATRIX_256.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    fields = ["sample_idx", "group_id", "formula", "N", "arity"]
    fields += [name + ":" + flag for name in PRIMARY for flag in FLAGS + ("terminal_status", "official_hull_status", "e_above_hull_eV_atom")]
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for i in range(256):
        p = rows["g_native"][i]["plan_state"]
        record = {"sample_idx": i, "group_id": rows["g_native"][i]["group_id"], "formula": formula(p),
                  "N": p["N"], "arity": len(p["elements"])}
        record.update({name + ":" + flag: rows[name][i]["attempt"][flag] for name in PRIMARY
                       for flag in FLAGS + ("terminal_status", "official_hull_status", "e_above_hull_eV_atom")})
        writer.writerow(record)
with (OUT / "STRATA.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    fields = ["endpoint", "feature", "band", "n", *FLAGS, "known_hull", "known_finite_terminal", "statuses"]
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for endpoint, features_table in strata.items():
        for feature, table in features_table.items():
            for band, stats in table.items():
                record = {key: stats[key] for key in fields if key in stats}
                record.update(endpoint=endpoint, feature=feature, band=band,
                              statuses=json.dumps(stats["statuses"], sort_keys=True))
                writer.writerow(record)
with (OUT / "GEOMETRY_FEATURES.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    fields = ["endpoint", "sample_idx", "group_id", "formula", "terminal_status", "strict_stable", "meta_stable",
              "strict_sun", "meta_sun", *next(iter(geometry_cache.values())).keys()]
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for endpoint, indexed in rows.items():
        for i, row in indexed.items():
            record = {"endpoint": endpoint, "sample_idx": i, "group_id": row["group_id"],
                      "formula": formula(row["plan_state"]), **{key: row["attempt"][key] for key in
                          ("terminal_status", "strict_stable", "meta_stable", "strict_sun", "meta_sun")}}
            record.update(features[endpoint][i] or {})
            writer.writerow(record)
print(json.dumps({"output": str(OUT), "summary": {n: {k: summary[n][k] for k in (
    "strict_stable", "meta_stable", "strict_sun", "meta_sun", "novel_unique", "statuses")} for n in PRIMARY},
    "prior_collision": {k: len(v) for k, v in prior_failure.items()},
    "movement": result["g_prior_movement"], "uncertified": result["uncertified_geometry_checks"]}, allow_nan=False))
