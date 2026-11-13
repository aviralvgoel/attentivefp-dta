"""Deterministic group splits and explicit leakage checks."""

from __future__ import annotations

from itertools import combinations

import numpy as np

PARTITIONS = ("train", "validation", "test")


def check_indices(payload, available, allow_excluded=False):
    sets = {}
    available = set(map(int, available))
    for key in PARTITIONS:
        values = payload.get(key, payload.get("val") if key == "validation" else None)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a non-empty list of integer IDs")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise ValueError(f"{key} contains non-integer IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate IDs within {key}")
        sets[key] = set(values)
        if not sets[key] <= available:
            raise ValueError(f"Unknown IDs in {key}: {sorted(sets[key] - available)[:5]}")
    for a, b in combinations(PARTITIONS, 2):
        if sets[a] & sets[b]:
            raise ValueError(f"Pair/sample overlap between {a} and {b}")
    missing = available - set.union(*sets.values())
    excluded_values = payload.get("excluded", [])
    excluded = set(excluded_values)
    if len(excluded_values) != len(excluded) or not excluded <= available:
        raise ValueError("Invalid/duplicated excluded IDs")
    if excluded & set.union(*sets.values()):
        raise ValueError("Excluded IDs appear in a partition")
    if missing and not (allow_excluded and missing == excluded):
        raise ValueError(f"{len(missing)} samples omitted from splits without explicit exclusions")
    return sets


def group_partition(keys, seed, fractions=(0.8, 0.1, 0.1)):
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError("Three positive split fractions must sum to 1")
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(str(key), []).append(i)
    if len(groups) < 3:
        raise ValueError("Need at least three distinct groups for a three-way split")
    rng = np.random.default_rng(seed)
    names = list(groups)
    rng.shuffle(names)
    names.sort(key=lambda key: len(groups[key]), reverse=True)  # random tie breaking
    target = np.asarray(fractions) * len(keys)
    assigned = [[] for _ in PARTITIONS]
    # Fill biggest groups first into the partition with greatest remaining capacity.
    # Splits can be imbalanced for large scaffolds; preserve groups rather than forcing sizes.
    for i, name in enumerate(names):
        empty = [j for j in range(3) if not assigned[j]]
        remaining_groups = len(names) - i
        j = (
            empty[0]
            if remaining_groups == len(empty)
            else int(np.argmax(target - np.array([len(a) for a in assigned])))
        )
        assigned[j].extend(groups[name])
    return {name: sorted(ids) for name, ids in zip(PARTITIONS, assigned)}


def make_dta_split(frame, mode="cold_drug", seed=42, fractions=(0.8, 0.1, 0.1)):
    columns = {
        "random": "pair_id",
        "cold_drug": "drug_id",
        "scaffold": "scaffold",
        "cold_target": "protein_id",
    }
    if mode not in columns and mode != "cold_both":
        raise ValueError(f"Unknown split mode {mode}")
    if mode == "cold_both":
        # Assign drug entities and protein entities independently, then retain
        # only pairs whose endpoints fall in the SAME partition. Cross pairs
        # are explicit exclusions, not allowed to leak back into training.
        drug_keys = sorted(frame.drug_id.unique())
        prot_keys = sorted(frame.protein_id.unique())
        ds = group_partition(drug_keys, seed, fractions)
        ps = group_partition(prot_keys, seed + 1, fractions)
        dm = {drug_keys[i]: part for part, ids in ds.items() for i in ids}
        pm = {prot_keys[i]: part for part, ids in ps.items() for i in ids}
        result = {p: [] for p in PARTITIONS}
        result["excluded"] = []
        for row in frame.itertuples():
            part = dm[row.drug_id]
            result[part if part == pm[row.protein_id] else "excluded"].append(int(row.pair_id))
    else:
        positional = group_partition(frame[columns[mode]].tolist(), seed, fractions)
        result = {
            part: frame.iloc[ids].pair_id.astype(int).tolist() for part, ids in positional.items()
        }
    result["metadata"] = {"mode": mode, "seed": seed, "fractions": list(fractions)}
    audit_dta_split(frame, result)
    return result


def audit_dta_split(frame, payload):
    mode = payload.get("metadata", {}).get("mode")
    if mode not in ("random", "cold_drug", "scaffold", "cold_target", "cold_both"):
        raise ValueError("Split metadata.mode must declare the intended generalization setting")
    parts = check_indices(payload, frame.pair_id, allow_excluded=mode == "cold_both")
    report = {
        "mode": mode,
        "pairs": {},
        "overlap": {},
        "excluded": len(payload.get("excluded", [])),
    }
    tables = {part: frame[frame.pair_id.isin(ids)] for part, ids in parts.items()}
    strict = {
        "random": [],
        "cold_drug": ["drug_id"],
        "scaffold": ["drug_id", "scaffold"],
        "cold_target": ["protein_id"],
        "cold_both": ["drug_id", "protein_id"],
    }[mode]
    for part, table in tables.items():
        report["pairs"][part] = len(table)
    for a, b in combinations(PARTITIONS, 2):
        for column in ("drug_id", "protein_id", "scaffold"):
            n = len(set(tables[a][column]) & set(tables[b][column]))
            report["overlap"][f"{a}/{b}/{column}"] = n
            if column in strict and n:
                raise ValueError(f"{mode} leakage: {n} overlapping {column} between {a}/{b}")
    return report
