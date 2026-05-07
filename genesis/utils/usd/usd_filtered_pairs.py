"""UsdPhysicsFilteredPairsAPI -> contype/conaffinity bit-pack via Z3.

Walks every prim that applies UsdPhysicsFilteredPairsAPI and turns
the resulting "no-contact" pair set into per-prim (contype,
conaffinity) bitmasks compatible with MuJoCo's pairwise rule
  (ct_a & ca_b) | (ct_b & ca_a) != 0.

This is the same bit-pack Genesis already uses for MJCF
`<contact><exclude>` (genesis/utils/mjcf.py:677-701) — a Z3 SAT
solver that searches K = 1..31 for the minimum bitwidth that
satisfies the input pair-collision matrix. We reuse the algorithm
verbatim here so USD imports get the same expressivity as the MJCF
path (no 31-group structural cap; arbitrary pair masks).

Usage:

    from .usd_filtered_pairs import build_collision_filter_bits
    bits = build_collision_filter_bits(stage)
    for prim_path, (ct, ca) in bits.items(): ...

Bodies absent from the returned dict are not in any filtered-pair
relationship; the caller falls back to its own default (typically
1/1 collider, 0/0 visual).
"""
from __future__ import annotations

from pxr import Usd, UsdPhysics
import z3

import genesis as gs


def _solve_compatible_bitmasks(N: int, invalid_set: set):
    """Z3 SAT search for per-index (contype, conaffinity) bitmask
    pairs. `invalid_set` contains frozenset((i, j)) for each pair
    that must NOT collide; all other (i < j) pairs must collide.

    Returns a list of N (ct, ca) tuples on success, or None if
    UNSAT for every K in [1, 31].
    """
    for K in range(1, 32):
        s = z3.Solver()
        ct_bits = [[z3.Bool(f"ct_{i}_{b}") for b in range(K)] for i in range(N)]
        ca_bits = [[z3.Bool(f"ca_{i}_{b}") for b in range(K)] for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                cond1 = z3.Or([z3.And(ct_bits[i][b], ca_bits[j][b])
                               for b in range(K)])
                cond2 = z3.Or([z3.And(ct_bits[j][b], ca_bits[i][b])
                               for b in range(K)])
                if frozenset((i, j)) in invalid_set:
                    s.add(z3.Not(cond1), z3.Not(cond2))
                else:
                    s.add(z3.Or(cond1, cond2))
        if s.check() != z3.sat:
            continue
        model = s.model()
        out = []
        for i in range(N):
            ct = sum((1 << b) if z3.is_true(model[e]) else 0
                     for b, e in enumerate(ct_bits[i]))
            ca = sum((1 << b) if z3.is_true(model[e]) else 0
                     for b, e in enumerate(ca_bits[i]))
            out.append((ct, ca))
        return out
    return None


def build_collision_filter_bits(stage: Usd.Stage) -> dict[str, tuple[int, int]]:
    """Walk the stage's PhysicsFilteredPairsAPI prims and return a
    prim-path-string -> (contype, conaffinity) map.

    Empty when no prim has the API applied; caller falls back to
    historical defaults."""
    out: dict[str, tuple[int, int]] = {}

    # Discover all bodies that author the API; collect their target
    # path lists. Pair filtering is mutual under UsdPhysics so we
    # symmetrise.
    pair_owners: dict[str, list[str]] = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            continue
        api = UsdPhysics.FilteredPairsAPI(prim)
        rel = api.GetFilteredPairsRel()
        if not rel:
            continue
        targets = [str(t) for t in rel.GetTargets()]
        if targets:
            pair_owners[str(prim.GetPath())] = targets

    if not pair_owners:
        return out

    # Universe: all bodies that participate in any filtered pair
    # (either side) need a slot in the SAT problem. Keep a stable
    # ordering for deterministic bit assignment across runs.
    universe: list[str] = []
    seen: set[str] = set()
    for src, tgts in pair_owners.items():
        for p in (src, *tgts):
            if p not in seen:
                seen.add(p)
                universe.append(p)

    if len(universe) > 256:
        # Z3 grows roughly O(N^2 * K) constraints. 256 bodies with
        # K = 31 is already heavy; warn and continue.
        gs.logger.warning(
            f"FilteredPairsAPI bit-pack: {len(universe)} bodies; SAT may "
            f"be slow.")

    idx = {p: i for i, p in enumerate(universe)}
    invalid_set: set = set()
    for src, tgts in pair_owners.items():
        i = idx[src]
        for t in tgts:
            if t not in idx:
                continue
            j = idx[t]
            if i == j:
                continue
            invalid_set.add(frozenset((i, j)))

    bits = _solve_compatible_bitmasks(len(universe), invalid_set)
    if bits is None:
        gs.logger.warning(
            "FilteredPairsAPI bit-pack: no compatible (contype, conaffinity) "
            "bitmask assignment within K<32; falling back to (1, 1) for all.")
        for p in universe:
            out[p] = (1, 1)
        return out

    # Reserve bit 0 for "default world" so packed bodies still collide
    # with non-grouped prims (which keep ct=1, ca=1). Shift the SAT
    # solution up by one bit and OR bit 0 into conaffinity. The pairwise
    # rule restricted to the universe is preserved exactly:
    #   (2ct & (2ca'|1)) | (2ct' & (2ca|1))
    #   = 2((ct & ca') | (ct' & ca))
    # so it's zero iff the original was zero. Pair vs default (1, 1)
    # always evaluates to (2ct & 1) | (1 & (2ca|1)) = 1.
    for i, p in enumerate(universe):
        ct, ca = bits[i]
        out[p] = (ct << 1, (ca << 1) | 1)
    return out
