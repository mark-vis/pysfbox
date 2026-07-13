"""Internal states and reaction equilibria (weak charges).

Mirrors Namics state.cpp / reaction.cpp: a segment type ("mon") may have
internal states (e.g. water W = {H2O, H3O, OH}; a weak acid A = {AH, AM})
that interconvert through reactions with equilibrium constants K = 10^-pK.
The states are annealed: locally equilibrated with the fields everywhere.
Conventions confirmed
against the compiled Namics (site-fraction K, water activity included,
products/reactants as written, stoichiometric powers).

This module owns the input side (State, Reaction, the equation parser) and
the bulk ionisation solve: given the reactions and the `alphabulk` anchors,
determine every bulk state fraction alpha_s^b. Namics pre-solves these
iteratively to a loose (~1e-6 relative) tolerance; PySFBox solves the same
equations to machine precision (log-space Newton) — see the theory note,
section 8.2.
"""

import math
import re

import numpy as np

from .inputreader import last


# ------------------------------------------------------------------ parsing
_TERM = re.compile(r"(\d+)\s*\(\s*([A-Za-z0-9_]+)\s*\)")


def parse_equation(text):
    """'1(AH) + 1(H2O) = 1(AM) + 1(H3O)' -> (reactants, products).

    Each side is a list of (stoichiometric coefficient, state name). The
    coefficient acts as a power on the site fraction in the equilibrium
    constant (so 2(H2O) contributes alpha_H2O**2).
    """
    sides = text.split("=")
    if len(sides) != 2:
        raise ValueError(f"reaction equation needs exactly one '=': '{text}'")

    def side(s):
        terms = _TERM.findall(s)
        # guard against silently dropped garbage: reassemble and compare
        stripped = re.sub(r"[+\s]", "", s)
        rebuilt = "".join(f"{n}({m})" for n, m in terms)
        if stripped != rebuilt.replace(" ", ""):
            raise ValueError(f"cannot parse reaction side '{s.strip()}' "
                             "(expected terms like 1(AH) + 2(H2O))")
        return [(int(n), name) for n, name in terms]

    return side(sides[0]), side(sides[1])


class State:
    """One internal state of a segment type (Namics `state : ...` block)."""

    def __init__(self, name, params):
        self.name = name
        for p in params:
            pk = p.replace(" ", "")
            if p not in ("mon", "valence", "alphabulk") \
                    and not (pk.startswith("chi_") or pk.startswith("chi-")):
                # in particular: no per-state epsilon exists (states use the
                # parent mon's, as in Namics) and no e.psi0 (states cannot
                # be frozen)
                raise ValueError(
                    f"state {name}: unknown parameter '{p}' (states take "
                    "mon, valence, alphabulk, and chi_X only)")
        self.mon = last(params, "mon")   # None for now; validated in System
        self.valence = float(last(params, "valence", 0.0))
        alphabulk = last(params, "alphabulk")
        self.alphabulk_given = alphabulk is not None
        self.alphabulk = float(alphabulk) if self.alphabulk_given else None
        # optional per-state overrides; anything not given is inherited from
        # the mon (confirmed against the compiled Namics: state chi overrides
        # the mon chi, otherwise the mon value applies)
        eps = last(params, "epsilon")
        self.epsilon = float(eps) if eps is not None else None
        self.chi = {}
        for p, v in params.items():
            pk = p.replace(" ", "")
            if pk.startswith("chi_") or pk.startswith("chi-"):
                self.chi[pk[4:]] = float(v[-1])


class Reaction:
    """One reaction (Namics `reaction : ...` block)."""

    def __init__(self, name, params):
        self.name = name
        eq = last(params, "equation")
        if eq is None:
            raise ValueError(f"reaction {name}: no equation given")
        pk = last(params, "pK")
        if pk is None:
            raise ValueError(f"reaction {name}: no pK given")
        self.reactants, self.products = parse_equation(eq)
        self.pK = float(pk)

    def check_balanced(self, state_mon):
        """Require the same total stoichiometry per mon on both sides.

        Then the mon volume fractions cancel from the equilibrium constant
        and the bulk state fractions are independent of the composition
        (theory note section 3). Unbalanced reactions couple the ionisation
        to the bulk composition; the shipped Namics examples are all
        balanced and PySFBox does not support the unbalanced case.
        """
        per_mon = {}
        for sign, side in ((-1, self.reactants), (+1, self.products)):
            for nu, sname in side:
                mon = state_mon[sname]
                per_mon[mon] = per_mon.get(mon, 0) + sign * nu
        unbalanced = [m for m, net in per_mon.items() if net != 0]
        if unbalanced:
            raise NotImplementedError(
                f"reaction {self.name}: not balanced per segment type "
                f"({', '.join(unbalanced)}); the bulk ionisation then "
                "depends on the composition — use the C++ Namics")


# ------------------------------------------------------- bulk ionisation
def solve_bulk_alphas(states_by_mon, reactions, tol=1e-14, maxiter=200):
    """Solve the bulk state fractions alpha_s^b to machine precision.

    states_by_mon: {mon_name: [State, ...]} for every multistate mon.
    Unknowns are ln(alpha_s) for every state without an alphabulk anchor;
    the equations are one normalisation sum_s alpha_s = 1 per mon and one
    equilibrium  sum_products nu*ln(alpha) - sum_reactants nu*ln(alpha)
    = -pK*ln(10) per reaction (site-fraction convention, water activity
    included — validated against the compiled Namics, theory note sec. 8).
    All equations except the normalisations are linear in ln(alpha), so the
    Newton iteration converges in a handful of steps.

    Returns {state_name: alpha}.
    """
    state_mon = {s.name: mon for mon, ss in states_by_mon.items() for s in ss}
    state_by_name = {s.name: s for ss in states_by_mon.values() for s in ss}
    for r in reactions:
        for nu, sname in r.reactants + r.products:
            if sname not in state_mon:
                raise ValueError(f"reaction {r.name}: unknown state {sname}")
        r.check_balanced(state_mon)
        # a reaction must conserve charge (Namics reaction.cpp:136-143)
        dv = (sum(nu * state_by_name[s].valence for nu, s in r.products)
              - sum(nu * state_by_name[s].valence for nu, s in r.reactants))
        if abs(dv) > 1e-10:
            raise ValueError(
                f"reaction {r.name}: not electroneutral (net valence "
                f"change {dv:+g})")

    anchored = {s.name: s.alphabulk
                for ss in states_by_mon.values() for s in ss
                if s.alphabulk_given}
    free = [s.name for ss in states_by_mon.values() for s in ss
            if not s.alphabulk_given]
    idx = {name: i for i, name in enumerate(free)}

    n_eq = len(states_by_mon) + len(reactions)
    if n_eq != len(free):
        raise ValueError(
            f"bulk ionisation: {len(free)} unknown state fractions but "
            f"{len(states_by_mon)} normalisations + {len(reactions)} "
            "reactions; give exactly enough 'state : X : alphabulk' anchors "
            "to close the system (e.g. one for H3O to set the pH)")

    # initial guess: split what the anchors leave over equally per mon
    x = np.empty(len(free))
    for mon, ss in states_by_mon.items():
        rest = 1.0 - sum(anchored.get(s.name, 0.0) for s in ss)
        n_free = sum(1 for s in ss if not s.alphabulk_given)
        for s in ss:
            if not s.alphabulk_given:
                x[idx[s.name]] = math.log(max(rest, 1e-30) / n_free)

    ln10 = math.log(10.0)

    def lnalpha(name):
        return x[idx[name]] if name in idx else math.log(anchored[name])

    for _ in range(maxiter):
        res = np.zeros(n_eq)
        jac = np.zeros((n_eq, len(free)))
        row = 0
        for mon, ss in states_by_mon.items():
            res[row] = sum(math.exp(lnalpha(s.name)) for s in ss) - 1.0
            for s in ss:
                if s.name in idx:
                    jac[row, idx[s.name]] = math.exp(lnalpha(s.name))
            row += 1
        for r in reactions:
            res[row] = r.pK * ln10
            for sign, side in ((+1, r.products), (-1, r.reactants)):
                for nu, sname in side:
                    res[row] += sign * nu * lnalpha(sname)
                    if sname in idx:
                        jac[row, idx[sname]] += sign * nu
            row += 1
        if np.max(np.abs(res)) < tol:
            break
        try:
            x -= np.linalg.solve(jac, res)
        except np.linalg.LinAlgError:
            raise ValueError(
                "bulk ionisation: singular equation system — the reactions "
                "and alphabulk anchors do not determine all state fractions "
                "independently") from None
    else:
        raise RuntimeError(
            "bulk ionisation solve did not converge; check that the pK "
            "values and alphabulk anchors are mutually consistent")

    out = dict(anchored)
    for name in free:
        out[name] = math.exp(x[idx[name]])
    return out
