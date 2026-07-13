"""Run a Namics input file: start blocks, var scans, output writing."""

import os

import numpy as np

from .inputreader import (get_blocks, last, read_input, set_value)
from .output import kal_path, pro_path, write_kal, write_pro
from .system import System


def _num(s):
    """Parse a Namics numeric field the way Namics' Input::Get_Real does:
    a value is numeric only if it starts with a digit (or '-' then a digit).
    Anything else (e.g. the typo 'e-10') is unparseable -> returns None so the
    caller falls back to its default, exactly like Namics."""
    if s is None:
        return None
    t = s.strip()
    head = t[1:2] if t[:1] == "-" else t[:1]
    if not head.isdigit():
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _output_specs(settings, kind):
    """Lines like `pro : mol : pol : phi` -> [('mol','pol','phi'), ...].
    A '*' object name is expanded to every mol / mon in definition order,
    exactly as Namics does."""
    mol_names = [name for name, _ in get_blocks(settings, "mol")]
    mon_names = [name for name, _ in get_blocks(settings, "mon")]
    specs = []
    for (key, name), params in settings.items():
        if key != kind:
            continue
        for param, values in params.items():
            for v in values:
                if param == "*" and name == "mol":
                    specs.extend((name, nm, v) for nm in mol_names)
                elif param == "*" and name == "mon":
                    specs.extend((name, nm, v) for nm in mon_names)
                else:
                    specs.append((name, param, v))
    return specs


def _pro_write_bounds(settings):
    """`output : pro : write_bounds : true` -> also emit the fjc ghost layers
    in the .pro (Namics default is false = interior only)."""
    for name, params in get_blocks(settings, "output"):
        if name == "pro":
            v = last(params, "write_bounds")
            if v is not None:
                return str(v).strip().lower() in ("true", "1", "yes")
    return False


def _newton_options(settings):
    opts = {}
    for name, params in get_blocks(settings, "newton"):
        tol = _num(last(params, "tolerance"))
        if tol is not None:
            opts["tolerance"] = tol
        ilim = _num(last(params, "iterationlimit"))
        if ilim is not None:
            opts["iterationlimit"] = int(ilim)
        dmax = _num(last(params, "deltamax"))
        if dmax is not None:
            opts["deltamax"] = dmax
        m = _num(last(params, "m"))
        if m is not None:
            opts["m_anderson"] = int(m)
        method = last(params, "method")
        if method:
            if method == "pseudohessian":
                # the Namics default: pseudohessian quasi-Newton as primary
                opts["method"] = "pseudohessian"
            elif method == "hessian":
                # Namics full-Hessian mode: recomputed every iteration
                opts["method"] = "hessian"
            elif method in ("DIIS", "diis", "Picard", "picard", "LBFGS"):
                # Anderson-family / alternative methods in Namics; PySFBox
                # maps them onto its Anderson-accelerated Picard solver
                opts["method"] = "anderson"
            else:
                print(f"  note: newton method '{method}' unknown; using the "
                      "default solver cascade")
        eng = last(params, "engine")
        if eng and eng.strip().lower() not in ("numpy", "np"):
            print(f"  note: newton engine '{eng}' unknown; "
                  "using the NumPy residual")
    return opts


# target observables a `var : ITEM-NAME : PROP : value` line can request
# (Namics variate.cpp KEYS): sys-level scalars and per-molecule scalars.
_SYS_TARGETS = ("grand_potential", "free_energy", "Laplace_pressure")
_MOL_TARGETS = ("mu", "theta", "n", "phibulk")


def _var_roles(settings):
    """Classify the `var` blocks into (scan, search, target) roles, exactly
    as Namics' Variate::CheckInput does. A block carrying `scan` is a scan; a
    block carrying `search` is the search variable (mol only); a block naming
    a target observable with a value is the target. Namics allows at most one
    of each (namics.cpp n_search/n_target limits). Returns
    (scan_block_or_None, search_or_None, target_or_None) where
    search = (item, name, prop) and target = (item, name, prop, value)."""
    scan = search = target = None
    for name, params in get_blocks(settings, "var"):
        item, _, obj = name.partition("-")
        item, obj = item.strip(), obj.strip()
        # a single block may carry more than one role: Namics puts the search
        # and its target on different objects (distinct blocks), but a mol
        # self-target (search theta, target that same mol's phibulk) merges
        # into one block by name -- so the roles are detected independently.
        if "scan" in params:
            scan = (name, params)
        if "search" in params:
            if search is not None:
                raise ValueError("more than one 'var : ... : search' block "
                                 "(Namics allows one)")
            search = (item, obj, last(params, "search").strip())
        cands = _SYS_TARGETS if item == "sys" else _MOL_TARGETS
        for prop in cands:
            if prop in params:
                if target is not None:
                    raise ValueError("more than one 'var' target block "
                                     "(Namics allows one)")
                tv = last(params, prop)
                # a mu target naming another molecule (mol-Y) is Namics'
                # 'eq_to_mu' equilibration -- not supported; name it clearly
                # rather than let float('mol-Y') throw a bare ValueError
                if prop == "mu" and not _num(tv):
                    raise NotImplementedError(
                        f"var target '{prop} : {tv}': equating a molecule's "
                        "mu to another molecule (eq_to_mu), and the "
                        "equate-to-solvent / balance-membrane searches, are "
                        "not supported (use the C++ Namics)")
                target = (item, obj, prop, float(tv))
                break
    if search is not None and target is None:
        raise ValueError("lonely search: a 'var : ... : search' needs a "
                         "target block, e.g. 'var : sys-NN : "
                         "grand_potential : 0' (as in Namics)")
    if target is not None and search is None:
        raise ValueError("lonely target: a 'var' target needs a "
                         "'var : mol-X : search : theta' (as in Namics)")
    return scan, search, target


def _super_options(settings):
    """Super-iteration (search) controls, read from the newton block like
    Namics solve_scf.cpp:156-180. Defaults track Namics:
    super_tolerance = 10*tolerance, super_deltamax = 0.5,
    super_iterationlimit = max(iterationlimit/10, 30)."""
    tol, ilim = 1e-7, 1000
    stol = sdmax = silim = None
    for _, params in get_blocks(settings, "newton"):
        v = _num(last(params, "tolerance"))
        if v is not None:
            tol = v
        v = _num(last(params, "iterationlimit"))
        if v is not None:
            ilim = int(v)
        v = _num(last(params, "super_tolerance"))
        if v is not None:
            stol = v
        v = _num(last(params, "super_deltamax"))
        if v is not None:
            sdmax = v
        v = _num(last(params, "super_iterationlimit"))
        if v is not None:
            silim = int(v)
    return {"tolerance": stol if stol is not None else 10.0 * tol,
            "deltamax": sdmax if sdmax is not None else 0.5,
            "iterationlimit": silim if silim is not None else max(ilim // 10,
                                                                  30)}


def _var_plan(settings):
    """The scan schedule, if a `var` scan block is present:
    (key, name, prop, values). Ignores search/target blocks."""
    scan, _, _ = _var_roles(settings)
    if scan is None:
        return None
    blockname, params = scan
    key, _, name = blockname.partition("-")
    key, name = key.strip(), name.strip()   # Namics writes 'lat- flat'
    prop = last(params, "scan")
    end = float(last(params, "end_value"))
    if key == "alias":
        cur = float(last(settings[("alias", name)], "value"))
    elif prop.endswith("-value"):
        cur = float(last(settings[("alias", prop[:-6])], "value"))
        key, name, prop = "alias", prop[:-6], "value"
    else:
        cur = float(last(settings.get((key, name), {}), prop, 0))
    if last(params, "scale", "").strip().lower() == "exponential":
        # Namics exponential scans (e.g. state alphabulk titrations,
        # state.cpp:268-304): 'steps' means steps PER DECADE, and the
        # values interpolate log10 -- verified against the oracle
        # (steps 12 over 6 decades -> 73 rows)
        if cur <= 0 or end <= 0:
            raise ValueError("var : scale : exponential needs positive "
                             "start and end values")
        per_decade = float(last(params, "steps", 1))
        decades = abs(np.log10(end / cur))
        n = max(int(round(per_decade * decades)), 1)
        values = [cur * 10.0 ** (i * np.log10(end / cur) / n)
                  for i in range(n + 1)]
    else:
        step = float(last(params, "step", 1))
        n_steps = int(round((end - cur) / step)) + 1
        values = [cur + i * step for i in range(max(n_steps, 1))]
    return key, name, prop, values


def _set_search(settings, search, X):
    """Write the current super-iteration trial value X into the settings so
    the next System build picks it up. The search variable is a molecule's
    theta / n / phibulk (Namics Molecule::PutValue). theta and n are linked
    (theta = n*chainlength), so setting one must clear the other."""
    item, obj, prop = search
    if item != "mol":
        raise NotImplementedError(
            f"search variable '{item}-{obj} : {prop}' is not supported; "
            "PySFBox searches a molecule theta/n/phibulk (needs the full "
            "Namics for lat/mon/state/reaction search variables)")
    if prop not in ("theta", "n", "phibulk"):
        raise NotImplementedError(
            f"search : {prop} is not supported (searchable molecule "
            "quantities are theta, n, phibulk)")
    molp = settings.get(("mol", obj))
    if molp is None:
        raise ValueError(f"var search: unknown mol '{obj}'")
    if prop in ("theta", "n"):        # linked: keep only the one we search
        molp.pop("theta", None)
        molp.pop("n", None)
    set_value(settings, "mol", obj, prop, repr(float(X)))


def _search_start(settings, search):
    """The starting value of the search variable (Namics uses the declared
    value, Variate::PutVarInfo Var_start_search_value). Requires the molecule
    to declare that quantity (theta/n/phibulk) explicitly."""
    item, obj, prop = search
    molp = settings.get(("mol", obj), {})
    v = last(molp, prop)
    if v is None and prop == "theta":          # theta implied by n*N
        n = last(molp, "n")
        comp = last(molp, "composition")
        if n is not None and comp is not None:
            from .model import parse_composition
            try:
                N = sum(c for _, c in parse_composition(comp))
                return float(n) * N
            except Exception:
                pass
    if v is None:
        raise ValueError(
            f"var search: mol '{obj}' must declare an initial '{prop}' "
            "(the search starts from it, as in Namics)")
    return float(v)


def _target_error(system, target):
    """Signed target error for the regula-falsi search, matching the Namics
    sign conventions (System::GetError / Molecule::GetError) so the
    false-position bracketing behaves the same. Returns (error, value)."""
    item, obj, prop, val = target
    if item == "sys":
        if prop == "grand_potential":
            gp = system.grand_potential()
            return -(gp - val), gp
        if prop == "free_energy":
            fe = system.free_energy()
            return fe - val, fe
        raise NotImplementedError(
            "var target 'Laplace_pressure' needs the sys:constraint:delta "
            "membrane-balance machinery, which PySFBox does not have "
            "(use the C++ Namics)")
    if item == "mol":
        prop_lookup = "Mu" if prop == "mu" else prop
        kind, cur = system.get_value("mol", obj, prop_lookup)
        if cur is None:
            raise ValueError(f"var target: mol '{obj}' has no '{prop}'")
        err = (cur / val - 1.0) if val != 0 else cur
        if prop == "phibulk":
            err = -err
        return err, cur
    raise NotImplementedError(
        f"var target '{item}-{obj} : {prop}' is not supported")


def _regula_falsi(resid, x_start, tol, deltamax, itlimit, verbose, label):
    """Scalar root find for the super-iteration: drive resid(X) (the signed
    target error) to zero by adjusting the search value X. resid(X) runs a
    full SCF at X. Returns (X_root, super_iterations).

    Namics' SFNewton::iterate_RF is a bare damped secant in multiplicative
    units of the start; it stalls when the root is far from the start or the
    two seed points fall on the same side. PySFBox uses a robust
    bracket-then-Illinois false position instead: it first expands a
    positive-definite bracket geometrically (the search variables -- theta,
    n, phibulk -- are all positive), then Illinois-weighted false position
    inside it. The root is unique, so this lands on the same solution Namics'
    RF would; only the path differs (and the oracle's search examples need
    the sys:constraint:delta machinery PySFBox lacks, so there is no
    trajectory to match anyway)."""
    if x_start == 0:
        raise ValueError("var search: the initial search value is 0; the "
                         "search scales by it (declare a nonzero start)")

    def rf(X):
        # a failed inner SCF while marching the search variable almost always
        # means the target is unreachable/non-monotone, not that the inner
        # deltamax is wrong -- say so
        try:
            return resid(X)
        except RuntimeError as e:
            raise RuntimeError(
                f"{label} search: the inner SCF did not converge at search "
                f"value {X:.6g}; the target is likely unreachable or "
                "non-monotone in the search variable -- check the target "
                f"value (underlying: {e})") from e

    grow = 1.0 + max(abs(deltamax), 1e-3)
    f0 = rf(x_start)
    it = 1
    if verbose:
        print(f"    {label} super-it 1: search = {x_start:.6g}, "
              f"target error = {f0:.3e}")
    if abs(f0) <= tol:
        return x_start, it

    # probe one step each way, then expand DOWNHILL (toward smaller |error|)
    # -- the search variables are positive, so multiplicative steps keep them
    # positive; committing to the downhill direction avoids marching into the
    # stiff, hard-to-converge large-value region.
    fu = rf(x_start * grow)
    it += 1
    if abs(fu) <= tol:
        return x_start * grow, it
    if fu * f0 < 0.0:
        bracket = (x_start, f0, x_start * grow, fu)
    else:
        fd = rf(x_start / grow)
        it += 1
        if abs(fd) <= tol:
            return x_start / grow, it
        if fd * f0 < 0.0:
            bracket = (x_start / grow, fd, x_start, f0)
        else:
            direction = grow if abs(fu) < abs(fd) else 1.0 / grow
            x, f = (x_start * grow, fu) if direction == grow \
                else (x_start / grow, fd)
            bracket = None
            while it < itlimit:
                xn = x * direction
                fn = rf(xn)
                it += 1
                if abs(fn) <= tol:
                    return xn, it
                if fn * f < 0.0:
                    bracket = (x, f, xn, fn)
                    break
                x, f = xn, fn
    if bracket is None:
        raise RuntimeError(
            f"{label} search could not bracket the target in {it} "
            "super-iterations (the target may be unreachable, or the target "
            "observable non-monotone in the search variable); check the "
            "target value")

    a, fa, b, fb = bracket
    while it < itlimit:
        c = b - fb * (b - a) / (fb - fa)          # false-position step
        fc = rf(c)
        it += 1
        if verbose and (it <= 5 or it % 5 == 0):
            print(f"    {label} super-it {it}: search = {c:.6g}, "
                  f"target error = {fc:.3e}")
        if abs(fc) <= tol or abs(b - a) <= 1e-13 * abs(c):
            return c, it
        if fc * fb < 0.0:
            a, fa = b, fb
        else:
            fa *= 0.5                              # Illinois down-weight
        b, fb = c, fc
    raise RuntimeError(
        f"{label} search did not converge in {itlimit} super-iterations "
        f"(target error {fb:.2e} > {tol:.1e}); raise super_iterationlimit "
        "or give a better initial value")


def _remap_layers(x_old, n_seg, M_old, M_new):
    """Adapt a converged potential stack to a changed layer count, as an
    initial guess only: keep the lower-wall half from the lower end and the
    upper-wall half from the upper end (so both surface regions survive a
    two-wall compression), pad the middle with u = 0 (bulk) when growing.
    Works in refined (fjc-internal) units since M includes the ghosts.

    The iteration vector is a stack of equal-length (M) blocks: one potential
    field per iteration species, plus a trailing psi block when the system is
    CHARGED (system.n_var adds one more M). All blocks are spatial profiles, so
    every block -- u fields AND psi -- is remapped the same way; the number of
    blocks is inferred from the vector length (n_seg is only a hint, and would
    miss the psi block for charged/weak systems -- the bug this guards). Returns
    None if the length is not an integer number of M-blocks, so the caller
    falls back to a cold start rather than crashing."""
    if M_old <= 0 or x_old.size % M_old != 0:
        return None
    n_blocks = x_old.size // M_old
    Xo = x_old.reshape(n_blocks, M_old)
    Xn = np.zeros((n_blocks, M_new))
    k = min(M_old, M_new)
    lo = k // 2
    hi = k - lo
    Xn[:, :lo] = Xo[:, :lo]
    Xn[:, M_new - hi:] = Xo[:, M_old - hi:]
    return Xn.ravel()


def run_file(path, verbose=True):
    """Run all calculations in a Namics input file; writes .kal/.pro next
    to the input file. Returns the last System (for interactive use)."""
    calculations = read_input(path)
    base = os.path.splitext(path)[0]
    kal_started = False
    system = None
    n_starts = len(calculations)

    # Warm-start continuously through the whole file -- across var steps AND
    # across `start` blocks -- exactly like Namics (multiple starts run
    # warm-started in one process). This is load-bearing for continuation
    # scans (e.g. ramping chi or a co-solvent into a hard regime) and for
    # symmetry-broken states (a freed collapsed layer must inherit the
    # previous profile, not restart from the trivial uniform solution).
    # Two refinements on top of the plain previous-solution seed:
    # - when only n_layers changed (same segments, same fjc), the previous
    #   potentials are remapped onto the new grid instead of cold-starting
    #   (load-bearing for wall-separation scans);
    # - within one var scan, from the third step on, a secant extrapolation
    #   x0 = 2 x_k - x_{k-1} predicts the next step (parameter ramps).
    # Both only shape the initial guess; they cannot change what a converged
    # solution satisfies.
    x_prev = None
    x_scan_prev = None          # solution of the step before x_prev, same scan
    seg_prev, M_prev, fjc_prev = None, 0, 0
    for start_i, settings in enumerate(calculations, 1):
        kal_specs = _output_specs(settings, "kal")
        pro_specs = _output_specs(settings, "pro")
        pro_bounds = _pro_write_bounds(settings)
        newton = _newton_options(settings)
        plan = _var_plan(settings)
        has_var = plan is not None
        steps = plan[3] if has_var else [None]

        x_scan_prev = None
        _, search, target = _var_roles(settings)
        super_opts = _super_options(settings) if search is not None else None

        for subl, value in enumerate(steps):
            if has_var:
                key, name, prop, _ = plan
                v = int(value) if float(value).is_integer() else value
                set_value(settings, key, name, prop, v)
                tag = f" [{key}:{name}:{prop} = {v}]"
            else:
                tag = ""

            def _solve_with(built, seed):
                """Solve an already-built System from `seed`, retrying once
                from the previous converged solution if a fancy seed fails."""
                nv = built.n_var()
                s = seed if (seed is not None and seed.size == nv) else None
                try:
                    return built.solve(x0=s, **newton)
                except RuntimeError:
                    fb = (x_prev if x_prev is not None
                          and x_prev.size == nv else None)
                    if s is None and fb is None:
                        raise
                    return built.solve(x0=fb, **newton)

            system = System(settings)
            seg_now = tuple(s.name for s in system.it_species)
            M_now, fjc_now = system.lat.M, system.lat.fjc
            if x_prev is None:
                seed = None
            elif seg_now != seg_prev or fjc_now != fjc_prev:
                seed = None                   # different problem structure
                x_scan_prev = None
            elif x_prev.size != system.n_var():
                seed = _remap_layers(x_prev, len(seg_now), M_prev, M_now)
                x_scan_prev = None
            else:
                seed = x_prev
                if (has_var and subl >= 2 and x_scan_prev is not None
                        and x_scan_prev.size == x_prev.size):
                    seed = 2.0 * x_prev - x_scan_prev
            if seed is None and start_i == 1 and subl == 0:
                # analytic initial guess (sys : initial_guess : ...), applied
                # to the first calculation only, exactly like Namics
                seed = system.initial_guess()

            if search is None:
                try:
                    x, it, err = _solve_with(system, seed)
                except RuntimeError as e:
                    print(f"start {start_i}{tag}: {e}")
                    raise
            else:
                # super-iteration: a scalar regula falsi drives the target
                # observable to its value by adjusting the search variable;
                # every residual evaluation is a full SCF solve (Namics
                # SuperIterate). Each solve warm-starts from the previous one.
                cap = {}
                warm = [seed]

                def resid(X):
                    _set_search(settings, search, X)
                    built = System(settings)
                    # after the first, warm[0] is the previous search iterate
                    # (a continuation)
                    xx, itc, errc = _solve_with(built, warm[0])
                    warm[0] = xx
                    cap["sys"], cap["x"] = built, xx
                    cap["it"], cap["err"] = itc, errc
                    return _target_error(built, target)[0]

                try:
                    x_root, super_it = _regula_falsi(
                        resid, _search_start(settings, search),
                        super_opts["tolerance"], super_opts["deltamax"],
                        super_opts["iterationlimit"], verbose,
                        f"start {start_i}{tag}")
                except RuntimeError as e:
                    print(f"start {start_i}{tag}: {e}")
                    raise
                system, x = cap["sys"], cap["x"]
                it, err = cap["it"], cap["err"]
                if verbose:
                    print(f"start {start_i}{tag}: search "
                          f"{search[1]}:{search[2]} -> {x_root:.6g} "
                          f"in {super_it} super-iterations")
            x_scan_prev = x_prev if has_var else None
            x_prev = x
            seg_prev, M_prev, fjc_prev = seg_now, M_now, fjc_now
            if verbose:
                phit_dev = np.abs(system.phitot * system.ksam
                                  + (1 - system.ksam) - 1).max()
                print(f"start {start_i}{tag}: converged in {it} iterations, "
                      f"max|g| = {err:.1e}, max|phi_T-1| = {phit_dev:.1e}")
            if kal_specs:
                write_kal(kal_path(base), kal_specs, system,
                          new_file=not kal_started)
                kal_started = True
            if pro_specs:
                p = pro_path(base, n_starts, start_i, subl, has_var)
                write_pro(p, pro_specs, system, write_bounds=pro_bounds)
    return system
