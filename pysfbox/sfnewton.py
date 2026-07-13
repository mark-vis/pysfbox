"""Pseudohessian quasi-Newton solver -- a Python/NumPy translation of Namics
``sfnewton.cpp`` (the Scheutjens variable-metric method).

    Original algorithm and C/C++ source:
        Author: Jan Scheutjens (1947-1992), Wageningen Agricultural University, NL.
        C  Copyright (1980) (1981-1989)  Wageningen Agricultural University, NL.
        C++ translation: Peter Barneveld; masked/vector adaptation: Frans
        Leermakers, Wageningen Agricultural University, NL.
        C  Copyright (2018)  Wageningen University, NL.
        *NO PART OF THIS WORK MAY BE REPRODUCED, EITHER ELECTRONICALLY OR OTHERWISE*

    Python/NumPy translation for PySFBox (2026), by Mark Vis, TU/e.
    Distributed only with permission of the copyright holders; the original
    copyright and the reproduction notice above are carried over unchanged.

The method keeps an approximate Hessian H in Gill-Murray LDL^T factored form
(references in sfnewton.cpp: Gill & Murray, NPL reports 1972-76; Shanno 1978).
It is a variable-metric (quasi-Newton) scheme: H is *updated* by rank-1 secant
updates each step (``newhessian``/``updatpos``), never recomputed, and the
Newton direction H p = -g is found by forward/back substitution on the factors
(``gausa``/``gausb``). Because the factorization is maintained positive-definite,
the direction is always a descent direction -- which is what lets it converge
the stiff / near-singular problems (poor-solvent collapse, co-solvent) that
plain Anderson mixing cannot.

Faithful-translation notes:
- Storage matches the C code exactly: ``h`` is a flat length-nvar*nvar array in
  the C layout (``h[a*nvar + b]``). Inside each routine a zero-copy 2-D view
  ``H = h.reshape(nvar, nvar)`` maps every literal C flat index mechanically
  (``h[a*nvar+b] == H[a, b]``; the C gathers ``l[I + J*nvar]`` over ``J`` are
  the column slices ``H[:, I]``; the diagonal ``h[i + nvar*i]`` is the stride
  view ``h[::nvar+1]``). The per-element arithmetic and its order are the
  C original's; only the innermost loops are replaced by NumPy slice
  operations on identical operands (no algorithm change).
- The full-numerical-Hessian option (``numhessian`` + ``decompos``) is
  translated and used as a one-shot anchor (``iterate(anchor_full=True)``);
  the walking-backwards Hessian reset and deltamax decay (from
  Solve_scf::inneriteration) are kept -- they are the load-bearing rescue.
"""

import numpy as np


class SFNewton:
    def __init__(self, residual_full, mask, accel=None):
        """residual_full(x_full) -> g_full over all variables; mask is a boolean
        array over those variables selecting the ones actually iterated (drops
        structurally-zero residual sites: ghost layers, frozen surfaces -- the
        role of Namics' ``filter``/``mask``).

        accel: optional dict overriding the dense linear-algebra kernels
        ("updatpos", "gausa", "gausb"; same signatures and in-place
        contracts) and/or providing an exact Jacobian ("hessian":
        callable(x_reduced) -> dg/dx matrix, used by numhessian in place of
        the finite-difference sweep). The pure-NumPy defaults below are the
        reference implementation."""
        self._residual_full = residual_full
        self._hessian_fn = None
        self.full_hessian = False
        if accel:
            for name in ("updatpos", "gausa", "gausb"):
                if name in accel:
                    setattr(self, name, accel[name])
            self._hessian_fn = accel.get("hessian")
        self.mask = np.asarray(mask, dtype=bool)
        self.idx = np.where(self.mask)[0]
        self._xfull = np.zeros(self.mask.size)
        # solver constants (sfnewton.cpp defaults)
        self.nbits = 52
        self.linesearchlimit = 20
        self.max_accuracy_for_hessian_scaling = 0.1
        self.resetHessianCriterion = 1e5
        self.smallAlpha = 1e-5
        self.maxNumSmallAlpha = 50
        self.ignore_newton_direction = True
        # state
        self.iterations = 0
        self.normg = np.inf

    # ---- residual on the reduced (masked) variable set (cf. COMPUTEG) --------
    def residuals(self, x):
        self._xfull[:] = 0.0
        self._xfull[self.idx] = x
        g = self._residual_full(self._xfull)
        return g[self.idx]

    # ---- linear-algebra primitives (translated from sfnewton.cpp) -----------
    @staticmethod
    def norm2(x):
        return float(np.sqrt(np.dot(x, x)))

    @staticmethod
    def signdeterminant(h, nvar):
        d = h[::nvar + 1]                        # diagonal h[i + nvar*i], a view
        return -1 if (np.sum(d < 0) % 2) else 1

    @staticmethod
    def startderivatives(h, g, nvar):
        diagonal = 1.0 + SFNewton.norm2(g)
        h[:] = 0.0
        h[::nvar + 1] = diagonal                 # h[i + nvar*i] = diagonal

    def resethessian(self, h, g, nvar):
        self.trouble = 0
        self.startderivatives(h, g, nvar)
        self.resetiteration = self.iterations

    @staticmethod
    def gausa(l, dup, g, nvar):
        """Forward substitution L dup = -g  (dup output). C: dupa[i] = -ga[i] -
        sum_{j<i} l[(i-1)+(j-1)*nvar] dup[j]; the flat gather l[I + J*nvar],
        J = 0..I-1, is column I of the factors: H[:I, I]."""
        H = l.reshape(nvar, nvar)                # zero-copy view of the C layout
        for I in range(nvar):
            dup[I] = -g[I] - np.dot(H[:I, I], dup[:I])

    @staticmethod
    def gausb(du, p, nvar):
        """Back substitution D L^T p = dup (p in/out). C: p[i] = p[i]/du[(i-1)+
        (i-1)*nvar] - sum_{j>i} du[(i-1)+(j-1)*nvar] p[j]; the flat gather is
        the column tail H[I+1:, I]."""
        H = du.reshape(nvar, nvar)
        for I in range(nvar - 1, -1, -1):
            p[I] = p[I] / H[I, I] - np.dot(H[I + 1:, I], p[I + 1:])

    @staticmethod
    def multiply(v, alpha, h, w, nvar):
        """v = alpha * H w with H in factored form (cf. sfnewton multiply).
        The C gathers h[i + nvar*j] over j are column i: H[:, i]."""
        H = h.reshape(nvar, nvar)
        x = np.empty(nvar)
        for i in range(nvar):
            x[i] = (np.dot(w[i + 1:], H[i + 1:, i]) + w[i]) * H[i, i]
            v[i] = alpha * (np.dot(x[:i], H[:i, i]) + x[i])

    @staticmethod
    def updatpos(l, w, v, nvar, alpha):
        """Positive rank-1 update of the LDL^T factors (translated updatpos).
        Modifies l (factors) and w, v (scratch) in place. C per-row order for
        j > i is: waj = w[j]; w[j] -= w[i]*row[j]; row[j] = row[j]*d + c*waj
        (and the same for v/column) -- i.e. the new row/column are built from
        the PRE-update row/column and w/v. Computing new_row/new_col first from
        those same pre-update operands is the identical arithmetic without the
        per-row copies. (A fully vectorised O(n^2)-temporary reformulation was
        measured SLOWER at n~600 -- memory traffic beats dispatch savings --
        so the per-row loop stays.)"""
        H = l.reshape(nvar, nvar)
        for I in range(nvar):
            vai = v[I]
            wai = w[I]
            d = H[I, I]
            bb = d + (alpha * wai) * vai
            H[I, I] = bb
            d /= bb
            c = vai * alpha / bb
            b = wai * alpha / bb
            alpha *= d
            if I + 1 < nvar:
                row = H[I, I + 1:]               # views into the factors
                col = H[I + 1:, I]
                new_row = row * d + c * w[I + 1:]    # pre-update row, w
                new_col = col * d + b * v[I + 1:]    # pre-update col, v
                w[I + 1:] -= wai * row               # pre-update row
                v[I + 1:] -= vai * col               # pre-update col
                row[:] = new_row
                col[:] = new_col

    @staticmethod
    def updateneg(l, w, nvar, alpha):
        """Negative rank-1 update (translated updateneg); only reached for the
        nvar==1 branch of newhessian in practice."""
        dmin = 1.0 / 2.0 ** 54
        alpha = np.sqrt(-alpha)
        t = 0.0
        for i in range(nvar):
            if i:
                J = np.arange(i)
                s = np.dot(l[i + nvar * J], w[J])
            else:
                s = 0.0
            w[i] = alpha * w[i] - s
            t += (w[i] / l[i + nvar * i]) * w[i]
        t = 1.0 - t
        if t < dmin:
            t = dmin
        for i in range(nvar - 1, -1, -1):
            p = w[i]
            d = l[i + nvar * i]
            b = d * t
            t += (p / d) * p
            l[i + nvar * i] = b / t
            b = -p / b
            if i + 1 < nvar:
                J = np.arange(i + 1, nvar)
                lji = l[J + nvar * i].copy()
                l[J + nvar * i] = lji + b * w[J]
                w[J] = w[J] + p * lji

    # ---- criteria (translated) ---------------------------------------------
    @staticmethod
    def residue(g, p, x, nvar):
        return float(np.sqrt(SFNewton.norm2(p) * SFNewton.norm2(g)
                             / (1.0 + SFNewton.norm2(x))))

    @staticmethod
    def newfunction(g, nvar):
        return SFNewton.norm2(g) ** 2

    @staticmethod
    def linecriterion(g, g0, p, p0, nvar):
        normg = SFNewton.norm2(g0)
        gg0 = float(np.dot(g, g0))
        gg0 = gg0 / normg / normg
        normg = (SFNewton.norm2(g) / normg) ** 2
        if (gg0 > 1 or normg > 1) and normg - gg0 * abs(gg0) < 0.2:
            normg = 1.5 * normg
        if gg0 < 0 and normg < 2:
            return 1.0
        elif normg > 10:
            return 0.01
        else:
            return 0.4 * (1 + 0.75 * normg) / (normg - gg0 * abs(gg0) + 0.1)

    # ---- full numerical Hessian (translated numhessian) ---------------------
    def numhessian(self, h, g, x, nvar):
        """Full Jacobian of the residual, stored as h[j+nvar*i] = dg_j/dx_i
        (translated Namics numhessian; the adaptive step `di` is
        Scheutjens'). Costs nvar+1 residual evaluations -- unless an exact
        Jacobian provider was injected (accel["hessian"]), in which case it
        costs one evaluation and is exact."""
        if self._hessian_fn is not None:
            J = np.asarray(self._hessian_fn(x))
            h[:] = J.T.ravel()                # h layout: H[i, j] = dg_j/dx_i
            return
        dmax2 = 2.0 ** (self.nbits / 2)
        dmax3 = 2.0 ** (self.nbits / 3)
        for i in range(nvar):
            xt = x[i]
            di = (1.0 / (dmax3 * dmax3 * abs(h[i + nvar * i]) + dmax3 + abs(g[i]))
                  + 1.0 / dmax2) * (1.0 + abs(x[i]))
            x[i] = xt + di
            g1 = self.residuals(x)
            x[i] = xt
            h[nvar * i: nvar * i + nvar] = (g1 - g) / di
        g[:] = self.residuals(x)

    def decompos(self, h, nvar):
        """LU-style factorization of h in place (translated Namics decompos):
        turns the full numerical Hessian into the factored form that gausa/gausb
        and the secant updates operate on. Returns ntr = number of negative
        pivots. O(nvar^3) flops but computed only once (the anchor); both inner
        loops vectorised per pivot: the J<I scaling acts on column/row slices
        (each J only reads its own, already-final pivot diag[J]), and the J>I
        reduction is two matrix-vector products."""
        H = h.reshape(nvar, nvar)
        diag = h[::nvar + 1]                     # pivots (view)
        ntr = 0
        for I in range(nvar):
            if I:
                dj = diag[:I]                    # finalized pivots J < I
                l_vec = H[:I, I] / dj            # C: l = h[J,I]/h[J,J]
                s = float(np.dot(l_vec, H[I, :I]))   # C: s += l*c2 (old row)
                H[:I, I] = l_vec
                H[I, :I] /= dj                   # C: h[I,J] = c2/h[J,J]
            else:
                s = 0.0
            phi = H[I, I] - s
            H[I, I] = phi
            if phi < 0:
                ntr += 1
            if I and I + 1 < nvar:
                # C inner k-sums for all J > I at once:
                #   h[I,J] -= sum_k h[I,k]*h[k,J];  h[J,I] -= sum_k h[J,k]*h[k,I]
                H[I, I + 1:] -= H[I, :I] @ H[:I, I + 1:]
                H[I + 1:, I] -= H[I + 1:, :I] @ H[:I, I]
        return ntr

    # ---- hessian update (translated newhessian, pseudohessian branch) -------
    def newhessian(self, h, g, g0, x, p, nvar, accuracy, ALPHA):
        if self.full_hessian:
            # Namics method:hessian (solve_scf.cpp:195, sfnewton.cpp:487):
            # the full Hessian is recomputed EVERY iteration -- no secant
            # updates in between. samehessian skips the recompute right
            # after an anchor computation.
            if not self.samehessian:
                self.numhessian(h, g, x, nvar)
                self.decompos(h, nvar)
            self.samehessian = False
            return
        dmin = 1.0 / 2.0 ** self.nbits
        if not (not self.samehessian and ALPHA != 0 and self.iterations != 0):
            if not self.samehessian:
                self.resethessian(h, g, nvar)
            return
        y = g - g0
        py = float(np.dot(p, y))
        # ignore_newton_direction keeps newtondirection true, so hp = -g0
        hp = -g0.copy()
        php = float(np.dot(p, hp))
        theta = py / (10 * dmin + ALPHA * php)
        if (theta > 0 and self.iterations == self.resetiteration + 1
                and accuracy > self.max_accuracy_for_hessian_scaling):
            ALPHA *= theta
            py /= theta
            php /= theta
            p /= theta
            h[::nvar + 1] *= theta               # diagonal h[i+nvar*i] *= theta
        self.trustfactor *= (4.0 / ((theta - 1) ** 2 + 1) + 0.5)
        if nvar > 1:
            ssum = ALPHA * self.norm2(p) ** 2
            theta = abs(py / (ALPHA * php))
            if theta < 0.01:
                ssum /= 0.8
            elif theta > 100:
                ssum *= theta / 50
            y = y - ALPHA * hp
            self.updatpos(h, y, p.copy(), nvar, 1.0 / ssum)
            self.trouble -= self.signdeterminant(h, nvar)
            if self.trouble < 0:
                self.trouble = 0
            elif self.trouble >= 3:
                self.resethessian(h, g, nvar)
        elif py > 0:                                    # nvar == 1
            self.trouble = 0
            theta = (1.0 if py > 0.2 * ALPHA * php
                     else 0.8 * ALPHA * php / (ALPHA * php - py))
            if theta < 1:
                y = theta * y + (1 - theta) * ALPHA * hp
                py = float(np.dot(p, y))
            self.updatpos(h, y.copy(), y.copy(), nvar, 1.0 / (ALPHA * py))
            self.updateneg(h, hp, nvar, -1.0 / php)

    def direction(self, h, p, g, g0, x, nvar, alpha, accuracy):
        self.newtondirection = True
        self.newhessian(h, g, g0, x, p, nvar, accuracy, alpha)
        self.gausa(h, p, g, nvar)
        self.gausb(h, p, nvar)
        if self.ignore_newton_direction:
            self.newtondirection = True
        else:
            self.newtondirection = self.signdeterminant(h, nvar) > 0
        if not self.newtondirection:
            p *= -1

    def newdirection(self, h, p, p0, g, g0, x, nvar, ALPHA):
        p0[:] = p
        accuracy = self.residue(g, p, x, nvar)
        self.direction(h, p, g, g0, x, nvar, ALPHA, accuracy)
        return accuracy

    def newtrustregion(self, p0, ALPHA, delta_max, delta_min, nvar):
        normp0 = self.norm2(p0)
        if normp0 > 0 and self.trustregion > 2 * ALPHA * normp0:
            self.trustregion = 2 * ALPHA * normp0
        self.trustregion *= self.trustfactor
        self.trustfactor = 1.0
        if self.trustregion > delta_max:
            self.trustregion = delta_max
        if self.trustregion < delta_min:
            self.trustregion = delta_min

    # ---- line search (translated zero/linesearch/stepchange) ----------------
    def zero(self, g, g0, p, x, x0, nvar, newalpha):
        alpha = newalpha
        self.lineiterations += 1
        if self.lineiterations == 5:
            x[:] = x0
            g[:] = self.residuals(x)
            for i in range(nvar):
                if not np.isfinite(g[i]):
                    alpha *= -1
                    break
        x[:] = x0 + alpha * p
        g[:] = self.residuals(x)
        if not np.all(np.isfinite(g)):
            g[~np.isfinite(g)] = 1.0
        self.minimum = self.newfunction(g, nvar)
        return alpha

    def linesearch(self, g, g0, p, x, x0, nvar, alphabound):
        newalpha = alphabound if alphabound < 1 else 1.0
        return self.zero(g, g0, p, x, x0, nvar, newalpha)

    def stepchange(self, g, g0, p, p0, x, x0, nvar, alpha):
        change = crit = self.linecriterion(g, g0, p, p0, nvar)
        while crit < 0.35 and self.lineiterations < self.linesearchlimit:
            alpha /= 4.0
            self.zero(g, g0, p, x, x0, nvar, alpha)
            crit = self.linecriterion(g, g0, p, p0, nvar)
            change = 1.0
        return change, alpha

    # ---- reset heuristics (translated Solve_scf::inneriteration, default) ---
    def inneriteration(self, h, g, x, accuracy, nvar, deltamax, ALPHA):
        if self.iterations > 0:
            self.samehessian = False
        if (accuracy < self.minAccuracySoFar and self.iterations > 0
                and accuracy == abs(accuracy)):
            self.minAccuracySoFar = accuracy
        if (accuracy > self.minAccuracySoFar * self.resetHessianCriterion
                and accuracy == abs(accuracy)):                 # walking backwards
            self.resethessian(h, g, nvar)
            self.minAccuracySoFar *= 1.5
            if deltamax > 0.005:
                deltamax *= 0.9
        if ALPHA < self.smallAlpha:
            self.smallAlphaCount += 1
        else:
            self.smallAlphaCount = 0
        if self.smallAlphaCount == self.maxNumSmallAlpha:         # too many small steps
            self.smallAlphaCount = 0
            self.resethessian(h, g, nvar)
            if deltamax > 0.005:
                deltamax *= 0.9
        return deltamax

    # ---- main loop (translated iterate) ------------------------------------
    def iterate(self, x, tolerance, iterationlimit, delta_max, delta_min,
                anchor_full=False, full_hessian=False):
        """x: reduced (masked) starting vector, updated in place. With
        anchor_full=True the full numerical Hessian is computed ONCE at the
        start (numhessian + decompos) as the anchor, then cheap pseudohessian
        secant updates continue from it -- a fast full-Hessian variant that
        cracks the stiffest problems without paying for a full Hessian every
        step. With full_hessian=True the Hessian is recomputed EVERY
        iteration instead (Namics method:hessian semantics) -- expensive
        with finite differences, cheap and exact if an exact Jacobian is
        injected. Returns (converged, iterations, accuracy)."""
        self.full_hessian = bool(full_hessian)
        nvar = x.size
        if nvar < 1:
            return False, 0, np.inf
        g = self.residuals(x)
        x0 = x.copy()
        g0 = np.zeros(nvar)
        p = np.zeros(nvar)
        p0 = np.zeros(nvar)
        h = np.zeros(nvar * nvar)

        self.trouble = self.resetiteration = 0
        self.minAccuracySoFar = 1e30
        self.samehessian = False
        self.newtondirection = False
        self.iterations = 0
        self.lineiterations = 0
        self.smallAlphaCount = 0
        ALPHA = 1.0
        self.trustregion = delta_max
        self.trustfactor = 1.0
        deltamax = delta_max

        if anchor_full:
            # compute the full numerical Hessian once and factorise it; keep it
            # (samehessian) through iteration 0 so the first step is a true
            # Newton step, then secant updates take over.
            self.numhessian(h, g, x, nvar)
            self.decompos(h, nvar)
            self.samehessian = True
        self.newhessian(h, g, g0, x, p, nvar, 1e30, ALPHA)
        self.minimum = self.newfunction(g, nvar)
        deltamax = self.inneriteration(h, g, x, 1e30, nvar, deltamax, ALPHA)
        accuracy = self.newdirection(h, p, p0, g, g0, x, nvar, ALPHA)
        self.normg = np.sqrt(self.minimum)
        accuracy = self.residue(g, p, x, nvar)

        it = 0
        while ((tolerance < accuracy or tolerance * 10 < self.normg)
               and it < iterationlimit and accuracy == abs(accuracy)):
            it += 1
            self.iterations = it
            self.lineiterations = 0
            self.newtrustregion(p0, ALPHA, deltamax, delta_min, nvar)
            alphabound = self.trustregion / (self.norm2(p) + 1.0 / 2.0 ** self.nbits)
            x0[:] = x
            g0[:] = g
            ALPHA = self.linesearch(g, g0, p, x, x0, nvar, alphabound)
            change, ALPHA = self.stepchange(g, g0, p, p0, x, x0, nvar, ALPHA)
            self.trustfactor *= change
            self.trustfactor *= ALPHA / alphabound
            deltamax = self.inneriteration(h, g, x, accuracy, nvar, deltamax, ALPHA)
            accuracy = self.newdirection(h, p, p0, g, g0, x, nvar, ALPHA)
            self.normg = np.sqrt(self.minimum)

        converged = accuracy < tolerance and self.normg < 10 * tolerance
        return converged, it, accuracy
