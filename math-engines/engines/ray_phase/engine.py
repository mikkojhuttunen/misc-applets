"""Optical phase of rays launched at slightly different angles into a multipass cell, and the fringe visibility
left when the launch angle is dithered.

A ray leaves the input point at angle θ from the inward normal and is traced through the cell (billiard_cell
SegmentedCell, any planar_cell wall: circle, stadium, faceted stadium, integrated Herriott; or a 3D Herriott cell
through herriott_cell.HerriottLaunch, where θ tilts the injected beam). After p mirror hits its geometric path is L_p(θ); its optical phase relative to the centre ray θc is

    Δφ_p(θ) = k0 n (L_p(θ) - L_p(θc)),      k0 = 2π / λ,  n = index of the medium (slab n_eff for a membrane)

Reflection phases are common to all rays with the same facet sequence and are ignored (rays that change facet
sequence pick up the same number of reflections per pass, so only the path term differs).

Angle dithering θ(t) = θc + δ(t) makes the phase of pass-p light wander by Δφ_p(θc + δ). Any interference term
between that light and light whose phase does not follow the dither (a reference, or a pass with a very
different dφ/dθ) is averaged over one dither period to

    V_p = | ⟨ exp(i Δφ_p(θc + δ)) ⟩_t |           (fringe visibility left; 1 = untouched, 0 = washed out)

For a linear phase Δφ = a δ: sine dither δ = A sin ωt gives V = |J0(a A)|, triangle dither (δ uniform in ±A)
gives V = |sin(a A)/(a A)|.

WHAT SETS V (dither_analysis). Around the centre ray, per pass p,

    Δφ_p(θc + δ) = a_p δ + ½ b_p δ² + ...   (+ jumps where dithered rays take a different facet sequence)
    a_p = k0 n dL_p/dθ,   b_p = k0 n d²L_p/dθ²

  1. dither depth   x_p = a_p A        -> V ≈ |J0(x_p)| (sine) or |sinc(x_p)| (triangle) while 2 and 3 are small.
                    Every parameter enters through x_p: A, n/λ, the cell (dL_p/dθ ∝ cell size, depends on θc, N,
                    curvature and tilts) and the pass number (dL_p/dθ grows ∝ p in a regular cell, ∝ e^{λp} in a
                    chaotic one).
  2. chirp          q_p = ½ |b_p| A²   (rad): the phase is no longer linear across the dither when q_p ≳ 1.
  3. path switching the share of dithered rays that keep the centre ray's facet sequence up to pass p; rays that
                    switch carry a path difference of order a chord (thousands of waves), so their contribution
                    averages out; the rays that keep the path are those near δ = 0 (a smaller effective dither), so
                    V is no longer set by x alone and can lie above or below the linear prediction.
Sampling is refined until the phase step between neighbouring dither samples on the same facet sequence is below
π/2 at every pass (or n_max is reached); steps across a facet switch are real discontinuities and do not count.
Unresolved values sit near the random-phase floor sqrt(π)/(2 sqrt(N)) (mean |Σ e^{iφ}|/N of N random phases). V² is the coherence (Strehl-like) factor of an angular fan of rays with those
weights. Ray picture only: rays at different angles end at different points after p passes, so V_p describes
the phase stability of each ray's own path, not the overlap of neighbouring rays. SI units.
"""
from __future__ import annotations

import numpy as np

from ..billiard_cell.engine import SegmentedCell, _reflect
from ..common import Result, require_choice, require_positive, require_range

_WAVEFORMS = ("sine", "triangle")
X_V09 = {"sine": 0.6406308771588489, "triangle": 0.786683072049212}     # x with model V(x) = 0.9
X_V05 = {"sine": 1.521144057668765, "triangle": 1.8954942670340231}     # V = 0.5
X_V0 = {"sine": 2.404825557695807, "triangle": np.pi}                   # first zero
_HASH_M = 2147483647                                                    # path ids stay exact in float64 (and in JS)


def launch(cell, theta, s_in=None):
    """Start points and unit directions for rays leaving boundary coordinate s_in at angles theta (array)."""
    s_in = cell.s_in_default if s_in is None else s_in
    th = np.atleast_1d(np.asarray(theta, float))
    x, y, nx, ny = cell.point_at(np.full(th.shape, s_in))
    ux, uy = -nx, -ny
    return x, y, ux * np.cos(th) - uy * np.sin(th), ux * np.sin(th) + uy * np.cos(th)


def _hit_segment(cell, x, y, dx, dy):
    """t, outward normal, s and segment (facet or element) index of the next hit. Cells with hit_k (planar_cell.Cell2D)
    report the element; billiard_cell.SegmentedCell's facet follows from s."""
    if hasattr(cell, "hit_k"):
        t, nx, ny, hs, k = cell.hit_k(x, y, dx, dy)
        return t, nx, ny, hs, np.where(np.isfinite(t), k, 0).astype(float)
    t, nx, ny, hs = cell.hit(x, y, dx, dy)
    return t, nx, ny, hs, np.clip(np.floor(np.nan_to_num(hs) / (2 * cell.h)), 0, cell.n_facets - 1)


def path_lengths(cell, theta, n_pass, s_in=None, return_ids=False):
    """Cumulative geometric path [m] to each of the first n_pass mirror hits: array (len(theta), n_pass).
    Rays that leak through a corner gap (or leave an open cell) get NaN from that hit on. return_ids=True also returns
    a path id per hit (hash of the facet / element sequence so far; equal ids = same sequence).
    Works for billiard_cell.SegmentedCell and planar_cell.Cell2D; any other cell object can supply its own
    path_lengths(theta, n_pass, s_in, return_ids) method (herriott_cell.HerriottLaunch does)."""
    if hasattr(cell, "path_lengths"):
        return cell.path_lengths(theta, n_pass, s_in, return_ids)
    x, y, dx, dy = launch(cell, theta, s_in)
    L = np.full((x.size, int(n_pass)), np.nan)
    ids = np.full((x.size, int(n_pass)), -1.0)
    pid = np.zeros(x.size)
    acc = np.zeros(x.size)
    for j in range(int(n_pass)):
        t, nx, ny, hs, fac = _hit_segment(cell, x, y, dx, dy)
        pid = np.where(np.isfinite(t) & (pid >= 0), np.mod(pid * 1000003 + fac + 1, _HASH_M), -1.0)   # -1 once leaked
        ids[:, j] = pid
        ok = np.isfinite(t)
        acc = np.where(ok, acc + np.where(ok, t, 0.0), np.nan)
        L[:, j] = acc
        t = np.where(ok, t, 0.0)
        x, y = x + t * dx, y + t * dy
        dx, dy = _reflect(dx, dy, np.where(ok, nx, 1.0), np.where(ok, ny, 0.0))
    return (L, ids) if return_ids else L


def relative_phase(L, L_ref, wavelength, n_index=1.0):
    """Δφ = 2π n (L - L_ref) / λ [rad]."""
    return 2 * np.pi * n_index * (np.asarray(L, float) - L_ref) / wavelength


def dither_offsets(amplitude, n=401, waveform="sine"):
    """Angle offsets δ sampled uniformly in time over one dither period: A sin(2π t) or a triangle wave (δ
    uniform in [-A, A]). Equal weights; the midpoint rule makes the sine case exact for band-limited phases."""
    require_choice("waveform", waveform, _WAVEFORMS)
    t = (np.arange(n) + 0.5) / n
    return amplitude * np.sin(2 * np.pi * t) if waveform == "sine" else amplitude * (2 * t - 1)


def visibility(dphi, weights=None):
    """|Σ w exp(iΔφ)| / Σ w over the samples (last axis), ignoring NaN samples."""
    dphi = np.asarray(dphi, float)
    w = np.ones(dphi.shape[-1]) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(dphi)
    W = np.where(ok, w, 0.0)
    z = (W * np.exp(1j * np.where(ok, dphi, 0.0))).sum(-1)
    return np.abs(z) / W.sum(-1)


def dither_scan(cell, theta_c, amplitude, n_pass, wavelength, n_index=1.0, waveform="sine", n_samples=401, s_in=None):
    """Phases and visibilities of passes 1..n_pass under angle dither. Returns dict with
    offsets δ, L (samples × passes), dphi (samples × passes, relative to the undithered ray),
    V (per pass), max_step (largest phase jump between neighbouring samples per pass; > π means the sampling
    no longer resolves the phase and V is only an upper-bound-like estimate)."""
    d = dither_offsets(amplitude, n_samples, waveform)
    L0 = path_lengths(cell, [theta_c], n_pass, s_in)[0]
    L = path_lengths(cell, theta_c + d, n_pass, s_in)
    dphi = relative_phase(L, L0[None, :], wavelength, n_index)
    order = np.argsort(d, kind="stable")
    steps = np.abs(np.diff(dphi[order], axis=0))
    max_step = np.nanmax(steps, axis=0) if len(d) > 1 else np.zeros(int(n_pass))
    return dict(offsets=d, L=L, L0=L0, dphi=dphi, V=visibility(dphi.T), max_step=max_step)


def _j0_large(x):
    """J0 for x > 1e4: two-term Hankel asymptote (error < 1e-12 there)."""
    return np.sqrt(2 / (np.pi * x)) * ((1 - 9 / (128 * x * x)) * np.cos(x - np.pi / 4) + 1 / (8 * x) * np.sin(x - np.pi / 4))


def model_visibility(x, waveform="sine", n_min=64):
    """|J0(x)| (sine) or |sinc(x)| (triangle): V of a linear phase a δ with x = a A. J0 by the same midpoint-in-time
    average the dither uses (exact to rounding for n > |x| + 30), so Python and the JS port agree bit for bit; above
    |x| = 1e4 (a chaotic cell after many passes) the Hankel asymptote replaces the |x|-sample average."""
    x = np.atleast_1d(np.asarray(x, float))
    out = np.empty_like(x)
    for i, v in enumerate(x):
        if not np.isfinite(v):
            out[i] = np.nan
        elif waveform == "sine" and abs(v) > 1e4:
            out[i] = abs(_j0_large(abs(v)))
        elif waveform == "sine":
            n = max(n_min, int(np.ceil(abs(v))) + 64)
            t = (np.arange(n) + 0.5) / n
            out[i] = abs(np.cos(v * np.sin(2 * np.pi * t)).mean())
        else:
            out[i] = 1.0 if v == 0 else abs(np.sin(v) / v)
    return out


def phase_derivatives(cell, theta_c, n_pass, wavelength, n_index=1.0, s_in=None, h1=1e-9, h2=1e-6):
    """a_p = dφ_p/dθ [rad/rad] (central difference, step h1) and b_p = d²φ_p/dθ² [rad/rad²] (step h2) at θc.
    NaN where the ±h rays do not share the centre ray's facet sequence (θc sits on a switching edge)."""
    k = 2 * np.pi * n_index / wavelength
    L, ids = path_lengths(cell, [theta_c - h2, theta_c - h1, theta_c, theta_c + h1, theta_c + h2], n_pass, s_in, True)
    same1 = (ids[1] == ids[2]) & (ids[3] == ids[2])
    same2 = (ids[0] == ids[2]) & (ids[4] == ids[2])
    a = np.where(same1, k * (L[3] - L[1]) / (2 * h1), np.nan)
    b = np.where(same2, k * (L[4] - 2 * L[2] + L[0]) / h2**2, np.nan)
    return a, b


def dither_analysis(cell, theta_c, amplitude, n_pass, wavelength, n_index=1.0, waveform="sine",
                    n_min=401, n_max=12801, step_tol=np.pi / 2, s_in=None):
    """Visibility per pass under angle dither, with the quantities that explain it (module docstring).

    Returns a dict of per-pass arrays (index p - 1):
      V           computed visibility                      V_model   |J0(x)| or |sinc(x)| of the linear model
      a, b        dφ/dθ, d²φ/dθ² at θc                     x, q      dither depth a A and chirp ½ |b| A²
      same_path   share of dither samples on the centre ray's facet sequence
      max_step    largest phase step between neighbouring same-path samples (rad)
      resolved    max_step <= step_tol                     A_09, A_05, A_0   amplitudes for model V = 0.9, 0.5, 0
    and scalars n_used, noise_floor = sqrt(π)/(2 sqrt(n_used)), L0 (path of the centre ray per pass)."""
    require_choice("waveform", waveform, _WAVEFORMS)
    n_pass = int(n_pass)
    k = 2 * np.pi * n_index / wavelength
    L0, id0 = path_lengths(cell, [theta_c], n_pass, s_in, True)
    L0, id0 = L0[0], id0[0]
    n = int(n_min)
    while True:
        d = dither_offsets(amplitude, n, waveform)
        L, ids = path_lengths(cell, theta_c + d, n_pass, s_in, True)
        dphi = k * (L - L0[None, :])
        order = np.argsort(d, kind="stable")
        ph, pi_ = dphi[order], ids[order]
        cont = (pi_[1:] == pi_[:-1]) & (pi_[1:] >= 0)
        steps = np.where(cont, np.abs(np.diff(ph, axis=0)), 0.0)
        max_step = np.nanmax(steps, axis=0) if n > 1 else np.zeros(n_pass)
        worst = float(np.nanmax(max_step)) if n_pass else 0.0
        if worst <= step_tol or n >= n_max:
            break
        n = int(min(n_max, max(2 * n - 1, np.ceil(n * worst / step_tol * 1.25))))
        n += 1 - n % 2                                   # odd n puts one sample at δ = 0
    a, b = phase_derivatives(cell, theta_c, n_pass, wavelength, n_index, s_in)
    x = np.abs(a) * amplitude
    with np.errstate(divide="ignore"):
        inv = 1 / np.abs(a)
    return dict(V=visibility(dphi.T), V_model=model_visibility(x, waveform), a=a, b=b, x=x,
                q=0.5 * np.abs(b) * amplitude**2, same_path=(ids == id0[None, :]).mean(0), max_step=max_step,
                resolved=max_step <= step_tol, A_09=X_V09[waveform] * inv, A_05=X_V05[waveform] * inv,
                A_0=X_V0[waveform] * inv, n_used=n, noise_floor=float(np.sqrt(np.pi) / (2 * np.sqrt(n))), L0=L0)


def _filon(h, a, b):
    """∫ over an interval of length h of exp(iφ) with φ linear from a to b: h e^{i(a+b)/2} sinc((b−a)/2), exact for any slope."""
    x = 0.5 * (b - a)
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = np.where(np.abs(x) < 1e-8, 1.0 - x * x / 6, np.sin(x) / np.where(x == 0, 1.0, x))
    return h * np.exp(0.5j * (a + b)) * sc


def dither_filon(cell, theta_c, amplitude, n_pass, wavelength, n_index=1.0, waveform="sine", tol=1e-7, n_min=256,
                 n_max=24000, s_in=None, passes=None):
    """Dither visibility as an adaptive oscillatory (Filon) integral: exact where the phase can be resolved, with an
    error estimate where it cannot. Over one period V_p = |Z_p|, Z_p = ∫₀¹ exp(iΔφ_p(θc + δ(s))) ds with
    δ(s) = A sin(π(s − ½)) (sine: s is time over the half period, so equal measure) or A(2s − 1) (triangle).

    The s range is cut into intervals; on each the phase is taken as linear between its ends and integrated exactly
    (any phase slope, however many waves), checked against the two halves through the midpoint. An interval is kept
    at pass p when the phase is finite at its ends and midpoint, its second difference is ≤ 1 rad and the halves
    change the integral by ≤ tol × its length (its error: that change). Path ids are not used: many path switches
    (a ray grazing an inclusion or a wall corner) keep the phase continuous, and the phase test catches those that do
    not. Open intervals are halved, widest first, while n_max ray evaluations last; passes (1-based, default all)
    are the passes whose test drives the refinement (e.g. only the exit pass).
    Intervals left open count with each half's end phases, an error of up to the half's width whose sign varies
    from interval to interval, so these add in quadrature:  V_err = Σ kept errors + sqrt(Σ open half-widths²).
    A regular cell (Herriott, mild deformation) is resolved completely and V is exact far below the random-phase
    floor √π/(2√n) of dither_analysis (to ~tol). After many passes of a chaotic cell the phase breaks into path
    segments down to 1e-11 rad of dither with long-range structure between them; there V_err falls only as ~1/√n_max,
    as in sampling, and V below V_err is not resolved (an upper bound V + V_err).
    Returns V, V_err (absolute; noise_floor: its largest value), frag (share of s left open), same_path (share kept
    on the centre ray's path ids), max_step (largest |Δφ| across a kept interval), resolved (V_err ≤ max(10 tol,
    V/5)), n_used, L0 and the linear-model keys of dither_analysis."""
    require_choice("waveform", waveform, _WAVEFORMS)
    P = int(n_pass)
    k = 2 * np.pi * n_index / wavelength
    L0, id0 = path_lengths(cell, [theta_c], P, s_in, True)
    L0, id0 = L0[0], id0[0]
    A = float(amplitude)
    delta = (lambda s: A * np.sin(np.pi * (s - 0.5))) if waveform == "sine" else (lambda s: A * (2 * s - 1))

    def ev(s):
        L, ids = path_lengths(cell, theta_c + delta(np.asarray(s, float)), P, s_in, True)
        return k * (L - L0[None, :]), ids

    tgt = np.zeros(P, bool)
    if passes is None:
        tgt[:] = True
    else:
        tgt[np.clip(np.atleast_1d(np.asarray(passes, int)) - 1, 0, P - 1)] = True
    n0 = int(n_min)
    s_nodes = np.arange(n0 + 1) / n0
    ph, ids = ev(s_nodes)
    used = n0 + 1
    lo, hi = s_nodes[:-1], s_nodes[1:]
    pl, pr, il = ph[:-1], ph[1:], ids[:-1]
    Isum, meas, err_s, err_q = np.zeros(P, complex), np.zeros(P), np.zeros(P), np.zeros(P)
    open_meas, same_meas, max_step = np.zeros(P), np.zeros(P), np.zeros(P)
    while lo.size:
        mph, mid = ev(0.5 * (lo + hi))
        used += lo.size
        H = (hi - lo)[:, None]
        fl, fm, fr = np.isfinite(pl), np.isfinite(mph), np.isfinite(pr)
        a_, m_, b_ = (np.where(np.isfinite(v), v, 0.0) for v in (pl, mph, pr))
        If = _filon(H / 2, a_, m_) + _filon(H / 2, m_, b_)
        e = np.abs(_filon(H, a_, b_) - If)
        ok = fl & fm & fr & (e <= tol * H) & (np.abs(a_ - 2 * m_ + b_) <= 1.0)
        dead = ~(fl | fm | fr)                                         # leaked throughout: no light, no measure
        done = np.all((ok | dead)[:, tgt], axis=1)
        refine = ~done
        budget = n_max - used
        n_open = int(refine.sum())
        if n_open > max(budget, 0):                                    # refine the widest open intervals that fit
            keep = np.zeros(n_open, bool)
            if budget > 0:
                keep[np.argsort(-(hi - lo)[refine], kind="stable")[:budget]] = True
            ri = np.nonzero(refine)[0]
            refine = refine.copy()
            refine[ri[~keep]] = False
        t = ~refine
        if t.any():
            okt, Ht, half = ok[t], H[t], H[t] / 2
            flt, fmt, frt = fl[t], fm[t], fr[t]
            jl, jr = flt & fmt, fmt & frt
            # open intervals: each half with its end phases (Filon when both ends are lit)
            left = np.where(jl, _filon(half, a_[t], m_[t]), np.where(flt | fmt, half * np.exp(1j * np.where(flt, a_[t], m_[t])), 0.0))
            right = np.where(jr, _filon(half, m_[t], b_[t]), np.where(fmt | frt, half * np.exp(1j * np.where(frt, b_[t], m_[t])), 0.0))
            Isum += np.where(okt, If[t], left + right).sum(0)
            lit_l, lit_r = (flt | fmt).astype(float), (fmt | frt).astype(float)
            meas += np.where(okt, Ht, half * (lit_l + lit_r)).sum(0)
            err_s += np.where(okt, e[t], 0.0).sum(0)
            op = ~okt & ~dead[t]
            err_q += np.where(op, half**2 * (lit_l + lit_r), 0.0).sum(0)
            open_meas += np.where(op, half * (lit_l + lit_r), 0.0).sum(0)
            same_meas += np.where(okt & (il[t] == id0[None, :]), Ht, 0.0).sum(0)
            max_step = np.maximum(max_step, np.where(okt, np.abs(b_[t] - a_[t]), 0.0).max(0, initial=0.0))
        if not refine.any():
            break
        r = refine
        m = 0.5 * (lo[r] + hi[r])
        lo, hi = np.r_[lo[r], m], np.r_[m, hi[r]]
        pl, pr = np.r_[pl[r], mph[r]], np.r_[mph[r], pr[r]]
        il = np.r_[il[r], mid[r]]
    good = meas > 0
    den = np.where(good, meas, 1.0)
    V = np.where(good, np.abs(Isum) / den, np.nan)
    V_err = np.where(good, (err_s + np.sqrt(err_q)) / den, np.nan)
    a, b = phase_derivatives(cell, theta_c, P, wavelength, n_index, s_in)
    x = np.abs(a) * A
    with np.errstate(divide="ignore"):
        inv = 1 / np.abs(a)
    return dict(V=V, V_err=V_err, frag=np.where(good, open_meas / den, 0.0), V_model=model_visibility(x, waveform), a=a,
                b=b, x=x, q=0.5 * np.abs(b) * A**2, same_path=np.where(good, same_meas / den, 0.0), max_step=max_step,
                resolved=V_err <= np.maximum(10 * tol, 0.2 * V), A_09=X_V09[waveform] * inv, A_05=X_V05[waveform] * inv,
                A_0=X_V0[waveform] * inv, n_used=used, noise_floor=float(np.nanmax(V_err)) if P else 0.0, L0=L0)


def _cell(radius, n_facets, curvature):
    require_positive(radius=radius)
    n = int(n_facets)
    if n < 3:
        raise ValueError("n_facets must be >= 3")
    return SegmentedCell(radius, n, curvature=curvature)


def first_mirror_phase(radius=5e-3, n_facets=24, theta_c=0.6545, wavelength=1.55e-6, n_index=1.0, curvature=0.0) -> Result:
    """Path from the input point to the first mirror and how fast its phase turns with launch angle."""
    require_positive(wavelength=wavelength, n_index=n_index)
    require_range("theta_c", theta_c, -1.55, 1.55)
    cell = _cell(radius, n_facets, curvature)
    h = 1e-6
    L = path_lengths(cell, [theta_c - h, theta_c, theta_c + h], 1)[:, 0]
    dL = (L[2] - L[0]) / (2 * h)
    dphi = 2 * np.pi * n_index * dL / wavelength
    return Result(
        values={"L1": L[1], "dL_dtheta": dL, "dphi_dtheta": dphi, "dtheta_2pi": abs(2 * np.pi / dphi) if dphi else np.inf},
        units={"L1": "m", "dL_dtheta": "m/rad", "dphi_dtheta": "rad/rad", "dtheta_2pi": "rad"},
        assumptions=["Ray optics in a regular segmented cell", "Phase = k0 n L, reflection phases ignored",
                     "Derivative by central difference (1 µrad)"],
    )


def dither_visibility(radius=5e-3, n_facets=24, theta_c=0.6545, wavelength=1.55e-6, n_index=1.0, amplitude=1e-4,
                      n_pass=24, waveform="sine", curvature=0.0) -> Result:
    """Fringe visibility at pass 1 and pass n_pass under launch-angle dither, with the dither depth x = a A,
    the linear-model prediction, chirp, path-switching share and the amplitudes that give V = 0.9, 0.5, 0."""
    require_positive(wavelength=wavelength, n_index=n_index, amplitude=amplitude)
    require_range("theta_c", theta_c, -1.55, 1.55)
    require_choice("waveform", waveform, _WAVEFORMS)
    n_pass = int(n_pass)
    if n_pass < 1:
        raise ValueError("n_pass must be >= 1")
    r = dither_analysis(_cell(radius, n_facets, curvature), theta_c, amplitude, n_pass, wavelength, n_index, waveform, n_max=6401)
    i = n_pass - 1
    return Result(
        values={"V_first": float(r["V"][0]), "V_last": float(r["V"][i]), "V_model_last": float(r["V_model"][i]),
                "x_last": float(r["x"][i]), "chirp_last": float(r["q"][i]), "same_path_last": float(r["same_path"][i]),
                "A_half_last": float(r["A_05"][i]), "A_zero_last": float(r["A_0"][i]), "L_last": float(r["L0"][i]),
                "noise_floor": r["noise_floor"], "n_samples": r["n_used"], "resolved": bool(r["resolved"][i])},
        units={"V_first": "", "V_last": "", "V_model_last": "", "x_last": "rad", "chirp_last": "rad", "same_path_last": "",
               "A_half_last": "rad", "A_zero_last": "rad", "L_last": "m", "noise_floor": "", "n_samples": "", "resolved": ""},
        assumptions=["Ray optics, reflection phases ignored", "V = |time average of exp(iΔφ)| over one dither period",
                     "Model V: linear phase, |J0(aA)| (sine) or |sinc(aA)| (triangle)",
                     "resolved = False: phase step between same-path samples > π/2 at 6401 samples; V is then near noise_floor"],
    )
