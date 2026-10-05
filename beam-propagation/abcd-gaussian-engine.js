/**
 * abcd-gaussian-engine.js
 *
 * Paraxial Gaussian-beam propagation via the ABCD-matrix complex
 * beam-parameter q. Used by the fys501-laser-physics cmpc-gaussian-beam
 * applet (in mikkojhuttunen/physics-applets) to propagate a beam through a
 * periodic sequence of free-space legs and oblique-incidence spherical-mirror
 * reflections.
 *
 * Conventions (pick consistent units once and stick to them — the applets
 * use millimeters throughout, i.e. lambdaMm, waistMm, LMm, RmMm):
 *   - q is a plain {re, im} object representing the complex beam parameter
 *     1/q = 1/R - i*lambda/(pi*w^2), equivalently q = z + i*zR where z is
 *     distance from the waist and zR = pi*w0^2/lambda is the Rayleigh range.
 *     im(q) is always the local Rayleigh range and must stay > 0.
 *   - Free-space propagation by distance x: q' = q + x  (cAdd).
 *   - A thin-lens-equivalent element of focal length f (converging f>0):
 *     q' = q / (1 - q/f)  (cMirror — name reflects the main use case, but
 *     it's the general thin-lens ABCD transform [[1,0],[-1/f,1]]).
 *   - Oblique incidence on a spherical mirror of radius of curvature R_m at
 *     angle-from-normal theta splits one radius of curvature into two focal
 *     lengths (standard astigmatic-mirror result):
 *       f_tangential = R_m*cos(theta)/2   (in the plane of incidence)
 *       f_sagittal   = R_m/(2*cos(theta)) (perpendicular to it)
 *     At theta=0 these coincide (f = R_m/2, the normal-incidence case);
 *     astigmatism (f_t != f_s) grows with obliqueness.
 *   - A "unit cell" = propagate distance L, then reflect off one mirror:
 *     matrix [[1,L],[0,1]] then [[1,0],[-1/f,1]], composed as
 *     M = [[1,0],[-1/f,1]] * [[1,L],[0,1]] = [[1,L],[-1/f, 1-L/f]]
 *     (unitCell). Its stability parameter g = (A+D)/2 = 1 - L/(2f)
 *     (stabilityG) must satisfy |g| < 1 for a periodic repetition of this
 *     unit cell to support a bounded (self-reproducing) Gaussian mode.
 *
 * Verified (see test/abcd-gaussian-engine.test.js):
 *   - solveMatchedQ's output is an exact fixed point of its own unit-cell
 *     matrix (self-consistency to ~1e-6 or better).
 *   - An unstable unit cell (|g|>=1) returns null from solveMatchedQ and
 *     produces monotonically diverging width under repeated cMirror/cAdd,
 *     with no NaNs.
 *
 * No dependencies. Works as a CommonJS module (Node) or a plain browser
 * global (ABCDGaussianEngine) via a <script> tag.
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = mod;
  } else {
    root.ABCDGaussianEngine = mod;
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /**
   * Tangential/sagittal focal lengths of a spherical mirror (radius of
   * curvature RmMm) at oblique incidence angle thetaRad from the normal.
   * @returns {{fT:number, fS:number}}
   */
  function obliqueMirrorFocalLengths(RmMm, thetaRad) {
    const cosT = Math.cos(thetaRad);
    return { fT: (RmMm * cosT) / 2, fS: RmMm / (2 * cosT) };
  }

  /**
   * ABCD matrix of one "unit cell": propagate distance L, then reflect off a
   * thin-lens-equivalent element of focal length f.
   * @returns {{A:number,B:number,C:number,D:number}}
   */
  function unitCell(L, f) {
    return { A: 1, B: L, C: -1 / f, D: 1 - L / f };
  }

  /** Stability parameter g = (A+D)/2 for an ABCD matrix; periodic repetition is stable iff |g| < 1. */
  function stabilityG(A, D) {
    return (A + D) / 2;
  }

  /** Free-space propagation of q by a real distance x (same units as q.im). */
  function cAdd(q, x) {
    return { re: q.re + x, im: q.im };
  }

  /**
   * Apply a thin-lens/mirror ABCD transform of focal length f to q:
   * q' = q / (1 - q/f), i.e. matrix [[1,0],[-1/f,1]].
   */
  function cMirror(q, f) {
    const dre = 1 - q.re / f, dim = -q.im / f;
    const denom = dre * dre + dim * dim;
    return { re: (q.re * dre + q.im * dim) / denom, im: (q.im * dre - q.re * dim) / denom };
  }

  /**
   * Beam radius w implied by q at the given wavelength (same length unit as q and lambda, e.g. mm).
   * w^2 = lambda*(re^2+im^2)/(pi*im); requires q.im > 0.
   */
  function widthFromQ(q, lambdaMm) {
    return Math.sqrt((lambdaMm * (q.re * q.re + q.im * q.im)) / (Math.PI * q.im));
  }

  /** Beam waist radius w0 corresponding to Rayleigh range zR at wavelength lambdaMm. w0 = sqrt(lambda*zR/pi). */
  function waistFromZr(zRMm, lambdaMm) {
    return Math.sqrt((lambdaMm * zRMm) / Math.PI);
  }

  /** Rayleigh range zR for a waist w0Mm at wavelength lambdaMm. zR = pi*w0^2/lambda. */
  function zRFromWaist(w0Mm, lambdaMm) {
    return (Math.PI * w0Mm * w0Mm) / lambdaMm;
  }

  /**
   * Self-consistent ("matched"/eigenmode) q for a periodic ABCD unit cell:
   * the fixed point of q = (A*q+B)/(C*q+D), i.e. the one complex beam
   * parameter that reproduces itself every time the unit cell is applied —
   * the periodic-system analogue of a laser-resonator eigenmode.
   *
   * Derivation: q = (Aq+B)/(Cq+D) => Cq^2 + (D-A)q - B = 0. For a unimodular
   * matrix (AD-BC=1) the discriminant reduces to (A+D)^2-4, so a stable cell
   * (|g|<1, g=(A+D)/2) gives a complex-conjugate pair of roots; the physical
   * one has Im(q) > 0 (positive Rayleigh range).
   *
   * @returns {{re:number, im:number}|null} null if the unit cell is unstable
   *   (|g| >= 1) — there is no bounded self-reproducing mode in that case.
   */
  function solveMatchedQ(A, B, C, D) {
    const g = (A + D) / 2;
    if (Math.abs(g) >= 1) return null;
    const s = Math.sqrt(1 - g * g);
    const re = (A - D) / (2 * C);
    const q1 = { re, im: s / C }, q2 = { re, im: -s / C };
    return q1.im > 0 ? q1 : (q2.im > 0 ? q2 : null);
  }

  /**
   * Propagate a beam through a sequence of identical unit cells (same L, f),
   * starting from an injected waist w0Mm at the start of the first leg
   * (q0 = {re:0, im: zRFromWaist(w0Mm, lambdaMm)}). Returns, for each leg,
   * the q-state at the *start* of that leg plus the accumulated plane-wave
   * (k0*s) and Gouy phase up to that point — everything needed to evaluate
   * the field anywhere along the path later (e.g. for a coherent-sum field
   * map), or to detect clipping against a finite aperture.
   *
   * Stops early if the beam exceeds `apertureMm` (full width, 2*w) at any
   * mirror plane, if apertureMm is given.
   *
   * @param {number} nLegs number of unit cells (bounces) to attempt
   * @param {number} L leg length (mm)
   * @param {number} f focal length of the repeated mirror element (mm)
   * @param {number} w0Mm injected waist radius (mm), assumed at the very start
   * @param {number} lambdaMm wavelength (mm)
   * @param {number} [apertureMm] optional full-width clipping aperture (mm)
   * @returns {{
   *   legs: Array<{qStart:{re:number,im:number}, sStart:number, gouyStart:number}>,
   *   clipIndex: number  // 1-based leg index where clipping first occurs, or -1
   * }}
   */
  function propagateUnitCellSequence(nLegs, L, f, w0Mm, lambdaMm, apertureMm) {
    let q = { re: 0, im: zRFromWaist(w0Mm, lambdaMm) };
    const legs = [];
    let sTotal = 0, gouyTotal = 0, clipIndex = -1;
    for (let k = 1; k <= nLegs; k++) {
      legs.push({ qStart: { re: q.re, im: q.im }, sStart: sTotal, gouyStart: gouyTotal });
      const qEnd = cAdd(q, L);
      if (apertureMm !== undefined) {
        const wAtMirror = widthFromQ(qEnd, lambdaMm);
        if (clipIndex < 0 && 2 * wAtMirror > apertureMm) clipIndex = k;
      }
      sTotal += L;
      gouyTotal += Math.atan2(qEnd.re, qEnd.im) - Math.atan2(q.re, q.im);
      q = cMirror(qEnd, f);
      if (clipIndex > 0) break;
    }
    return { legs, clipIndex };
  }

  return {
    obliqueMirrorFocalLengths,
    unitCell,
    stabilityG,
    cAdd,
    cMirror,
    widthFromQ,
    waistFromZr,
    zRFromWaist,
    solveMatchedQ,
    propagateUnitCellSequence
  };
});
