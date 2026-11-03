"""Power budget of one probe (or smooth reference) run, per monitor frequency.

All powers are normalised to the excited mode's forward power at the input monitor:
  same_fwd     forward power left in the excited mode after the section
  other_fwd    forward power converted into other guided modes (per mode and total)
  same_bwd     power reflected into the excited mode
  other_bwd    power reflected into other modes
  rad_free     outward flux through the top and bottom (z) planes: free-space radiation
  rad_side     outward flux through the side (y) planes: radiation plus slab-mode scattering
  deficit      1 - all forward - all backward: radiation including what leaves through the box ends
Mode conversion is never counted as radiation. Subtract a smooth reference of the same
layout (`subtract_reference`) before reading off alpha.
"""

from __future__ import annotations

import numpy as np

from .probe import alpha_from_power, kernel_from_probe

C0_UM_PER_S = 299792458.0e6


def _mode_k(label: str):
    return label[:2], int(label[2:])


def probe_budget(sim_data) -> dict:
    sim = sim_data.simulation
    attrs = dict(sim.attrs)
    _, k = _mode_k(attrs["mode"])
    amps_in = sim_data["in"].amps
    amps_out = sim_data["out"].amps
    f = amps_in.f.values
    p_in = np.abs(amps_in.sel(direction="+", mode_index=k).values) ** 2
    bwd = np.abs(amps_in.sel(direction="-").values) ** 2          # (f, mode)
    fwd = np.abs(amps_out.sel(direction="+").values) ** 2
    n_eff_out = np.real(sim_data["out"].n_eff.values)
    te_out = sim_data["out"].pol_fraction.te.values

    def flux(name, outward_sign):
        return outward_sign * np.asarray(sim_data[name].flux.values)

    rad_free = flux("rad_z+", 1) + flux("rad_z-", -1)
    rad_side = flux("rad_y+", 1) + flux("rad_y-", -1)
    others = [i for i in range(fwd.shape[1]) if i != k]
    res = {
        "attrs": attrs,
        "freq_hz": f,
        "lam_um": C0_UM_PER_S / f,
        "p_in": p_in,
        "same_fwd": fwd[:, k] / p_in,
        "other_fwd_by_mode": fwd[:, others] / p_in[:, None],
        "other_fwd": fwd[:, others].sum(axis=1) / p_in,
        "same_bwd": bwd[:, k] / p_in,
        "other_bwd": bwd[:, others].sum(axis=1) / p_in,
        "rad_free": rad_free / p_in,
        "rad_side": rad_side / p_in,
        "other_mode_n_eff": n_eff_out[:, others],
        "other_mode_te_fraction": te_out[:, others],
    }
    res["deficit"] = 1.0 - fwd.sum(axis=1) / p_in - bwd.sum(axis=1) / p_in
    return res


def subtract_reference(rough: dict, smooth: dict) -> dict:
    """Rough minus smooth for every loss channel (same layout, same frequencies)."""
    out = dict(rough)
    for key in ("rad_free", "rad_side", "deficit", "other_fwd", "other_bwd", "same_bwd"):
        out[key] = rough[key] - smooth[key]
    out["same_fwd"] = rough["same_fwd"] / smooth["same_fwd"]
    return out


def kernel_point(budget: dict, channel: str = "box") -> dict:
    """alpha and H(K) at K = 2 pi / Lambda from a reference-subtracted budget.

    channel "box" = rad_free + rad_side, "free" = rad_free, "deficit" = deficit.
    """
    a = budget["attrs"]
    rad = {"box": budget["rad_free"] + budget["rad_side"], "free": budget["rad_free"],
           "deficit": budget["deficit"]}[channel]
    alpha = alpha_from_power(rad, a["corrugated_length_um"])
    return {"K_per_um": 2 * np.pi / a["period_um"], "lam_um": budget["lam_um"], "alpha_per_um": alpha,
            "H": kernel_from_probe(alpha, a["amplitude_um"]), "walls": a["walls"]}
