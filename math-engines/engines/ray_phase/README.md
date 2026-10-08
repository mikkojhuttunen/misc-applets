# ray_phase

Optical path and phase of rays launched from the input point of a segmented multipass cell at slightly different angles, pass by pass (`path_lengths`, `relative_phase`), and the fringe visibility left when the launch angle is dithered (`dither_offsets`, `visibility`, `dither_scan`). Sine dither of a linear phase reproduces |J₀(aA)| and triangle dither |sinc(aA)| (tested). Geometry from `billiard_cell.SegmentedCell`, so facet tilts and curvatures carry over.

Version 2 adds `dither_analysis`, which explains V pass by pass: the phase slope a_p = k₀n dL_p/dθ and curvature b_p (`phase_derivatives`), the dither depth x_p = a_p A with the linear-model prediction |J₀(x_p)| or |sinc(x_p)| (`model_visibility`), the chirp ½|b_p|A², the share of dithered rays that keep the centre ray's facet path, the amplitudes for model V = 0.9, 0.5 and 0, adaptive sampling until every pass is resolved, and the random-phase noise floor √π/(2√N). While every dithered ray keeps the centre path and the chirp is small, V equals the J₀/sinc prediction (tested); x scales as n A dL_p/dθ / λ, and in a regular cell |a_p| grows in proportion to p.

`first_mirror_phase` and `dither_visibility` return Results; the rest are helpers. The `cmpc-ray-tracer.html` applet carries a JS port checked against `test_vectors/vectors.json`.

Ray picture: V_p is the stability of each ray's own optical phase under the dither, i.e. what happens to an interference term between pass-p light and light that does not follow the dither. Reflection phases are ignored; neighbouring rays separate spatially after a few passes, so this is not an overlap-integral (wave) calculation.

Other cells: `path_lengths` also runs on any `planar_cell` wall (stadium, faceted stadium, circle, integrated Herriott; path id = element sequence, NaN once a ray leaves an open wall) and on any object with its own `path_lengths` method, such as `herriott_cell.HerriottLaunch` (3D Herriott cell, θ = injection-angle tilt), so `dither_analysis` applies to them unchanged. Front ends for those cells are in `planar_cell` and `herriott_cell`.
