# ray_phase

Optical path and phase of rays launched from the input point of a segmented multipass cell at slightly different angles, pass by pass (`path_lengths`, `relative_phase`), and the fringe visibility left when the launch angle is dithered (`dither_offsets`, `visibility`, `dither_scan`). Sine dither of a linear phase reproduces |J₀(aA)| and triangle dither |sinc(aA)| (tested). Geometry from `billiard_cell.SegmentedCell`, so facet tilts and curvatures carry over.

Version 1. `first_mirror_phase` and `dither_visibility` return Results; the rest are helpers. The `cmpc-ray-tracer.html` applet carries a JS port checked against `test_vectors/vectors.json`.

Ray picture: V_p is the stability of each ray's own optical phase under the dither, i.e. what happens to an interference term between pass-p light and light that does not follow the dither. Reflection phases are ignored; neighbouring rays separate spatially after a few passes, so this is not an overlap-integral (wave) calculation.
