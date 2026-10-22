# Component graphics

`setup_schematic.py` draws the bulk optics with SVGs from this folder when they exist, otherwise with built-in
vector stand-ins. To use Alexander Franzen's ComponentLibrary (https://www.gwoptics.org/ComponentLibrary/):

1. Download the library SVG from that page and open it in Inkscape.
2. For each component: select its group, copy, File > New, paste in place, Edit > Resize Page to Selection
   (Ctrl+Shift+R), and save as plain SVG here under the name below. Orient it with the beam along +x (light enters
   from the left, leaves to the right); the script rotates it where needed.

| File | Source (ComponentLibrary file) | Used in `setup_schematic.py` |
|---|---|---|
| `laser.svg` | `c-laser1.svg` | laser head (output face on the right) |
| `isolator.svg` | `c-isolator.svg` | Faraday isolator |
| `lens.svg` | `b-lens2.svg` (biconvex) | coupling and collecting lenses (sized by aperture) |
| `photodiode.svg` | `e-pd1.svg` (red) | signal detectors |
| `photodiode_ref.svg` | `e-pdgrn1.svg` (green) | reference detector |
| `lens1.svg`, `lens2.svg`, `lens3.svg` | `b-lens1/2/3.svg` | spares: plano-convex, biconvex, biconcave |
| `fiber_coupler.svg` | `b-coupler.svg` | spare (fibre coupler) |
| `eom.svg` | `c-eom1.svg` | spare (external modulator, if dithering moves off-chip) |
| `pd_symbol.svg` | `e-pd2.svg` | spare (electrical photodiode symbol) |

The files here are cleaned copies of the uploaded library files: the Illustrator DTD entities are resolved and the
metadata is removed; the drawings are unchanged. Each file names its source in a header comment.

License: the ComponentLibrary is CC BY-NC 3.0 (attribution, non-commercial). When any library SVG is drawn, the
script prints the attribution line in the figure (see `components.ATTRIBUTION`); keep it in published figures. Check
the license terms on the library page before committing the SVGs here or using the figures commercially.
