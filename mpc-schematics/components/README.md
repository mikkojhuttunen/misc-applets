# Component graphics

`setup_schematic.py` draws the bulk optics with SVGs from this folder when they exist, otherwise with built-in
vector stand-ins. To use Alexander Franzen's ComponentLibrary (https://www.gwoptics.org/ComponentLibrary/):

1. Download the library SVG from that page and open it in Inkscape.
2. For each component: select its group, copy, File > New, paste in place, Edit > Resize Page to Selection
   (Ctrl+Shift+R), and save as plain SVG here under the name below. Orient it with the beam along +x (light enters
   from the left, leaves to the right); the script rotates it where needed.

| File | Library component (suggestion) |
|---|---|
| `laser.svg` | laser (box with output aperture) |
| `isolator.svg` | Faraday isolator |
| `lens.svg` | biconvex / plano-convex lens |
| `photodiode.svg` | photodiode |

License: the ComponentLibrary is CC BY-NC 3.0 (attribution, non-commercial). When any library SVG is drawn, the
script prints the attribution line in the figure (see `components.ATTRIBUTION`); keep it in published figures. Check
the license terms on the library page before committing the SVGs here or using the figures commercially.
