# Run from the repository root.  make test | make figs-nir | make midir | make chaos | make multibeam | make proposal-figs
PY ?= python3
test:            ; $(PY) tests/test_core.py
figs-nir:        ; $(PY) scripts/run_figures.py                 # 10 figures, Si 1.55 um (results/figs)
midir:           ; $(PY) scripts/run_midir.py                   # Ge/Si 5.26 um (results/figs_midir)
mirror:          ; $(PY) scripts/design_mirror.py               # three-index TE mirror designs -> data/mirror_designs.json
chaos:           ; $(PY) scripts/chaos_scan.py && $(PY) scripts/budget_scan.py && $(PY) scripts/run_chaos_figs.py   # ~3 min
multibeam:       ; $(PY) scripts/multibeam_nir.py && $(PY) scripts/make_prelim_figs2.py                             # ~5 min
single-port:     ; $(PY) scripts/single_port_scan.py && $(PY) scripts/single_port_scan_ge.py && $(PY) scripts/single_port_figs.py
proposal-figs:   ; $(PY) scripts/make_prelim_figs.py && $(PY) scripts/make_prelim_figs2.py
lines-mir:       ; $(PY) scripts/fetch_hitran.py mir && $(PY) scripts/select_lines.py NO 1880 1920
