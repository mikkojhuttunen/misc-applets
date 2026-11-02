QUIET simulation bundle (extracted from the Chaos Lab and Fig. 2 artifacts)

engine/engine_fig2_patched.js : PlanarEngine, GasEngine, DitherLab from the Fig. 2 artifact, with membrane loss exp(-alpha_m L) added
                                (option alpha_m = power loss in 1/m; 0.1 dB/cm = 2.303 /m). module.exports at the end.
engine/engine_lab_patched.js  : same for the Chaos Lab artifact (used by ens.js for the Fig. 3 ensembles)
scripts/ens.js   : Fig. 3 ensembles. node ens.js R dB_per_cm nSeeds out.json (needs fig3data.json in cwd; engine_lab_patched.js as ./eng_loss.js)
scripts/ens2.js  : Fig. 2 ensembles. node ens2.js R dB_per_cm circle|herriott nSeeds out.json (engine as ./eng2.js)
scripts/crit.js  : fraction of bounces below the critical angle; writes sinhist.json (sin chi histogram)
scripts/build_fig*.py : build the ideal/minimum/target artifact copies (need the original artifact HTML)
design_calcs/slab.py : free-standing slab TM0/TE0 n_eff and gas fraction (reproduces Gamma 0.64 for 250 nm Si)
design_calcs/tmm.py  : effective-index trench DBR transfer matrix (s-pol), reflectivity vs angle
results/ : ensemble outputs: ens_base (R .995, 0 dB/cm), ens_new (R .95, 0.3), ens_tgt (R .99, 0.1); f2_* are Fig. 2 ensembles
Settings: circle 4 mm, 24 facets, 16 inclusions dn 0.01, launch 7.5 deg from the wall normal, eps 1e-6, +-2 mrad, +-20 pm, 1 mW, Gamma 0.2.
Runtime: about 5-15 s per layout per cell; four parallel jobs of 30 layouts took 10-25 min.
