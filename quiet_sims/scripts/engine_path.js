/* engine selection for the scripts: --engine quiet (default, engine/quiet_engine.js), fig2 or lab (the originals) */
const path = require('path');
const ENGINES = {
  quiet: path.join(__dirname, '..', 'engine', 'quiet_engine.js'),
  fig2: path.join(__dirname, '..', 'engine', 'original', 'engine_fig2_patched.js'),
  lab: path.join(__dirname, '..', 'engine', 'original', 'engine_lab_patched.js'),
};
function takeFlag(argv, name, def) {
  const i = argv.indexOf('--' + name);
  if (i < 0) return def;
  const v = argv[i + 1];
  argv.splice(i, 2);
  return v;
}
function loadEngine(argv, def = 'quiet') {
  const key = takeFlag(argv, 'engine', def);
  if (!ENGINES[key]) throw new Error(`unknown engine ${key}: use quiet, fig2 or lab`);
  return require(ENGINES[key]);
}
module.exports = { ENGINES, takeFlag, loadEngine, RESULTS: path.join(__dirname, '..', 'results') };
