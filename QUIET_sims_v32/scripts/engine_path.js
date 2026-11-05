/* engine of the QUIET v32 simulations: engine/quiet_engine.js (copy of quiet_sims/engine/quiet_engine.js: the Fig. 2
   artifact engine PlanarEngine / GasEngine / DitherLab with membrane loss and R(|sin chi|) tables) */
const path = require('path');
const ENGINE = path.join(__dirname, '..', 'engine', 'quiet_engine.js');
function loadEngine() { return require(ENGINE); }
module.exports = { ENGINE, loadEngine, DATA: path.join(__dirname, '..', 'data'), RESULTS: path.join(__dirname, '..', 'results') };
