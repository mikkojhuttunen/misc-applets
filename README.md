# misc-applets

Physics engines and the interactive applets built on them.

| Folder | Contents |
|---|---|
| [math-engines/](math-engines/) | Small, tested, UI-free Python physics engines (SI units, `Result` objects, `spec.yaml`), a generated index and a generic Pyodide calculator. See the [tutorial](math-engines/docs/math-engines-tutorial.md). |
| [amplifiers/parametric-amplifiers/](amplifiers/parametric-amplifiers/) | Fiber parametric amplifier applet (χ⁽²⁾ and χ⁽³⁾). Its physics is a JavaScript port of the math engines, verified against shared test vectors. |

CI (`.github/workflows/math-engines-tests.yml`) runs the Python engine tests, checks that the web index
and the JavaScript test vectors are current, and tests the JavaScript port on every push.
