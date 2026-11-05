"""Insert the generated tables (results/report_tables.md) into REPORT.md at the <<T1>> ... <<T7>> markers."""
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parent.parent
T = (ROOT / "results" / "report_tables.md").read_text()
secs = {m.group(1): m.group(2).strip() for m in re.finditer(r"### (Task \d[^\n]*)\n(.*?)(?=\n### |\Z)", T, re.S)}
pick = lambda prefix: "\n\n".join(f"**{k}**\n\n{v}" for k, v in secs.items() if k.startswith(prefix))
R = (ROOT / "REPORT.md").read_text()
for tag, prefix in (("<<T1>>", "Task 1"), ("<<T2>>", "Task 2"), ("<<T3>>", "Task 3"), ("<<T4>>", "Task 4"), ("<<T7>>", "Task 7")):
    if tag in R:
        R = R.replace(tag, pick(prefix))
(ROOT / "REPORT.md").write_text(R)
print("ok", [k for k in secs])
