"""Assemble the explorer applet: web/explorer.src.html + web/ngrc-engine.js + results/progress.json + TASKS.md →
web/cmpc-ngrc-explorer.html (complete document) and web/cmpc-ngrc-explorer.artifact.html (body fragment for
publishing as a claude.ai artifact)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tasks():
    out, section = [], ""
    for line in (ROOT / "TASKS.md").read_text().splitlines():
        h = re.match(r"^## (\d+)\. (.+)", line)
        if h:
            section = h.group(2)
            continue
        m = re.match(r"^- \[([ x~])\] \*\*(T[\d.]+) (.+?)\.?\*\*(.*)", line)
        if m:
            st = {"x": "done", "~": "part", " ": "todo"}[m.group(1)]
            note = ""
            n = re.search(r"\*Now:\*\s*(.+)", m.group(4))
            if n:
                note = n.group(1).strip()
            out.append(dict(id=m.group(2), title=m.group(3).rstrip("."), status=st, section=section, note=note))
    return out


def main():
    src = (ROOT / "web" / "explorer.src.html").read_text()
    engine = (ROOT / "web" / "ngrc-engine.js").read_text()
    pj = ROOT / "results" / "progress.json"
    progress = json.loads(pj.read_text()) if pj.exists() else None
    body = (src.replace("/*__ENGINE__*/", engine)
               .replace("/*__PROGRESS__*/null", json.dumps(progress, separators=(",", ":")))
               .replace("/*__TASKS__*/[]", json.dumps(tasks(), ensure_ascii=False, separators=(",", ":"))))
    (ROOT / "web" / "cmpc-ngrc-explorer.artifact.html").write_text(body)
    title = re.search(r"<title>.*?</title>", body).group(0)
    rest = body.replace(title, "", 1)
    head, _, tail = rest.partition("</style>")
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            f"{title}\n{head}</style>\n</head>\n<body>\n{tail}\n</body>\n</html>\n")
    (ROOT / "web" / "cmpc-ngrc-explorer.html").write_text(full)
    print(f"built web/cmpc-ngrc-explorer.html ({len(full) // 1024} kB), {len(tasks())} tasks, "
          f"progress: {'yes' if progress else 'no'}")


if __name__ == "__main__":
    main()
