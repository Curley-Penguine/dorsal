"""Build the self-contained hosted demo page from the web console and the engine bundle.

    python scripts/build_static.py docs/demo_bundle.json out.html
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
html = (root / "dorsal" / "web" / "index.html").read_text()
bundle = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
head = re.search(r"<head>(.*)</head>", html, re.S).group(1)
body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
head = re.sub(r'<meta charset="utf-8">\s*', "", head)
head = re.sub(r'<meta name="viewport"[^>]*>\s*', "", head)
inject = "<script>window.DORSAL_BUNDLE = " + bundle.replace("</", "<\\/") + ";</script>\n"
body = body.replace("<script>\n(() => {", inject + "<script>\n(() => {", 1)
out.write_text(head.strip() + "\n" + body.strip() + "\n")
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
