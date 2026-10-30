#!/usr/bin/env python3
"""
Refuse to commit a key. Scans the tracked files (or the paths given) for
strings that look like live credentials — PayPal, Nebius/OpenAI-style, AWS,
GitHub, private keys, Heroku — and for a `.env` that slipped in.

    python tools/secret_scan.py              # every tracked file
    python tools/secret_scan.py README.md    # some files

Exit 1 with the offending lines when something is found. The tests run it
over the tree, and it is cheap enough for a pre-commit hook:

    printf '#!/bin/sh\npython tools/secret_scan.py\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

Placeholders (`change-me`, `your_..._here`, `sb-client-id`, `<fill in...>`)
are allowed on purpose; the patterns look for the shapes real keys have.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("PayPal client id", re.compile(r"\bA[A-Za-z0-9_-]{79}\b")),
    ("PayPal secret", re.compile(r"\bE[A-Za-z0-9_-]{79}\b")),
    ("PayPal access token", re.compile(r"\bA21AA[A-Za-z0-9_-]{40,}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("Nebius JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("Heroku API key", re.compile(r"\bHRKU-[A-Za-z0-9_-]{20,}\b")),
    ("Postgres URL with password", re.compile(r"\bpostgres(?:ql)?://[^:\s/]+:[^@\s/]{8,}@[^\s]+")),
    ("assigned secret", re.compile(r"(?i)\b(?:PAYPAL_CLIENT_SECRET|NEBIUS_API_KEY|DEMO_ACCESS_CODE|ADMIN_DEMO_KEY)\s*=\s*['\"]?(?!your_|change-me|the-code|sb-|<|\$|test|\s|$)[A-Za-z0-9_\-]{24,}")),
]
SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".mp4", ".mov", ".svg", ".ico", ".woff", ".woff2", ".pyc", ".sqlite")
SKIP_PARTS = (".git/", ".venv/", "node_modules/", "__pycache__/", "samples/", "out/")
FORBIDDEN_FILES = (".env",)


def tracked_files() -> List[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], check=True, capture_output=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [path for path in Path(".").rglob("*") if path.is_file()]
    return [Path(part.decode("utf-8")) for part in out.split(b"\0") if part]


def scan(paths: Iterable[Path]) -> List[str]:
    findings: List[str] = []
    for path in paths:
        text_path = str(path).replace("\\", "/")
        if any(part in text_path for part in SKIP_PARTS) or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.name in FORBIDDEN_FILES:
            findings.append(f"{path}: a .env file must never be committed")
            continue
        if path.name == Path(__file__).name:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, start=1):
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append(f"{path}:{number}: looks like a {label}: {line.strip()[:80]}")
    return findings


def main(argv: List[str]) -> int:
    paths = [Path(arg) for arg in argv] if argv else tracked_files()
    findings = scan(paths)
    if findings:
        print("Possible secrets found — do not commit:", file=sys.stderr)
        for finding in findings:
            print("  " + finding, file=sys.stderr)
        return 1
    print(f"secret scan: {len(list(paths))} files clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
