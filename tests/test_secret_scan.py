"""The tree carries no key, and the scanner would notice one."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_tracked_tree_is_clean():
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "secret_scan.py")], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_scanner_catches_the_shapes_of_real_keys(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import secret_scan  # noqa: E402

    # Built at runtime so this file itself never holds a key-shaped literal.
    jwt = ".".join(["eyJ" + "a" * 30, "eyJ" + "b" * 30, "c" * 20])
    pem = "-----BEGIN " + "RSA PRIVATE KEY-----"
    bad = tmp_path / "config.py"
    bad.write_text(
        "\n".join(
            [
                "PAYPAL_CLIENT_ID = 'A" + "b1" * 39 + "b'",  # 80 chars, as PayPal issues them
                "NEBIUS_API_KEY=" + jwt,
                "DATABASE_URL=" + "postgres" + "://user:supersecretpw@host:5432/db",
                pem,
            ]
        )
    )
    findings = secret_scan.scan([bad])
    labels = " ".join(findings)
    assert "PayPal client id" in labels and "Nebius JWT" in labels and "Postgres URL" in labels and "private key" in labels

    fine = tmp_path / "docs.md"
    fine.write_text("PAYPAL_CLIENT_SECRET=change-me\nNEBIUS_API_KEY=your_nebius_token_factory_key_here\nDEMO_ACCESS_CODE=the-code\n")
    assert secret_scan.scan([fine]) == []

    env = tmp_path / ".env"
    env.write_text("x=1\n")
    assert any("never be committed" in finding for finding in secret_scan.scan([env]))
