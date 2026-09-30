"""Prevent accidental publication of personal paths, contact emails, or credentials."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_tracked_files_do_not_contain_private_metadata():
    try:
        paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("requires a Git checkout")
    patterns = {
        "personal home path": re.compile(r"/(?:Users|home)/[\w.-]+/"),
        "private key": re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
        "credential token": re.compile(
            r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|"
            r"github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}"
        ),
    }
    emails = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    findings = []
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for label, pattern in patterns.items():
                if pattern.search(line):
                    findings.append(f"{name}:{line_number}: {label}")
            for email in emails.findall(line):
                domain = email.rsplit("@", 1)[1].lower()
                if domain not in {
                    "example.com",
                    "example.org",
                    "example.invalid",
                    "users.noreply.github.com",
                }:
                    findings.append(f"{name}:{line_number}: non-placeholder email")
    assert not findings, "\n".join(findings)
