#!/usr/bin/env python3
"""Derive environment variables from a Dockerfile's ENV instructions.

Parses single and multi-line ENV instructions and prints the variables in
several formats suitable for use locally (e.g., export lines or .env).

Usage examples:
- Print KEY=VALUE lines:
    python3 scripts/docker_env_from_dockerfile.py
- Print export lines and eval in current shell:
    eval "$(python3 scripts/docker_env_from_dockerfile.py --format export)"
- Generate a .env file:
    python3 scripts/docker_env_from_dockerfile.py --format dotenv --out .env.generated
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from collections import OrderedDict
from typing import Dict, List, Tuple, Iterable


def _read_dockerfile(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def _iter_env_instructions(lines: List[str]) -> Iterable[str]:
    """Yield the content following ENV across line continuations.

    - Handles trailing backslashes and skips pure comment lines between
      continuations (common in hand-edited Dockerfiles).
    - Preserves quoting so that shlex can parse values with spaces.
    """
    i = 0
    n = len(lines)
    env_prefix = re.compile(r"^\s*ENV\s+(.*)$", re.IGNORECASE)
    while i < n:
        line = lines[i].rstrip("\n")
        m = env_prefix.match(line)
        if not m:
            i += 1
            continue
        # Start of an ENV instruction
        first = m.group(1).rstrip()
        acc = re.sub(r"\\\s*$", "", first)
        continued = bool(re.search(r"\\\s*$", first))
        i += 1
        # Continue consuming following lines while the previous line ended with '\'
        while continued and i < n:
            # Read next non-empty line (comments are skipped but do not end continuation)
            next_line = lines[i].rstrip("\n")
            i += 1
            if re.match(r"^\s*#", next_line):
                # Skip comments inside continuation without changing state
                continue
            # Determine if this line itself continues
            stripped = next_line.rstrip()
            part = re.sub(r"\\\s*$", "", stripped).lstrip()
            acc += " " + part
            continued = bool(re.search(r"\\\s*$", stripped))
        yield acc
    return


_ENV_PREFIX = re.compile(r"^\s*ENV\s+(.*)$", re.IGNORECASE)
_ASSIGN = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<val>.*)$")
_VAR = re.compile(r"\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)|\$\{(?P<braced>[^}]+)\}")


def _expand(value: str, env: Dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        name = m.group("name") or m.group("braced")
        if name in env:
            return env[name]
        return m.group(0)  # leave as-is if unknown

    return _VAR.sub(repl, value)


def _parse_env_tokens(tokens: List[str], env: Dict[str, str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not tokens:
        return pairs
    # Two forms are allowed:
    # 1) ENV key value
    # 2) ENV key1=value1 key2=value2 ...
    if "=" not in tokens[0]:
        # key value form (value may include spaces if quoted; shlex already handled quotes)
        key = tokens[0]
        val = " ".join(tokens[1:]) if len(tokens) > 1 else ""
        pairs.append((key, _expand(val, env)))
    else:
        for t in tokens:
            m = _ASSIGN.match(t)
            if not m:
                # Ignore unexpected token subtly
                continue
            key = m.group("key")
            val = m.group("val")
            pairs.append((key, _expand(val, env)))
    return pairs


def _collect_env_from_dockerfile(path: str) -> "OrderedDict[str, str]":
    lines = _read_dockerfile(path)
    env: "OrderedDict[str, str]" = OrderedDict()
    for rest in _iter_env_instructions(lines):
        # Use shlex to split while respecting quotes and comments
        lex = shlex.shlex(rest, posix=True)
        lex.whitespace_split = True
        # Ensure # starts comments only outside quotes
        lex.commenters = "#"
        tokens = list(lex)
        for k, v in _parse_env_tokens(tokens, env):
            env[k] = v
    return env


def _quote_export(val: str) -> str:
    # Use double quotes and escape $, `, \\, and "
    val = val.replace("\\", "\\\\").replace("\"", "\\\"").replace("`", "\\`").replace("$", "\\$")
    return f'"{val}"'


def _quote_dotenv(val: str) -> str:
    # If safe (alnum, punctuation w/o spaces or control), leave unquoted; else double-quote and escape
    if re.fullmatch(r"[A-Za-z0-9_./:@+-]*", val or ""):
        return val
    # Escape newlines and quotes minimally
    val = val.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\"", "\\\"")
    return f'"{val}"'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dockerfile", nargs="?", default="Dockerfile", help="Path to Dockerfile")
    ap.add_argument("--format", choices=["env", "export", "dotenv"], default="env", help="Output format")
    ap.add_argument("--out", help="Write to file instead of stdout")
    args = ap.parse_args()

    env = _collect_env_from_dockerfile(args.dockerfile)

    lines: List[str] = []
    if args.format == "env":
        lines = [f"{k}={v}" for k, v in env.items()]
    elif args.format == "export":
        lines = [f"export {k}={_quote_export(v)}" for k, v in env.items()]
    elif args.format == "dotenv":
        lines = [f"{k}={_quote_dotenv(v)}" for k, v in env.items()]

    content = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
