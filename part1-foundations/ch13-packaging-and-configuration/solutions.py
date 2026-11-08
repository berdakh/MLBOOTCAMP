"""Reference solutions for notebook 12. Graded by tests/test_solutions.py."""

from __future__ import annotations


def _instructions(text: str) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) pairs, with comments dropped and continuations joined."""
    logical, buffer = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if not buffer and (not line or line.startswith("#")):
            continue
        if buffer and line.startswith("#"):  # comment inside a continued instruction
            continue
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        logical.append(buffer + line)
        buffer = ""
    if buffer:
        logical.append(buffer)
    out = []
    for line in logical:
        parts = line.split(None, 1)
        out.append((parts[0].upper(), parts[1] if len(parts) > 1 else ""))
    return out


def _image_unpinned(args: str) -> bool:
    words = [w for w in args.split() if not w.startswith("--")]
    if not words:
        return True
    image = words[0]
    if image.lower() == "scratch" or "@sha256:" in image:
        return False
    last = image.rsplit("/", 1)[-1]  # a registry host may carry a port: host:5000/name
    if ":" not in last:
        return True
    return last.rsplit(":", 1)[1] == "latest"


def ex1(text: str) -> list[str]:
    instructions = _instructions(text)
    problems = set()
    users = [args.strip() for name, args in instructions if name == "USER"]
    if not users or users[-1].split(":")[0] in ("root", "0"):
        problems.add("root_user")
    healthchecks = [args.strip().upper() for name, args in instructions if name == "HEALTHCHECK"]
    if not healthchecks or healthchecks[-1] == "NONE":
        problems.add("no_healthcheck")
    for name, args in instructions:
        if name == "FROM" and _image_unpinned(args):
            problems.add("latest_tag")
        if name in ("COPY", "ADD"):
            for word in args.split():
                base = word.rstrip("/").rsplit("/", 1)[-1]
                if base == ".env" or base.startswith(".env."):
                    problems.add("copies_env_file")
        if name == "RUN" and "pip install" in args and "--no-cache-dir" not in args:
            problems.add("pip_cache")
    return sorted(problems)


LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def _int_setting(env: dict, name: str, default: int, low: int, high: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer from {low} to {high}, got {raw!r}") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be from {low} to {high}, got {value}")
    return value


def ex2(env: dict) -> dict:
    level = env.get("CHURN_LOG_LEVEL") or "INFO"
    if level.upper() not in LEVELS:
        raise ValueError(f"CHURN_LOG_LEVEL must be one of {', '.join(LEVELS)}, got {level!r}")
    return {
        "port": _int_setting(env, "CHURN_PORT", 8000, 1, 65535),
        "workers": _int_setting(env, "CHURN_WORKERS", 1, 1, 64),
        "log_level": level.upper(),
    }
