"""Self-checks for notebook 12 · Packaging & configuration."""

from __future__ import annotations

from ..checks import run_check

GOOD_DOCKERFILE = """# syntax=docker/dockerfile:1
FROM python:3.12-slim
WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY src/ src/
RUN pip install --no-cache-dir \\
        --no-deps .
RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0"]
"""

BAD_DOCKERFILE = """FROM python:latest
WORKDIR /app
COPY .env /app/.env
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "serve.py"]
"""

ALL_CODES = ["copies_env_file", "latest_tag", "no_healthcheck", "pip_cache", "root_user"]


def ex1(dockerfile_problems):
    def check(text, expected, why):
        got = dockerfile_problems(text)
        assert isinstance(got, list), f"return a list, got {type(got).__name__}"
        assert got == sorted(set(got)), f"return a sorted list without duplicates, got {got}"
        assert got == expected, f"{why}: expected {expected}, got {got}"

    def test():
        check(GOOD_DOCKERFILE, [], "a clean Dockerfile")
        check(BAD_DOCKERFILE, ALL_CODES, "a Dockerfile with all five problems")
        base = GOOD_DOCKERFILE
        check(base.replace("FROM python:3.12-slim", "FROM python"), ["latest_tag"], "FROM without a tag")
        check(base.replace("FROM python:3.12-slim", "from --platform=linux/amd64 python:latest AS build"),
              ["latest_tag"], "lower-case FROM with --platform, :latest and AS")
        check(base.replace("FROM python:3.12-slim", "FROM python:3.12-slim@sha256:" + "a" * 64),
              [], "a digest-pinned image")
        check(base.replace("FROM python:3.12-slim", "FROM python:3.12-slim AS build\nFROM scratch"),
              [], "multi-stage with FROM scratch")
        check(base.replace("USER app", "USER app\nUSER root"), ["root_user"], "last USER is root")
        check(base.replace("USER app", "USER 0"), ["root_user"], "USER 0 is root")
        check(base.replace("USER app\n", ""), ["root_user"], "no USER at all")
        check(base.replace("COPY src/ src/", "COPY src/ src/\nADD config/.env.prod /app/"),
              ["copies_env_file"], "ADD of a .env.prod file")
        check(base.replace("COPY src/ src/", "COPY src/ src/\nCOPY environment.yml ."),
              [], "environment.yml is not a .env file")
        check(base.replace("HEALTHCHECK CMD", "# HEALTHCHECK CMD"), ["no_healthcheck"], "commented-out HEALTHCHECK")
        check(base.replace("HEALTHCHECK CMD python", "HEALTHCHECK NONE\n# python"), ["no_healthcheck"],
              "HEALTHCHECK NONE")
        check(base.replace("--no-cache-dir -r", "-r"), ["pip_cache"], "pip install without --no-cache-dir")
        check(base + "# remember: pip install foo\n", [], "a comment mentioning pip install")
        check(base.replace("RUN pip install --no-cache-dir \\\n        --no-deps .",
                           "RUN pip install \\\n        --no-deps . && rm -rf /tmp/x"),
              ["pip_cache"], "a multi-line RUN without --no-cache-dir")

    return run_check("Exercise 1 · dockerfile_problems", test,
                     hint="first join continuation lines and drop comments; then look at each instruction's "
                          "first word (upper-cased) and the rest of the line")


def ex2(settings_from_env):
    def expect_error(env, variable):
        try:
            settings_from_env(env)
        except ValueError as exc:
            assert variable in str(exc), f"the error for {env} should name {variable}, got: {exc}"
        else:
            raise AssertionError(f"{env} should raise ValueError")

    def test():
        got = settings_from_env({})
        assert got == {"port": 8000, "workers": 1, "log_level": "INFO"}, f"defaults wrong: {got}"
        got = settings_from_env({"CHURN_PORT": "9000", "CHURN_WORKERS": "4", "CHURN_LOG_LEVEL": "debug",
                                 "PORT": "1", "HOME": "/root"})
        assert got == {"port": 9000, "workers": 4, "log_level": "DEBUG"}, f"parsed values wrong: {got}"
        assert type(got["port"]) is int and type(got["workers"]) is int, "port and workers must be int"
        got = settings_from_env({"CHURN_PORT": "", "CHURN_WORKERS": "", "CHURN_LOG_LEVEL": ""})
        assert got == {"port": 8000, "workers": 1, "log_level": "INFO"}, "empty strings must mean 'use default'"
        assert settings_from_env({"CHURN_PORT": "65535"})["port"] == 65535, "65535 is a valid port"
        assert settings_from_env({"CHURN_WORKERS": "64"})["workers"] == 64, "64 workers is allowed"
        for bad in ("abc", "0", "70000", "-1", "80.5"):
            expect_error({"CHURN_PORT": bad}, "CHURN_PORT")
        for bad in ("0", "65", "two", "2.5"):
            expect_error({"CHURN_WORKERS": bad}, "CHURN_WORKERS")
        for bad in ("LOUD", "verbose", "trace"):
            expect_error({"CHURN_LOG_LEVEL": bad}, "CHURN_LOG_LEVEL")

    return run_check("Exercise 2 · settings_from_env", test,
                     hint="loop over the three variables; treat None and '' as missing; wrap int() in "
                          "try/except and re-raise ValueError(f'CHURN_PORT must be ..., got {raw!r}')")
