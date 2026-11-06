"""Back-of-envelope calculators and interview-rubric helpers for system design.

Introduced in Level 3 notebooks 17-20 (staff-level design and the full loop).
Everything here is plain arithmetic on *stated assumptions*: no function knows
real prices, real traffic or any company's real system. The point is to make
the assumptions explicit, so that a reviewer (or an interviewer) can change one
input and see what moves.

Conventions
-----------
* Storage in **decimal** units: 1 GB = 1e9 bytes, 1 TB = 1e12 bytes.
* Money in whatever currency the price inputs use; prices are always inputs,
  never constants, because they change and differ by provider and contract.
* A month is 730 hours (8,760 / 12).

Sections: capacity, storage, compute & cost, LLM cost, latency budgets,
risk registers and decision records, interview rubrics and progress logs,
project briefs.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path

SECONDS_PER_DAY = 86_400
HOURS_PER_MONTH = 730
GB = 1e9
TB = 1e12

# --------------------------------------------------------------------------- assumptions


def show_assumptions(assumptions: Mapping[str, object], title: str = "ASSUMPTIONS") -> None:
    """Print a labelled block of assumptions, so every estimate carries its inputs."""
    print(f"{title} (inputs to the estimate, not measurements):")
    width = max(len(k) for k in assumptions) if assumptions else 0
    for k, v in assumptions.items():
        v = f"{v:,}" if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        print(f"  {k.ljust(width)} = {v}")


def table(rows: list[dict], floatfmt: str = "{:,.2f}") -> None:
    """Print a list of dicts as an aligned text table (same helper as Level 1 notebook 17)."""
    if not rows:
        print("(empty)")
        return
    cols = list(rows[0])
    cells = [[floatfmt.format(r[c]) if isinstance(r[c], float) else
              (f"{r[c]:,}" if isinstance(r[c], int) and not isinstance(r[c], bool) else str(r[c]))
              for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for row, raw in zip(cells, rows):
        print("  ".join(v.rjust(w) if isinstance(raw[c], (int, float)) and not isinstance(raw[c], bool)
                        else v.ljust(w) for v, w, c in zip(row, widths, cols)))


# --------------------------------------------------------------------------- capacity


def peak_qps(dau: float, requests_per_user_per_day: float, peak_factor: float) -> float:
    """Peak requests per second: daily requests / 86,400 x (peak / average)."""
    return dau * requests_per_user_per_day / SECONDS_PER_DAY * peak_factor


def replicas_needed(load_qps: float, per_replica_qps: float, target_utilisation: float = 0.6,
                    min_replicas: int = 2, zones: int = 1) -> int:
    """Replicas to serve ``load_qps`` at ``target_utilisation``, spread over ``zones``.

    With ``zones > 1`` the fleet is sized so that losing one whole zone still
    leaves enough capacity (N+1 at the zone level), and rounded up to a
    multiple of ``zones`` so each zone holds the same number.
    """
    if per_replica_qps <= 0 or not 0 < target_utilisation <= 1:
        raise ValueError("per_replica_qps must be > 0 and 0 < target_utilisation <= 1")
    base = math.ceil(load_qps / (per_replica_qps * target_utilisation))
    if zones > 1:
        per_zone = math.ceil(base / (zones - 1))  # survive the loss of one zone
        base = per_zone * zones
    return max(min_replicas, base)


# --------------------------------------------------------------------------- storage


def online_store_gb(entities: float, features: int, bytes_per_value: float, *,
                    key_bytes: int = 16, overhead: float = 1.5, replication: int = 3) -> float:
    """Approximate memory/SSD for an online key-value feature store (decimal GB).

    ``overhead`` covers hash-table/index structures, fragmentation and
    serialisation (1.3-2x is a common planning range; measure your store).
    """
    per_entity = key_bytes + features * bytes_per_value
    return entities * per_entity * overhead * replication / GB


def offline_store_tb(rows_per_day: float, columns: int, bytes_per_value: float, days_retained: int, *,
                     compression_ratio: float = 4.0, copies: int = 1) -> float:
    """Columnar offline store (e.g. Parquet in object storage), decimal TB after compression."""
    raw = rows_per_day * columns * bytes_per_value * days_retained
    return raw / compression_ratio * copies / TB


def storage_monthly_cost(gb: float, price_per_gb_month: float) -> float:
    return gb * price_per_gb_month


# --------------------------------------------------------------------------- compute and cost


def train_flops(params: float, tokens: float) -> float:
    """Dense-transformer training compute: ~6 x parameters x tokens (forward + backward)."""
    return 6.0 * params * tokens


def gpu_hours_from_flops(flops: float, peak_flops_per_gpu: float, mfu: float) -> float:
    """GPU-hours to do ``flops`` at model-FLOPs utilisation ``mfu`` (0-1) of the peak."""
    if not 0 < mfu <= 1:
        raise ValueError("mfu must be in (0, 1]")
    return flops / (peak_flops_per_gpu * mfu) / 3600


def monthly_fleet_cost(replicas: int, price_per_replica_hour: float, hours: float = HOURS_PER_MONTH) -> float:
    return replicas * price_per_replica_hour * hours


def cost_per_1k_requests(monthly_cost: float, avg_qps: float, hours: float = HOURS_PER_MONTH) -> float:
    requests = avg_qps * 3600 * hours
    return monthly_cost / requests * 1000 if requests else float("nan")


def llm_cost_per_conversation(turns: float, prompt_tokens_per_turn: float, output_tokens_per_turn: float,
                              price_in_per_million: float, price_out_per_million: float,
                              cached_fraction: float = 0.0, cached_discount: float = 0.0) -> float:
    """Token cost of one multi-turn conversation.

    ``prompt_tokens_per_turn`` should include the system prompt, the retrieved
    context and the conversation history re-sent each turn (the usual reason
    long conversations cost more than you expect). ``cached_fraction`` of the
    prompt tokens is billed at ``(1 - cached_discount)`` of the input price,
    if your provider offers prompt caching; leave both at 0 otherwise.
    """
    effective_in = price_in_per_million * (1 - cached_fraction * cached_discount)
    per_turn = prompt_tokens_per_turn * effective_in / 1e6 + output_tokens_per_turn * price_out_per_million / 1e6
    return turns * per_turn


# --------------------------------------------------------------------------- rollout and monitoring


def two_proportion_n(p_base: float, relative_change: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Samples per arm to detect ``p_base -> p_base * (1 + relative_change)`` (two-sided z-test).

    Used for canaries and A/B guardrails: "how long must the canary run before
    a +20% error-rate regression would be visible?"
    """
    from statistics import NormalDist

    p2 = p_base * (1 + relative_change)
    z_a, z_b = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    p_bar = (p_base + p2) / 2
    num = (z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(p_base * (1 - p_base) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p_base) ** 2)


def label_maturity(days_since_event: float, mean_delay_days: float) -> float:
    """Fraction of eventual labels already observed, assuming exponential label delay."""
    return 1 - math.exp(-days_since_event / mean_delay_days)


# --------------------------------------------------------------------------- latency


def latency_table(total_ms: float, stages: Mapping[str, float]) -> list[dict]:
    """Rows of (stage, ms, share of budget) plus a HEADROOM row. Does not validate."""
    rows = [{"stage": k, "ms": float(v), "share": float(v) / total_ms} for k, v in stages.items()]
    left = total_ms - sum(stages.values())
    rows.append({"stage": "HEADROOM", "ms": float(left), "share": left / total_ms})
    return rows


def sequential_p99_upper_bound(stage_p99s: Iterable[float]) -> float:
    """Sum of per-stage p99s: an upper bound on the end-to-end p99 of a sequential chain.

    Percentiles do not add exactly; the sum is pessimistic for independent
    stages and roughly right when slow requests are slow everywhere (correlated).
    """
    return float(sum(stage_p99s))


def fanout_tail_probability(per_call_slow_prob: float, fanout: int) -> float:
    """P(at least one of ``fanout`` parallel calls is slow) - why fan-out amplifies tails."""
    return 1 - (1 - per_call_slow_prob) ** fanout


# --------------------------------------------------------------------------- risk register and ADRs

LIKELIHOOD = {"rare": 1, "unlikely": 2, "possible": 3, "likely": 4, "almost certain": 5}
IMPACT = {"negligible": 1, "minor": 2, "moderate": 3, "major": 4, "severe": 5}


def risk_register(risks: list[dict]) -> list[dict]:
    """Score risks (likelihood x impact, 1-25) and sort, highest first.

    Each risk dict needs ``risk``, ``likelihood`` and ``impact`` (words from
    LIKELIHOOD / IMPACT or integers 1-5), ``mitigation`` and ``owner``.
    """
    out = []
    for r in risks:
        lk = LIKELIHOOD.get(r["likelihood"], r["likelihood"])
        im = IMPACT.get(r["impact"], r["impact"])
        if not (isinstance(lk, int) and isinstance(im, int) and 1 <= lk <= 5 and 1 <= im <= 5):
            raise ValueError(f"bad likelihood/impact in {r['risk']!r}")
        score = lk * im
        level = "HIGH" if score >= 15 else "MEDIUM" if score >= 8 else "LOW"
        out.append({"risk": r["risk"], "L": lk, "I": im, "score": score, "level": level,
                    "mitigation": r.get("mitigation", ""), "owner": r.get("owner", "")})
    return sorted(out, key=lambda d: (-d["score"], d["risk"]))


def adr_markdown(number: int, title: str, *, status: str, context: str, options: Mapping[str, str],
                 decision: str, consequences: Iterable[str], revisit_when: Iterable[str],
                 decided_on: str | None = None) -> str:
    """An Architecture Decision Record (ADR) in the common Nygard-style layout."""
    lines = [f"# ADR-{number:03d}: {title}", "",
             f"* **Status:** {status}", f"* **Date:** {decided_on or date.today().isoformat()}", "",
             "## Context", "", context.strip(), "", "## Options considered", ""]
    lines += [f"* **{name}:** {desc}" for name, desc in options.items()]
    lines += ["", "## Decision", "", decision.strip(), "", "## Consequences", ""]
    lines += [f"* {c}" for c in consequences]
    lines += ["", "## Revisit when", ""] + [f"* {c}" for c in revisit_when]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- interview rubrics

#: Generic leveling anchors ("signals interviewers commonly look for"). They describe
#: no specific company's ladder; use them to calibrate practice, not to label yourself.
LEVEL_ANCHORS = {
    "mid": "solves the stated problem correctly with some guidance; explains own choices",
    "senior": "drives the problem independently; quantifies trade-offs; anticipates failure modes and operations",
    "staff": "frames the problem beyond the prompt: multiple teams, platform reuse, long-term cost and risk, "
             "what to build first and what not to build; changes the design when evidence changes",
}

SCORE_ANCHORS = {1: "missing or wrong; needed heavy hints", 2: "partly there; gaps under follow-up",
                 3: "solid and independent; handles follow-ups", 4: "excellent; raises the bar"}


def validate_scores(scores: Mapping[str, int]) -> None:
    bad = {k: v for k, v in scores.items() if v not in (1, 2, 3, 4) or isinstance(v, bool)}
    if bad:
        raise ValueError(f"scores must be whole numbers 1-4, got {bad}")


def append_jsonl(path: Path, record: dict) -> int:
    """Append one JSON record to a .jsonl log (never rewrites history); return the record count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return len(read_jsonl(path))


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- portfolio


def scan_artifacts(art: Path, patterns: Iterable[str] = ("*.json", "*.md", "*.csv", "*.txt")) -> list[dict]:
    """List result-like files under an artifact directory (path, size, a short preview of JSON keys)."""
    art = Path(art)
    found = []
    for pattern in patterns:
        for p in sorted(art.rglob(pattern)):
            if not p.is_file() or p.stat().st_size > 5_000_000:
                continue
            entry = {"path": p.relative_to(art).as_posix(), "bytes": p.stat().st_size, "keys": ""}
            if p.suffix == ".json":
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(obj, dict):
                        entry["keys"] = ", ".join(list(obj)[:6])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    entry["keys"] = "(unreadable JSON)"
            elif p.suffix == ".csv":
                with p.open(encoding="utf-8", errors="replace") as f:
                    entry["keys"] = ", ".join(next(csv.reader(f), [])[:6])
            found.append(entry)
    return found
