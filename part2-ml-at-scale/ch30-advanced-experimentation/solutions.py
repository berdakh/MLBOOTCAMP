"""Reference solutions for notebook 12. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np


def ex1(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    var_x = np.var(x, ddof=1)
    if var_x == 0:
        return y.copy()
    theta = np.cov(x, y, ddof=1)[0, 1] / var_x
    return y - theta * (x - x.mean())


def ex2(num, den):
    c, n = np.asarray(num, float), np.asarray(den, float)
    mc, mn = c.mean(), n.mean()
    cov = np.cov(c, n, ddof=1)
    return float((cov[0, 0] / mn**2 - 2 * mc * cov[0, 1] / mn**3 + mc**2 * cov[1, 1] / mn**4) / len(c))


def ex3(list_a, list_b, rng):
    k = len(list_a)
    items, teams, shown = [], [], set()
    picks = {"A": 0, "B": 0}
    ranked = {"A": list(list_a), "B": list(list_b)}
    while len(items) < k:
        if picks["A"] < picks["B"] or (picks["A"] == picks["B"] and rng.random() < 0.5):
            team = "A"
        else:
            team = "B"
        candidates = [x for x in ranked[team] if x not in shown]
        if not candidates:  # this team has run out of new items: the other team picks
            team = "B" if team == "A" else "A"
            candidates = [x for x in ranked[team] if x not in shown]
            if not candidates:
                break
        items.append(candidates[0])
        teams.append(team)
        shown.add(candidates[0])
        picks[team] += 1
    return items, teams
