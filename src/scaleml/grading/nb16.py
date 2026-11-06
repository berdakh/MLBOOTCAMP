"""Self-checks for Level 3 notebook 16 · Evaluating LLM systems & guardrails."""

from __future__ import annotations

from churnlab.checks import run_check

#: the policy target for Exercise 3 (stated in the notebook)
MIN_RECALL, MAX_FPR = 0.80, 0.05


def ex1(field_f1):
    def test():
        gold = {"name": "Dana Smith", "order_id": "48213", "product": "blender", "issue": "damaged"}
        f = float(field_f1(dict(gold), gold))
        assert abs(f - 1.0) < 1e-9, f"identical dicts must score 1.0, got {f}"
        f = float(field_f1({"name": " dana smith ", "order_id": "48213", "product": "Blender", "issue": "damaged"}, gold))
        assert abs(f - 1.0) < 1e-9, f"compare values case- and whitespace-insensitively (strip, lower), got {f}"
        pred = {"name": "Dana Smith", "order_id": "48218", "product": "blender"}
        f = float(field_f1(pred, gold))  # P = 2/3, R = 2/4 -> F1 = 4/7
        assert abs(f - 4 / 7) < 1e-9, f"P = 2/3 (2 of 3 predicted fields right), R = 2/4 -> F1 = 4/7 = 0.5714; got {f:.4f}"
        pred = {"name": "Dana Smith", "order_id": "48213", "product": "blender", "issue": "damaged", "priority": "high"}
        f = float(field_f1(pred, gold))  # P = 4/5, R = 1
        assert abs(f - 8 / 9) < 1e-9, f"an extra key the gold lacks is a wrong prediction: P = 4/5, R = 1, F1 = 8/9; got {f:.4f}"
        f = float(field_f1({"name": "", "order_id": None, "product": "blender"}, gold))  # P = 1/1, R = 1/4
        assert abs(f - 0.4) < 1e-9, f"empty or None values count as 'not predicted': P = 1, R = 1/4, F1 = 0.4; got {f:.4f}"
        assert float(field_f1({}, gold)) == 0.0, "an empty prediction scores 0"
        assert float(field_f1({"order_id": 48213}, {"order_id": "48213"})) == 1.0, "compare values as strings"

    return run_check("Exercise 1 · field_f1", test,
                     hint="norm = lambda v: str(v).strip().lower(); predicted = keys with non-empty values; "
                          "tp = predicted keys in gold with equal normalised values; P = tp/len(predicted), R = tp/len(gold)")


def ex2(position_bias_rate):
    def test():
        ab = ["A", "A", "B", "A", "B"]
        ba = ["B", "A", "A", "A", "B"]
        r = float(position_bias_rate(ab, ba))
        assert abs(r - 0.6) < 1e-9, (
            f"pairs 2, 4 and 5 give the same LETTER in both orders (the position won, not the content): 3/5 = 0.6; got {r}")
        assert float(position_bias_rate(["A", "B"], ["B", "A"])) == 0.0, "fully consistent judgments have rate 0"
        assert float(position_bias_rate(["A"] * 4, ["A"] * 4)) == 1.0, "always picking the first slot has rate 1"
        assert float(position_bias_rate(["a", "B"], ["b", "b"])) == 0.5, "accept lower-case letters too"

    return run_check("Exercise 2 · position_bias_rate", test,
                     hint="the same candidate is 'A' in the first order and 'B' in the second; a consistent judge "
                          "flips the letter. rate = mean(ab[i].upper() == ba[i].upper())")


def ex3(guardrail):
    def test():
        import numpy as np

        from scaleml import llmeval as le

        texts, labels, kinds = le.make_guardrail_set(400, seed=987_654, families=tuple(le._INJECTION_FAMILIES),
                                                     heldout_goals=True)
        flags = np.array([bool(guardrail(t)) for t in texts])
        recall = flags[labels == 1].mean()
        fpr = flags[labels == 0].mean()
        hidden = np.array([k.split(":")[-1] in le.HIDDEN_FAMILIES for k in kinds])
        msg = (f"recall {recall:.2f} (need >= {MIN_RECALL}), FPR {fpr:.3f} (need <= {MAX_FPR}); "
               f"recall on held-out injection families {flags[hidden].mean():.2f}")
        assert recall >= MIN_RECALL and fpr <= MAX_FPR, msg
        print("    " + msg)

    return run_check("Exercise 3 · guardrail on the hidden set", test,
                     hint="combine the classifier score with rules for instruction-override phrases "
                          "(ignore/disregard/forget ... instructions/rules/prompt), role-play, fake system tags, "
                          "prompt-extraction requests and insults; tune the threshold for FPR <= 5%")
