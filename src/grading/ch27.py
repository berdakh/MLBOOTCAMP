"""Self-checks for Chapter 27 · PyTorch training done right."""

from __future__ import annotations

import copy
import hashlib
import math

from churnlab.checks import close, run_check

REQUIRES = {"torch"}


def _tiny_problem(seed: int = 0):
    import torch

    g = torch.Generator().manual_seed(seed)
    X = torch.randn(96, 3, generator=g)
    y = X @ torch.tensor([1.5, -2.0, 0.5]) + 0.3 + 0.1 * torch.randn(96, generator=g)
    ds = torch.utils.data.TensorDataset(X, y.unsqueeze(1))
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(3, 1))
    return model, loader


def _reference_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total, count = 0.0, 0
    for *inputs, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(*inputs), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


def ex1(train_one_epoch):
    def test():
        import torch

        loss_fn = torch.nn.MSELoss()
        # 1) lr = 0: nothing changes, so the mean loss is the full-data loss.
        model, loader = _tiny_problem()
        X, y = loader.dataset.tensors
        with torch.no_grad():
            full = loss_fn(model(X), y).item()
        out = train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.0), loss_fn)
        assert isinstance(out, float), f"return a Python float (use loss.item()), got {type(out).__name__}"
        assert close(out, full, 1e-5), f"with lr=0 the mean loss should be {full:.6f}, got {out:.6f}"
        # 2) uneven last batch: weight batch means by batch size.
        ds = torch.utils.data.TensorDataset(X[:80], y[:80])
        uneven = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)  # 32, 32, 16
        with torch.no_grad():
            full80 = loss_fn(model(X[:80]), y[:80]).item()
        out = train_one_epoch(model, uneven, torch.optim.SGD(model.parameters(), lr=0.0), loss_fn)
        assert close(out, full80, 1e-5), (
            f"with batches of 32, 32, 16 the per-example mean is {full80:.6f}, got {out:.6f} "
            "(a plain mean of batch losses over-weights the last, smaller batch)")
        # 3) real steps: weights must match a reference loop exactly (step order, zero_grad).
        model, loader = _tiny_problem(1)
        ref = copy.deepcopy(model)
        model.eval()
        opt, ref_opt = (torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.5) for m in (model, ref))
        for _ in range(3):
            got = train_one_epoch(model, loader, opt, loss_fn)
            want = _reference_epoch(ref, loader, ref_opt, loss_fn)
            assert close(got, want, 1e-5), f"mean loss {got:.6f} differs from the reference loop's {want:.6f}"
        for (name, p), q in zip(model.named_parameters(), ref.parameters()):
            assert torch.allclose(p, q, atol=1e-6), (
                f"parameter {name} differs from the reference loop after 3 epochs: "
                "is zero_grad called once per batch, before backward?")
        assert model.training, "the model should be in training mode (model.train()) after the epoch"

    return run_check("Exercise 1 · train_one_epoch", test,
                     hint="model.train(); for *inputs, y in loader: zero_grad -> loss -> backward -> step; "
                          "total += loss.item() * len(y)")


def ex2(expected_initial_loss):
    def test():
        for p in (0.5, 0.137, 0.01, 0.9):
            want = -(p * math.log(p) + (1 - p) * math.log(1 - p))
            got = expected_initial_loss(p)
            assert close(got, want, 1e-6), f"pos_rate={p}: expected {want:.6f}, got {got}"
        assert close(expected_initial_loss(0.5), math.log(2), 1e-9), "pos_rate=0.5 should give log(2)"
        for p in (0.0, 1.0):
            got = expected_initial_loss(p)
            assert got == got and close(got, 0.0, 1e-9), f"pos_rate={p}: expected 0.0 (0*log 0 = 0), got {got}"

    return run_check("Exercise 2 · expected_initial_loss", test,
                     hint="binary entropy in nats: -(p ln p + (1-p) ln(1-p)), treating 0*ln 0 as 0")


# Hashes of sha256(f"nb09:{letter}")[:16] for the letters that ARE bugs, so reading
# this file does not hand you the answer at a glance.
_KEY = {"8bf8e65ce924e10c", "2686f4dd96906219", "8564ecfd610a4531"}


def _h(letter: str) -> str:
    return hashlib.sha256(f"nb09:{letter.strip().upper()}".encode()).hexdigest()[:16]


def ex3(find_bugs):
    def test():
        given = find_bugs()
        assert isinstance(given, (list, tuple, set)), f"return a list of letters, got {type(given).__name__}"
        letters = {str(x).strip().upper() for x in given}
        assert letters <= set("ABCDEFGH"), f"unknown option(s): {sorted(letters - set('ABCDEFGH'))}"
        found = sum(_h(x) in _KEY for x in letters)
        false_alarms = len(letters) - found
        assert found == len(_KEY) and false_alarms == 0, (
            f"found {found} of {len(_KEY)} bugs, {false_alarms} false alarm(s)")

    return run_check("Exercise 3 · find_bugs", test,
                     hint="read the loop line by line: where is model.train()? what does CrossEntropyLoss expect? "
                          "which days end up in valid?")
