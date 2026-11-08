"""Self-checks for Level 3 notebook 09 · Distributed data parallel."""

from __future__ import annotations

from churnlab.checks import close, run_check

REQUIRES = {"torch"}


def ex1(ring_allreduce_bytes):
    def test():
        cases = [  # (n_params, bytes_per, world_size, expected bytes sent per rank)
            (1_000_000, 4, 2, 4_000_000.0),
            (1_000_000, 4, 4, 6_000_000.0),
            (110_000_000, 2, 8, 2 * 7 / 8 * 220_000_000),
            (7_000_000_000, 2, 64, 2 * 63 / 64 * 14_000_000_000),
            (123, 4, 1, 0.0),
        ]
        for n, b, w, want in cases:
            got = ring_allreduce_bytes(n, b, w)
            assert close(got, want, 1e-9), (f"n_params={n:,}, bytes_per={b}, world_size={w}: expected "
                                            f"{want:,.0f} bytes, got {got:,.0f}")
        big = ring_allreduce_bytes(1_000_000, 4, 1024)
        assert big < 2 * 4_000_000, "per-rank traffic must stay below 2 x the gradient size, however many ranks"

    return run_check("Exercise 1 · ring_allreduce_bytes", test,
                     hint="reduce-scatter + all-gather: 2 (N-1) steps, each sending S/N bytes, S = n_params x bytes_per")


def ex2(average_gradients):
    def test():
        import torch

        g = torch.Generator().manual_seed(0)
        shapes = [(3, 4), (4,), (2, 2, 2)]
        for world in (1, 2, 3, 4):
            grads = [[torch.randn(s, generator=g) for s in shapes] for _ in range(world)]
            copies = [[t.clone() for t in r] for r in grads]
            out = average_gradients(grads)
            assert isinstance(out, (list, tuple)), f"return a list of tensors, got {type(out).__name__}"
            assert len(out) == len(shapes), f"expected {len(shapes)} averaged tensors, got {len(out)}"
            for i, s in enumerate(shapes):
                want = sum(r[i] for r in copies) / world
                assert tuple(out[i].shape) == s, f"parameter {i}: shape {tuple(out[i].shape)}, expected {s}"
                err = (out[i] - want).abs().max().item()
                assert err < 1e-6, f"world_size={world}, parameter {i}: off by {err:.2e} from the mean"
            for r, (a, b) in enumerate(zip(grads, copies)):
                for t, c in zip(a, b):
                    assert torch.equal(t, c), f"rank {r}'s input gradients were modified in place"
        # the point of averaging: equal-size shards -> exactly the full-batch gradient
        torch.manual_seed(1)
        model = torch.nn.Linear(5, 3)
        x, y = torch.randn(8, 5), torch.randint(0, 3, (8,))
        model.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        full = [p.grad.clone() for p in model.parameters()]
        per_rank = []
        for xs, ys in zip(x.chunk(2), y.chunk(2)):
            model.zero_grad()
            torch.nn.functional.cross_entropy(model(xs), ys).backward()
            per_rank.append([p.grad.clone() for p in model.parameters()])
        for a, b in zip(average_gradients(per_rank), full):
            assert (a - b).abs().max().item() < 1e-6, "averaging two half-batch gradients should give the full-batch one"

    return run_check("Exercise 2 · average_gradients", test,
                     hint="for parameter i: torch.stack([rank[i] for rank in grads_per_rank]).sum(0) / world_size")


def ex3(shard_indices):
    def test():
        from torch.utils.data import DistributedSampler

        for n, world, shuffle in [(10, 4, False), (10, 4, True), (16, 2, True), (7, 3, True), (3, 4, True),
                                  (1, 3, False), (100, 8, True)]:
            for epoch in (0, 1, 5):
                for rank in range(world):
                    s = DistributedSampler(range(n), num_replicas=world, rank=rank, shuffle=shuffle, seed=42)
                    s.set_epoch(epoch)
                    want = list(s)
                    got = shard_indices(n, world, rank, epoch, 42, shuffle)
                    assert list(got) == want, (f"n={n}, world_size={world}, rank={rank}, epoch={epoch}, "
                                               f"shuffle={shuffle}: expected {want}, got {list(got)}")

    return run_check("Exercise 3 · shard_indices", test,
                     hint="randperm with a Generator seeded seed+epoch; pad by repeating from the start to a "
                          "multiple of world_size; take indices[rank::world_size]")
