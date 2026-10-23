"""Unit and integration tests for HOFM4 + JointBCD (no MovieLens downloads)."""
from itertools import combinations
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from src.hofm4_interaction import FourthOrderJointBCD, fourth_anova
from fast_unified_hofm4 import install_fourth_order


def test_degree_four_equals_bruteforce_combinatorial_formula():
    torch.manual_seed(42)
    x = torch.randn(4, 7, 5, requires_grad=True)
    exact = fourth_anova(x)
    brute = sum((x[:, a] * x[:, b] * x[:, c] * x[:, d])
                for a, b, c, d in combinations(range(7), 4)).sum(-1)
    assert torch.allclose(exact, brute, atol=1e-5, rtol=1e-5)
    exact.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_quartic_branch_all_parameters_train_and_eval_is_deterministic():
    torch.manual_seed(42)
    branch = FourthOrderJointBCD(16, 32, bcd_dim=15, rank=9,
                                input_dropout=0.3)
    u = torch.tensor([1, 2, 3])
    i = torch.tensor([8, 9, 10])
    bcd = torch.randn(3, 15)
    branch.train()
    value = branch(u, i, bcd)
    assert value.shape == (3,)
    value.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in branch.parameters())
    branch.eval()
    assert torch.equal(branch(u, i, bcd), branch(u, i, bcd))


def test_hofm4_wrapper_adds_trainable_params_and_preserves_hofm3_parent(monkeypatch):
    class TinyHOFM3(nn.Module):
        def __init__(self):
            super().__init__()
            self.user = nn.Embedding(12, 7)
            self.movie = nn.Embedding(20, 7)
            self.bcd_linear = nn.Linear(15, 1)

        def forward(self, batch, bcd):
            return (self.user(batch["user"]) *
                    self.movie(batch["item"])).sum(-1) + self.bcd_linear(bcd).squeeze(-1)

        def third_regularization(self, positive, negative, coefficient=0.0):
            return self.user(positive["user"]).square().mean() * coefficient

    original_optimizer_init = torch.optim.Optimizer.__init__
    try:
        ns = SimpleNamespace(UnifiedHOFM3BCD=TinyHOFM3)
        seen = install_fourth_order(ns, rank=8, strength=1.0,
                                    dropout=0.2, factor_l2=0.002)
        model = TinyHOFM3()
        assert isinstance(model.fourth_order, FourthOrderJointBCD)
        old_checkpoint = {
            k: v.clone() for k, v in model.state_dict().items()
            if not k.startswith("fourth_order.")
        }
        # HOFM3 parent checkpoints can be loaded strictly without silently
        # modifying any old tensor keys or values.
        model.load_state_dict(old_checkpoint, strict=True)
        for name, expected in old_checkpoint.items():
            assert torch.equal(model.state_dict()[name], expected)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        assert seen["count"] == 1
        ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert all(id(p) in ids for p in model.fourth_order.parameters())
        pos = {"user": torch.tensor([1, 2, 3]),
               "item": torch.tensor([4, 5, 6])}
        neg = {"item": torch.tensor([7, 8, 9])}
        bcd = torch.randn(3, 15)
        score = model(pos, bcd)
        loss = score.square().mean() + model.third_regularization(pos, neg, 0.01)
        loss.backward()
        assert all(p.grad is not None for p in model.fourth_order.parameters())
        optimizer.step()
        # No optimizer is allowed to train a HOFM3 subset but omit degree-4.
        with pytest.raises(RuntimeError, match="omitted HOFM4"):
            torch.optim.AdamW(list(model.user.parameters()) +
                              list(model.movie.parameters()), lr=0.01)
    finally:
        monkeypatch.setattr(torch.optim.Optimizer, "__init__", original_optimizer_init)


def test_fourth_order_input_validation():
    module = FourthOrderJointBCD(10, 20)
    u = torch.tensor([0, 1])
    i = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="JointBCD shape"):
        module(u, i, torch.randn(2, 14))
    with pytest.raises(FloatingPointError):
        bcd = torch.randn(2, 15)
        bcd[0, 1] = float("nan")
        module(u, i, bcd)



def test_warmstart_preserves_new_quartic_parameters_with_upstream_strict_check(monkeypatch):
    """Regression: original trainer rejects missing fourth_order.* keys."""
    class StrictParent(nn.Module):
        def __init__(self):
            super().__init__()
            self.user = nn.Embedding(12, 7)
            self.movie = nn.Embedding(20, 7)
            self.bcd_linear = nn.Linear(15, 1)

        def forward(self, batch, bcd):
            return (self.user(batch["user"]) * self.movie(batch["item"])).sum(-1)

        def third_regularization(self, positive, negative, coefficient=0.0):
            return self.user(positive["user"]).square().mean() * coefficient

        def load_hofm3_state(self, source):
            state = source.state_dict() if isinstance(source, nn.Module) else source
            missing, unexpected = self.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(
                    f"HOFM3 warm-start mismatch: missing={missing}, "
                    f"unexpected={unexpected}")

    # Create an old HOFM3 instance before installing the HOFM4 class patch.
    old = StrictParent()
    with torch.no_grad():
        old.user.weight.fill_(0.23)
        old.movie.weight.fill_(0.31)
    original_optimizer_init = torch.optim.Optimizer.__init__
    try:
        ns = SimpleNamespace(UnifiedHOFM3BCD=StrictParent)
        install_fourth_order(ns, rank=8, strength=1.0,
                             dropout=0.2, factor_l2=0.002)
        target = StrictParent()
        quartic_before = {
            k: v.detach().clone()
            for k, v in target.fourth_order.state_dict().items()
        }
        # The exact error in the user's screenshot occurs without this fix.
        target.load_hofm3_state(old)
        assert torch.equal(target.user.weight, old.user.weight)
        assert torch.equal(target.movie.weight, old.movie.weight)
        assert all(
            torch.equal(target.fourth_order.state_dict()[k], v)
            for k, v in quartic_before.items()
        )
        assert "fourth_order" in target._modules
        # A second HOFM4 model is also a valid HOFM3 warm-start source:
        # it must NOT import quartic parameters as pretrained HOFM3 data.
        second = StrictParent()
        second_quartic = {
            k: v.detach().clone()
            for k, v in second.fourth_order.state_dict().items()
        }
        second.load_hofm3_state(target)
        assert all(
            torch.equal(second.fourth_order.state_dict()[k], v)
            for k, v in second_quartic.items()
        )
        assert "fourth_order" in target._modules
        assert "fourth_order" in second._modules

        # On genuinely incompatible parent tensors, still fail closed,
        # while restoring the fourth-order module after the failure.
        bad = {"user.weight": torch.ones_like(old.user.weight)}
        with pytest.raises(ValueError, match="warm-start mismatch"):
            target.load_hofm3_state(bad)
        assert "fourth_order" in target._modules
        assert all(
            torch.equal(target.fourth_order.state_dict()[k], v)
            for k, v in quartic_before.items()
        )
    finally:
        monkeypatch.setattr(torch.optim.Optimizer, "__init__", original_optimizer_init)
