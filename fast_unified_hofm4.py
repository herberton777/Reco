#!/usr/bin/env python3
"""HOFM4 + JointBCD launcher: exact trainable quartic branch over HOFM3.

Depends on the existing (research-patch) temporal_fm.unified_hofm3_bcd module.
Uses the same optimized Stage D temporal features and existing BCD regularizers.
New fourth-order parameters are attached before the trainer constructs its
optimizer. The parent HOFM3 model is preserved and used for initialization.

Examples:
    python -u fast_unified_hofm4.py --self-test
    python -u fast_unified_hofm4.py --dataset ml-32m --run-label hofm4_trial01 \\
        --init-from fm_all_hofm3semi25_bpr --epochs 80 --min-epochs 80 \\
        --force-retrain [existing trainer options]
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import weakref

import torch

from fast_unified_regularized import (
    _preflight_run_label,
    _read_option,
    install_optimizations_and_regularization,
)
from src.hofm4_interaction import FourthOrderJointBCD, fourth_anova


def install_fourth_order(trainer, *, rank, strength, dropout, factor_l2):
    """Extend the already-regularized HOFM3 model *in this process only*.

    This extends the forward score and BPR step's regularization, without
    changing any persisted HOFM3 source file or parent checkpoint.
    """
    cls = trainer.UnifiedHOFM3BCD
    if getattr(cls, "_hofm4_installed", False):
        raise RuntimeError("HOFM4 extension was already installed")
    original_init = cls.__init__
    original_forward = cls.forward
    original_regularization = cls.third_regularization
    live_models = weakref.WeakSet()
    optimizer_matched = {"count": 0}

    def init_hofm4(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not isinstance(self.user, torch.nn.Embedding) or not isinstance(
                self.movie, torch.nn.Embedding):
            raise RuntimeError("Expected HOFM3 user/movie embeddings")
        if not hasattr(self, "bcd_linear") or not isinstance(
                self.bcd_linear, torch.nn.Linear):
            raise RuntimeError("Expected JointBCD linear feature projection")
        fields = self.bcd_linear.in_features
        if fields != 15:
            raise RuntimeError(
                f"Expected 15 JointBCD temporal fields, got {fields}; refusing wrong model")
        self.fourth_order = FourthOrderJointBCD(
            self.user.num_embeddings, self.movie.num_embeddings,
            bcd_dim=fields, rank=rank, strength=strength,
            input_dropout=dropout,
        )

        def permit_parent_checkpoint(module, state_dict, prefix, *remaining):
            # A frozen HOFM3 parent checkpoint lacks the new quartic tensors.
            # Supply ONLY missing fourth-order tensors; no HOFM3 parameter
            # names are modified. A saved HOFM4 state keeps its real weights.
            for key, tensor in module.fourth_order.state_dict().items():
                target = prefix + "fourth_order." + key
                if target not in state_dict:
                    state_dict[target] = tensor.detach().clone()

        self.register_load_state_dict_pre_hook(permit_parent_checkpoint)
        live_models.add(self)

    def forward_hofm4(self, batch, bcd):
        original_score = original_forward(self, batch, bcd)
        if "user" not in batch or "item" not in batch:
            raise RuntimeError("HOFM4 requires user/item IDs in trainer batches")
        extra = self.fourth_order(batch["user"], batch["item"], bcd)
        if original_score.numel() != extra.numel():
            raise RuntimeError("Cannot align HOFM4 quartic and HOFM3 scores")
        return original_score + extra.reshape_as(original_score)

    def regularize_hofm4(self, positive, negative, coefficient=0.0):
        result = original_regularization(self, positive, negative, coefficient)
        if factor_l2:
            result = result + factor_l2 * self.fourth_order.sampled_l2(
                positive, negative)
        return result

    # Fail closed if the actual optimizer includes the HOFM3 embeddings but
    # excludes the new fourth-order parameters. No silent untrained "HOFM4".
    opt_class = torch.optim.Optimizer
    original_optimizer_init = opt_class.__init__

    def checked_optimizer_init(opt, *args, **kwargs):
        original_optimizer_init(opt, *args, **kwargs)
        present = {id(p) for g in opt.param_groups for p in g["params"]}
        for model in list(live_models):
            base = {id(param) for name, param in model.named_parameters()
                    if not name.startswith("fourth_order.")}
            if not (base & present):
                continue
            optimizer_matched["count"] += 1
            fourth = {id(p) for p in model.fourth_order.parameters()}
            if not fourth.issubset(present):
                raise RuntimeError(
                    "Trainer optimizer omitted HOFM4 parameters. "
                    "Ensure the research trainer optimizes model.parameters().")

    cls.__init__ = init_hofm4
    cls.forward = forward_hofm4
    cls.third_regularization = regularize_hofm4
    cls._hofm4_installed = True
    opt_class.__init__ = checked_optimizer_init
    return optimizer_matched


def _self_test():
    # Correctness: dynamic program equals brute-force enumeration of 4 fields.
    fields = torch.randn(3, 6, 7, requires_grad=True)
    exact = fourth_anova(fields)
    brute = sum(fields[:, a] * fields[:, b] * fields[:, c] * fields[:, d]
                for a, b, c, d in combinations(range(fields.size(1)), 4)).sum(-1)
    assert torch.allclose(exact, brute, atol=1e-5, rtol=1e-5)
    model = FourthOrderJointBCD(12, 20, rank=8, input_dropout=0.25)
    model.train()
    batch = model(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]),
                  torch.randn(3, 15))
    loss = batch.square().sum() + 0.001 * model.sampled_l2(
        {"user": torch.tensor([1, 2, 3]), "item": torch.tensor([4, 5, 6])},
        {"item": torch.tensor([6, 7, 8])})
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    model.eval()
    x = torch.randn(3, 15)
    assert torch.allclose(model(torch.tensor([1, 2, 3]),
                                torch.tensor([4, 5, 6]), x),
                          model(torch.tensor([1, 2, 3]),
                                torch.tensor([4, 5, 6]), x))
    print("HOFM4 SELF-TEST PASS: exact quartic, trainable factors, evaluation dropout off")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False,
        description="Trainable HOFM4 interaction + existing HOFM3/Semi25/JointBCD")
    parser.add_argument("--fourth-rank", type=int, default=24)
    parser.add_argument("--fourth-strength", type=float, default=1.0)
    parser.add_argument("--fourth-dropout", type=float, default=0.20)
    parser.add_argument("--fourth-factor-l2", type=float, default=0.001)
    parser.add_argument("--bcd-dropout", type=float, default=0.30)
    parser.add_argument("--factor-l2", type=float, default=0.010)
    parser.add_argument("--content-l2", type=float, default=0.002)
    parser.add_argument("--bcd-l1", type=float, default=0.004)
    parser.add_argument("--bcd-l2", type=float, default=0.008)
    parser.add_argument("--disable-fast-features", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--help", action="store_true")
    extra, passthrough = parser.parse_known_args(argv)
    if extra.help:
        parser.print_help()
        print("All other options are passed to temporal_fm.unified_hofm3_bcd.")
        return 0
    if extra.self_test:
        _self_test()
        return 0
    values = ("fourth-strength", "fourth-dropout", "fourth-factor-l2",
              "bcd-dropout", "factor-l2", "content-l2", "bcd-l1", "bcd-l2")
    if extra.fourth_rank < 1:
        raise SystemExit("--fourth-rank must be >= 1")
    for key in values:
        v = getattr(extra, key.replace("-", "_"))
        if not math.isfinite(v) or v < 0:
            raise SystemExit(f"--{key} must be finite and nonnegative")
    if extra.fourth_strength <= 0:
        raise SystemExit("--fourth-strength must be positive")
    if not 0 <= extra.fourth_dropout < 1 or not 0 <= extra.bcd_dropout < 1:
        raise SystemExit("Dropout probabilities must be in [0,1)")

    config = {
        "model_family": "hofm4_jointbcd",
        "fourth_rank": extra.fourth_rank,
        "fourth_strength": extra.fourth_strength,
        "fourth_dropout": extra.fourth_dropout,
        "fourth_factor_l2": extra.fourth_factor_l2,
        "bcd_dropout": extra.bcd_dropout,
        "factor_l2": extra.factor_l2,
        "content_l2": extra.content_l2,
        "bcd_l1": extra.bcd_l1,
        "bcd_l2": extra.bcd_l2,
        "fast_features": not extra.disable_fast_features,
        "trainer_weight_decay": float(_read_option(
            passthrough, "--weight-decay", "0.15")),
        "trainer_third_l2": float(_read_option(
            passthrough, "--third-l2", "0.002")),
        "hard_negative_share": float(_read_option(
            passthrough, "--hard-negative-share", "0.25")),
        "launcher_version": 1,
    }
    if any(not math.isfinite(config[k]) or config[k] < 0 for k in
           ("trainer_weight_decay", "trainer_third_l2", "hard_negative_share")):
        raise SystemExit("Invalid base optimizer penalty")
    if config["hard_negative_share"] > 1:
        raise SystemExit("--hard-negative-share must be <= 1")

    # Reuse the HOFM3 trainer's existing output directory conventions but
    # require a unique HOFM4 label. Even --force-retrain may not overwrite
    # old experiment directories in this new launcher.
    dataset = _read_option(passthrough, "--dataset", "ml-32m")
    parent = _read_option(passthrough, "--init-from", "fm_all_hofm3semi25_bpr")
    label = _read_option(passthrough, "--run-label", "")
    run_dir = Path("models") / dataset / "unified_hofm3_bcd" / f"{parent}_{label}"
    if label and run_dir.exists():
        raise SystemExit("Run directory already exists: " + str(run_dir) +
                         ". Choose a NEW --run-label for HOFM4.")
    if "--resume" in passthrough:
        raise SystemExit("HOFM4 resume is not validated; use a new label.")
    run_dir = _preflight_run_label(passthrough, config)
    trainer = install_optimizations_and_regularization(
        bcd_dropout=extra.bcd_dropout,
        factor_l2=extra.factor_l2,
        content_l2=extra.content_l2,
        bcd_l1=extra.bcd_l1,
        bcd_l2=extra.bcd_l2,
        fast_features=not extra.disable_fast_features,
    )
    matched = install_fourth_order(
        trainer, rank=extra.fourth_rank,
        strength=extra.fourth_strength,
        dropout=extra.fourth_dropout,
        factor_l2=extra.fourth_factor_l2,
    )
    print("TRAINABLE HOFM4 + JointBCD (pretrained HOFM3 parent)", flush=True)
    print("Quartic rank:", extra.fourth_rank, "strength:", extra.fourth_strength, flush=True)
    print("Checkpoint:", run_dir, flush=True)
    print("Penalties and optimizer config:", json.dumps(config, sort_keys=True), flush=True)
    result = trainer.main(passthrough)
    if matched["count"] == 0:
        raise RuntimeError(
            "No HOFM4 optimizer initialization was verified. "
            "Do not interpret this run as trained HOFM4.")
    return result


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
