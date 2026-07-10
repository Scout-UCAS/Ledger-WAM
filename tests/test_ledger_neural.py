"""Unit tests for the dependency-free neural causal ledger."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


# Loading by path prevents the legacy wan_va/__init__.py from importing the
# optional runtime stack (diffusers/flash-attn) in this focused CPU unit test.
MODULE_PATH = Path(__file__).parents[1] / "wan_va" / "ledger" / "neural.py"
SPEC = importlib.util.spec_from_file_location("ledger_neural", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ledger_neural = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger_neural)

NeuralLedgerHead = ledger_neural.NeuralLedgerHead
NeuralLedgerLoss = ledger_neural.NeuralLedgerLoss
LEDGER_LOSS_NAMES = ledger_neural.LEDGER_LOSS_NAMES
build_repair_debt_reward_targets = ledger_neural.build_repair_debt_reward_targets


class NeuralLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.batch_size = 2
        self.num_slots = 4
        self.num_relations = 5
        self.num_rollback = 6
        self.num_repairs = 3
        self.num_claim_types = 3
        self.num_subjects = 7
        self.num_objects = 8
        self.num_preconditions = 4
        self.num_effects = 6
        self.delta_dim = 7
        self.model = NeuralLedgerHead(
            latent_channels=4,
            action_dim=5,
            text_dim=6,
            hidden_dim=16,
            num_claim_slots=self.num_slots,
            num_relations=self.num_relations,
            num_rollback_steps=self.num_rollback,
            num_repair_actions=self.num_repairs,
            delta_dim=self.delta_dim,
            num_heads=4,
            dropout=0.0,
            num_claim_types=self.num_claim_types,
            num_subjects=self.num_subjects,
            num_objects=self.num_objects,
            num_preconditions=self.num_preconditions,
            num_effects=self.num_effects,
        )
        self.latents = torch.randn(self.batch_size, 4, 3, 2, 2)
        self.actions = torch.randn(self.batch_size, 5, 3, 2, 1)
        self.text_emb = torch.randn(self.batch_size, 4, 6)

    def _outputs(self, previous_claim_slots=None):
        return self.model(
            self.latents,
            self.actions,
            self.text_emb,
            previous_claim_slots=previous_claim_slots,
        )

    def test_output_shapes_and_bounds(self) -> None:
        output = self._outputs()
        scalar_slot_names = (
            "presence_logits",
            "presence",
            "claim_presence_logits",
            "claim_presence",
            "claim_logits",
            "confidence",
            "evidence_logits",
            "evidence",
            "uncertainty_logits",
            "uncertainty",
            "dependency_logits",
            "dependency",
            "repair_cost_logits",
            "repair_cost",
            "observability_logits",
            "observability",
            "importance_logits",
            "importance",
            "debt_logits",
            "debt",
        )
        for name in scalar_slot_names:
            self.assertEqual(output[name].shape, (self.batch_size, self.num_slots))
        for name in ("claim_mask", "active_claim_mask"):
            self.assertEqual(output[name].shape, (self.batch_size, self.num_slots))
            self.assertEqual(output[name].dtype, torch.bool)
        self.assertEqual(
            output["claim_aggregation_weight"].shape,
            (self.batch_size, self.num_slots),
        )

        self.assertEqual(
            output["claim_slots"].shape,
            (self.batch_size, self.num_slots, 16),
        )
        self.assertEqual(
            output["relation_logits"].shape,
            (self.batch_size, self.num_slots, self.num_relations),
        )
        categorical_shapes = {
            "claim_type_logits": self.num_claim_types,
            "subject_logits": self.num_subjects,
            "object_logits": self.num_objects,
            "precondition_logits": self.num_preconditions,
            "effect_logits": self.num_effects,
        }
        for name, cardinality in categorical_shapes.items():
            self.assertEqual(
                output[name].shape,
                (self.batch_size, self.num_slots, cardinality),
            )
        for name in (
            "dependency_matrix_logits",
            "dependency_matrix",
            "dependency_matrix_probability",
            "dependency_matrix_probabilities",
        ):
            self.assertEqual(
                output[name].shape,
                (self.batch_size, self.num_slots, self.num_slots),
            )
        self.assertEqual(
            output["rollback_logits"].shape,
            (self.batch_size, self.num_slots, self.num_rollback),
        )
        self.assertEqual(
            output["factual_delta"].shape,
            (self.batch_size, self.num_slots, self.delta_dim),
        )
        self.assertEqual(
            output["counterfactual_delta"].shape,
            (self.batch_size, self.num_slots, self.delta_dim),
        )
        self.assertEqual(output["ledger_context"].shape, (self.batch_size, 16))
        self.assertEqual(output["global_risk"].shape, (self.batch_size,))
        self.assertEqual(
            output["repair_post_debt_per_claim"].shape,
            (self.batch_size, self.num_repairs, self.num_slots),
        )
        for name in (
            "repair_post_debt",
            "repair_scores",
            "repair_logits",
            "repair_debt_reduction",
        ):
            self.assertEqual(output[name].shape, (self.batch_size, self.num_repairs))
        self.assertEqual(
            output["repair_world_delta"].shape,
            (self.batch_size, self.num_repairs, self.delta_dim),
        )
        self.assertEqual(
            output["repair_debt_change_per_claim"].shape,
            (self.batch_size, self.num_repairs, self.num_slots),
        )

        self.assertTrue(bool(torch.all(output["debt_weights"] > 0)))
        self.assertTrue(bool(torch.all(output["debt"] >= 0)))
        self.assertTrue(bool(torch.all(output["debt"] <= 1)))
        self.assertTrue(bool(torch.all(output["global_risk"] >= 0)))
        for name in (
            "presence",
            "confidence",
            "evidence",
            "uncertainty",
            "dependency",
            "dependency_matrix",
            "repair_cost",
            "observability",
            "importance",
        ):
            self.assertTrue(bool(torch.all(output[name] >= 0)))
            self.assertTrue(bool(torch.all(output[name] <= 1)))
        self.assertTrue(bool(torch.all(output["repair_post_debt"] >= 0)))
        self.assertTrue(bool(torch.all(output["repair_post_debt"] <= 1)))
        self.assertTrue(bool(torch.all(output["repair_reduction"] >= -1)))
        self.assertTrue(bool(torch.all(output["repair_reduction"] <= 1)))

    def test_all_losses_have_finite_gradients(self) -> None:
        output = self._outputs()
        targets = {
            "presence": torch.randint(0, 2, (self.batch_size, self.num_slots)).float(),
            "claim": torch.randint(0, 2, (self.batch_size, self.num_slots)).float(),
            "claim_type": torch.randint(
                0, self.num_claim_types, (self.batch_size, self.num_slots)
            ),
            "subject": torch.randint(
                0, self.num_subjects, (self.batch_size, self.num_slots)
            ),
            "object": torch.randint(
                0, self.num_objects, (self.batch_size, self.num_slots)
            ),
            "precondition": torch.randint(
                0, self.num_preconditions, (self.batch_size, self.num_slots)
            ),
            "effect": torch.randint(
                0, self.num_effects, (self.batch_size, self.num_slots)
            ),
            "evidence": torch.rand(self.batch_size, self.num_slots),
            "uncertainty": torch.rand(self.batch_size, self.num_slots),
            "dependency": torch.rand(self.batch_size, self.num_slots),
            "dependency_matrix": torch.rand(
                self.batch_size, self.num_slots, self.num_slots
            ),
            "debt": torch.rand(self.batch_size, self.num_slots),
            # Cost is a non-negative regression target rather than a binary
            # label, and therefore may be larger than one.
            "repair_cost": 2.0 * torch.rand(self.batch_size, self.num_slots),
            "observability": torch.rand(self.batch_size, self.num_slots),
            "importance": torch.rand(self.batch_size, self.num_slots),
            "relation": torch.randint(
                0, self.num_relations, (self.batch_size, self.num_slots)
            ),
            "rollback": torch.randint(
                0, self.num_rollback, (self.batch_size, self.num_slots)
            ),
            "cf": torch.randint(0, 2, (self.batch_size, self.num_slots)).float(),
            "repair": torch.tensor([0, 2]),
            "action_cost": torch.rand(self.batch_size, self.num_repairs),
            "repair_world": torch.randn(
                self.batch_size, self.num_repairs, self.delta_dim
            ),
            "debt_reward": torch.rand(self.batch_size, self.num_repairs),
        }
        losses = NeuralLedgerLoss()(output, targets)
        self.assertEqual(set(losses), set(LEDGER_LOSS_NAMES) | {"total"})
        for loss in losses.values():
            self.assertTrue(bool(torch.isfinite(loss)))

        losses["total"].backward()
        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))
        self.assertGreater(sum(float(grad.abs().sum()) for grad in gradients), 0.0)

    def test_out_of_range_class_labels_are_safely_ignored(self) -> None:
        output = self._outputs()
        relation = torch.full(
            (self.batch_size, self.num_slots),
            self.num_relations + 100,
        )
        losses = NeuralLedgerLoss()(output, {"relation": relation})
        self.assertTrue(bool(torch.isfinite(losses["relation"])))
        self.assertEqual(float(losses["relation"].detach()), 0.0)

    def test_pairwise_dependency_uses_directed_masked_bce(self) -> None:
        output = self._outputs()
        labels = torch.full(
            (self.batch_size, self.num_slots, self.num_slots),
            -100.0,
        )
        mask = torch.zeros_like(labels, dtype=torch.bool)
        labels[0, 0, 1] = 1.0
        labels[1, 2, 0] = 0.25
        mask[0, 0, 1] = True
        mask[1, 2, 0] = True

        loss = NeuralLedgerLoss()(
            output,
            {
                "ledger_dependency_matrix": labels,
                "ledger_dependency_matrix_mask": mask,
            },
        )["dependency_matrix"]
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            output["dependency_matrix_logits"][mask], labels[mask]
        )
        self.assertTrue(bool(torch.allclose(loss, expected)))

        # Source/target projections are distinct, so the graph is directed
        # rather than constrained to be symmetric.
        logits = output["dependency_matrix_logits"]
        self.assertFalse(bool(torch.allclose(logits, logits.transpose(-1, -2))))

    def test_previous_claim_slots_preserve_identity_and_backpropagate(self) -> None:
        baseline = self._outputs()
        self.assertNotIn("slot_matching_weights", baseline)

        previous = torch.randn(
            self.batch_size,
            self.num_slots,
            16,
            requires_grad=True,
        )
        output = self._outputs(previous_claim_slots=previous)
        for name in (
            "claim_slot_candidates",
            "slot_update_gate",
            "slot_update_delta",
        ):
            self.assertEqual(
                output[name].shape,
                (self.batch_size, self.num_slots, 16),
            )
        self.assertEqual(
            output["slot_matching_weights"].shape,
            (self.batch_size, self.num_slots, self.num_slots),
        )
        self.assertTrue(
            bool(
                torch.allclose(output["claim_slot_candidates"], baseline["claim_slots"])
            )
        )
        self.assertFalse(
            bool(torch.allclose(output["claim_slots"], baseline["claim_slots"]))
        )
        self.assertTrue(bool(torch.all(output["slot_update_gate"] >= 0)))
        self.assertTrue(bool(torch.all(output["slot_update_gate"] <= 1)))
        self.assertTrue(
            bool(
                torch.allclose(
                    output["slot_matching_weights"].sum(dim=-1),
                    torch.ones(self.batch_size, self.num_slots),
                )
            )
        )

        # Reordering previous slots reorders the updated outputs in exactly the
        # same way: previous slots are queries and therefore own the identity.
        permutation = torch.tensor([2, 0, 3, 1])
        permuted_output = self._outputs(
            previous_claim_slots=previous.detach()[:, permutation]
        )
        self.assertTrue(
            bool(
                torch.allclose(
                    permuted_output["claim_slots"],
                    output["claim_slots"].detach()[:, permutation],
                    atol=1e-6,
                    rtol=1e-5,
                )
            )
        )

        output["confidence"].sum().backward()
        self.assertIsNotNone(previous.grad)
        self.assertTrue(bool(torch.isfinite(previous.grad).all()))
        self.assertGreater(float(previous.grad.abs().sum()), 0.0)
        updater_grad = self.model.slot_update_attention.in_proj_weight.grad
        self.assertIsNotNone(updater_grad)
        self.assertTrue(bool(torch.isfinite(updater_grad).all()))
        self.assertGreater(float(updater_grad.abs().sum()), 0.0)

    def test_previous_claim_slots_shape_is_validated(self) -> None:
        invalid_shapes = (
            (self.batch_size, self.num_slots, 15),
            (self.batch_size, self.num_slots - 1, 16),
            (self.batch_size + 1, self.num_slots, 16),
            (self.batch_size, self.num_slots * 16),
        )
        for shape in invalid_shapes:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    self._outputs(previous_claim_slots=torch.randn(*shape))

    def test_previous_matching_excludes_padded_current_slots(self) -> None:
        previous = torch.randn(self.batch_size, self.num_slots, 16)
        current_mask = torch.tensor(
            [[True, True, False, False], [False, False, False, False]]
        )
        output = self.model(
            self.latents,
            self.actions,
            self.text_emb,
            masks={"claim": current_mask},
            previous_claim_slots=previous,
            previous_claim_mask=torch.ones_like(current_mask),
        )
        self.assertTrue(bool(torch.isfinite(output["claim_slots"]).all()))
        # Real candidates masked by the current record receive no matching
        # probability.  The all-empty row uses only the temporary safe key.
        self.assertTrue(bool(torch.all(output["slot_matching_weights"][0, :, 2:] == 0)))
        self.assertTrue(bool(torch.all(output["slot_matching_weights"][1, :, 1:] == 0)))

        with self.assertRaises(ValueError):
            self.model(
                self.latents,
                self.actions,
                self.text_emb,
                previous_claim_slots=previous,
                previous_claim_mask=torch.ones(
                    self.batch_size, self.num_slots - 1, dtype=torch.bool
                ),
            )

    def test_repair_transition_represents_harmful_repairs(self) -> None:
        with torch.no_grad():
            self.model.repair_reduction_head.weight.zero_()
            self.model.repair_world_head.weight.zero_()
            self.model.repair_world_head.bias.zero_()
            self.model.repair_reduction_head.bias.fill_(-4.0)
        harmful = self._outputs()
        self.assertTrue(
            bool(
                torch.all(
                    harmful["repair_post_debt_per_claim"] > harmful["debt"].unsqueeze(1)
                )
            )
        )
        self.assertTrue(bool(torch.all(harmful["repair_debt_reduction"] < 0)))

    def test_identical_actions_have_identical_shared_transition(self) -> None:
        output = self._outputs()
        self.assertTrue(
            bool(
                torch.allclose(
                    output["factual_delta"],
                    output["counterfactual_delta"],
                    atol=0.0,
                    rtol=0.0,
                )
            )
        )
        positive = NeuralLedgerLoss(cf_margin=1.0)(
            output,
            {"cf_global": torch.ones(self.batch_size)},
        )["cf"]
        negative = NeuralLedgerLoss(cf_margin=1.0)(
            output,
            {"cf_global": torch.zeros(self.batch_size)},
        )["cf"]
        self.assertTrue(bool(torch.allclose(positive, torch.tensor(1.0))))
        self.assertTrue(bool(torch.allclose(negative, torch.tensor(0.0))))

        padded_only = dict(output)
        padded_only["counterfactual_delta"] = output["factual_delta"].clone()
        padded_only["counterfactual_delta"][:, -1, 0] += 2.0
        masked = NeuralLedgerLoss(cf_margin=1.0)(
            padded_only,
            {"cf_global": torch.ones(self.batch_size)},
            masks={
                "cf_global_slot": torch.tensor(
                    [[True, False, False, False]] * self.batch_size
                )
            },
        )["cf"]
        self.assertTrue(bool(torch.allclose(masked, torch.tensor(1.0))))

    def test_invalid_selected_repair_labels_are_masked_before_indexing(self) -> None:
        output = self._outputs()
        repair_labels = torch.tensor([-100, 1])
        valid_samples = torch.tensor([False, True])
        losses = NeuralLedgerLoss()(
            output,
            {
                "repair": repair_labels,
                "repair_world": torch.randn(self.batch_size, self.delta_dim),
                "debt_reward": torch.randn(self.batch_size),
            },
            masks={
                "repair": valid_samples,
                "repair_world": valid_samples,
                "debt_reward": valid_samples,
            },
        )
        self.assertTrue(bool(torch.isfinite(losses["repair_world"])))
        self.assertTrue(bool(torch.isfinite(losses["debt_reward"])))

    def test_repair_reward_target_matches_signed_global_risk_change(self) -> None:
        current = torch.tensor([[0.8, 0.4, 0.2]])
        post = torch.tensor([[0.2, 0.8, -100.0]])
        valid = torch.tensor([[True, True, False]])
        repair_labels = torch.tensor([[0, 1, -100]])
        targets, mask = build_repair_debt_reward_targets(
            current_debt=current,
            current_debt_mask=torch.ones_like(valid),
            post_repair_debt=post,
            post_repair_mask=valid,
            repair_labels=repair_labels,
            repair_label_mask=valid,
            num_repair_actions=3,
            importance=torch.tensor([[2.0, 1.0, 1.0]]),
            importance_mask=torch.ones_like(valid),
        )
        self.assertEqual(mask.tolist(), [[True, True, False]])
        self.assertTrue(bool(torch.allclose(targets[0, 0], torch.tensor(0.3))))
        self.assertTrue(bool(torch.allclose(targets[0, 1], torch.tensor(-0.1))))
        self.assertEqual(float(targets[0, 2]), -100.0)

    def test_repair_reward_target_can_include_cost_and_task_risk(self) -> None:
        current = torch.tensor([[0.8, 0.4]])
        post = torch.tensor([[0.2, 0.1]])
        valid = torch.tensor([[True, True]])
        repair_labels = torch.tensor([[0, 1]])
        action_cost = torch.tensor([[0.2, 0.4]])
        task_risk = torch.tensor([[0.1, 0.5]])

        targets, mask = build_repair_debt_reward_targets(
            current_debt=current,
            current_debt_mask=valid,
            post_repair_debt=post,
            post_repair_mask=valid,
            repair_labels=repair_labels,
            repair_label_mask=valid,
            num_repair_actions=2,
            action_cost=action_cost,
            task_risk=task_risk,
            cost_weight=0.5,
            risk_weight=0.25,
        )

        self.assertEqual(mask.tolist(), [[True, True]])
        self.assertTrue(
            bool(torch.allclose(targets[0, 0], torch.tensor(0.3 - 0.1 - 0.025)))
        )
        self.assertTrue(
            bool(torch.allclose(targets[0, 1], torch.tensor(0.15 - 0.2 - 0.125)))
        )

    def test_debt_reward_loss_uses_paper_cost_and_risk_terms(self) -> None:
        output = self._outputs()
        task_risk = torch.rand(self.batch_size, self.num_repairs)
        target = (
            output["repair_debt_reduction"].detach()
            - 0.5 * output["repair_action_cost"].detach()
            - 0.25 * task_risk
        )

        cost_aware = NeuralLedgerLoss(
            repair_action_cost_weight=0.5,
            repair_task_risk_weight=0.25,
        )(output, {"debt_reward": target, "repair_task_risk": task_risk})
        raw = NeuralLedgerLoss()(
            output, {"debt_reward": target, "repair_task_risk": task_risk}
        )

        self.assertLess(float(cost_aware["debt_reward"].detach()), 1e-7)
        self.assertGreater(float(raw["debt_reward"].detach()), 0.0)

    def test_post_debt_target_wins_when_delta_and_slot_width_match(self) -> None:
        model = NeuralLedgerHead(
            latent_channels=4,
            action_dim=5,
            text_dim=6,
            hidden_dim=16,
            num_claim_slots=self.num_slots,
            num_relations=self.num_relations,
            num_rollback_steps=self.num_rollback,
            num_repair_actions=self.num_repairs,
            delta_dim=self.num_slots,
            num_heads=4,
        )
        output = model(self.latents, self.actions, self.text_emb)
        target = torch.rand(self.batch_size, self.num_repairs, self.num_slots)
        loss = NeuralLedgerLoss()(output, {"repair_post_debt": target})["repair_world"]
        expected = torch.nn.functional.smooth_l1_loss(
            output["repair_post_debt_per_claim"], target
        )
        wrong = torch.nn.functional.smooth_l1_loss(output["repair_world_delta"], target)
        self.assertTrue(bool(torch.allclose(loss, expected)))
        self.assertFalse(bool(torch.allclose(loss, wrong)))

    def test_recurrent_consistency_is_part_of_total_loss(self) -> None:
        output = self._outputs()
        recurrent = torch.tensor(0.75, requires_grad=True)
        output["recurrent_consistency_loss"] = recurrent
        losses = NeuralLedgerLoss()(output, {})
        self.assertTrue(bool(torch.allclose(losses["recurrent"], recurrent)))
        self.assertTrue(bool(torch.allclose(losses["total"], recurrent)))

    def test_presence_softly_weights_unmasked_ledger_aggregations(self) -> None:
        output = self._outputs()
        presence = output["presence"]
        self.assertTrue(
            bool(torch.allclose(output["claim_aggregation_weight"], presence))
        )
        self.assertTrue(bool(torch.all(output["claim_mask"])))
        self.assertTrue(
            bool(
                torch.equal(
                    output["active_claim_mask"],
                    presence.ge(0.5),
                )
            )
        )

        risk_weight = output["importance"] * presence
        expected_risk = (output["debt"] * risk_weight).sum(dim=-1) / (
            risk_weight.sum(dim=-1).clamp_min(1e-6)
        )
        self.assertTrue(bool(torch.allclose(output["global_risk"], expected_risk)))

        context_weight = output["confidence"] * risk_weight
        expected_context = (output["claim_slots"] * context_weight.unsqueeze(-1)).sum(
            dim=1
        ) / context_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.assertTrue(
            bool(torch.allclose(output["ledger_context"], expected_context))
        )

        expected_post_debt = (
            output["repair_post_debt_per_claim"] * risk_weight.unsqueeze(1)
        ).sum(dim=-1) / risk_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.assertTrue(
            bool(torch.allclose(output["repair_post_debt"], expected_post_debt))
        )

    def test_explicit_claim_mask_overrides_presence_aggregation_weight(self) -> None:
        claim_mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=torch.bool)
        output = self.model(
            self.latents,
            self.actions,
            self.text_emb,
            masks={"claim": claim_mask},
        )
        self.assertTrue(bool(torch.equal(output["claim_mask"], claim_mask)))
        self.assertTrue(
            bool(
                torch.equal(
                    output["active_claim_mask"],
                    output["presence"].ge(0.5) & claim_mask,
                )
            )
        )
        self.assertTrue(
            bool(torch.equal(output["claim_aggregation_weight"], claim_mask.float()))
        )

        risk_weight = output["importance"] * claim_mask.to(output["importance"])
        expected_post_debt = (
            output["repair_post_debt_per_claim"] * risk_weight.unsqueeze(1)
        ).sum(dim=-1) / risk_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.assertTrue(
            bool(torch.allclose(output["repair_post_debt"], expected_post_debt))
        )

    def test_debt_is_monotone_by_construction(self) -> None:
        shape = (2, self.num_slots)
        base = torch.full(shape, 0.5)

        def debt_and_risk(**updates):
            values = {
                "confidence": base,
                "uncertainty": base,
                "dependency": base,
                "repair_cost": base,
                "observability": base,
                "importance": base,
            }
            values.update(updates)
            result = self.model.compute_debt(**values)
            return result["debt"], result["global_risk"]

        debt_base, risk_base = debt_and_risk()
        higher = torch.full(shape, 0.8)

        for increasing_component in (
            "uncertainty",
            "dependency",
            "repair_cost",
        ):
            debt_new, risk_new = debt_and_risk(**{increasing_component: higher})
            self.assertTrue(bool(torch.all(debt_new >= debt_base)))
            self.assertTrue(bool(torch.all(risk_new >= risk_base)))

        for decreasing_component in ("confidence", "observability"):
            debt_new, _ = debt_and_risk(**{decreasing_component: higher})
            self.assertTrue(bool(torch.all(debt_new <= debt_base)))

    def test_masks_and_missing_labels_are_safe(self) -> None:
        output = self._outputs()
        loss_module = NeuralLedgerLoss()

        missing = loss_module(output, {})
        for name in LEDGER_LOSS_NAMES:
            self.assertEqual(float(missing[name].detach()), 0.0)
        self.assertEqual(float(missing["total"].detach()), 0.0)

        mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        labels_a = torch.zeros(self.batch_size, self.num_slots)
        labels_b = labels_a.clone()
        labels_b[~mask] = 1.0
        loss_a = loss_module(output, {"claim": labels_a, "claim_mask": mask})["claim"]
        loss_b = loss_module(output, {"claim": labels_b}, masks={"claim": mask})[
            "claim"
        ]
        self.assertTrue(bool(torch.allclose(loss_a, loss_b)))

        all_masked = loss_module(
            output,
            {
                "debt": torch.full((self.batch_size, self.num_slots), -100.0),
                "dependency": torch.full((self.batch_size, self.num_slots), -100.0),
            },
            masks={
                "debt": torch.zeros_like(mask),
                "dependency": torch.zeros_like(mask),
            },
        )
        self.assertEqual(float(all_masked["debt"].detach()), 0.0)
        self.assertEqual(float(all_masked["dependency"].detach()), 0.0)
        self.assertTrue(bool(torch.isfinite(all_masked["total"])))

        claim_mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=torch.bool)
        claim_masked_output = self.model(
            self.latents,
            self.actions,
            self.text_emb,
            masks={"claim": claim_mask},
        )
        expected_risk = (
            claim_masked_output["debt"]
            * claim_masked_output["importance"]
            * claim_mask.to(claim_masked_output["debt"])
        ).sum(dim=-1) / (
            claim_masked_output["importance"]
            * claim_mask.to(claim_masked_output["importance"])
        ).sum(
            dim=-1
        )
        self.assertTrue(
            bool(torch.allclose(claim_masked_output["global_risk"], expected_risk))
        )

        context_weights = (
            claim_masked_output["confidence"]
            * claim_masked_output["importance"]
            * claim_mask.to(claim_masked_output["confidence"])
        )
        expected_context = (
            claim_masked_output["claim_slots"] * context_weights.unsqueeze(-1)
        ).sum(dim=1) / context_weights.sum(dim=1, keepdim=True)
        self.assertTrue(
            bool(
                torch.allclose(claim_masked_output["ledger_context"], expected_context)
            )
        )

        # Even fully masked input modalities are safe because the head appends
        # a permanently valid null context token.
        masked_output = self.model(
            self.latents,
            self.actions,
            self.text_emb,
            masks={
                "latent": torch.zeros(self.batch_size, 3, 2, 2, dtype=torch.bool),
                "action": torch.zeros(self.batch_size, 3, 2, dtype=torch.bool),
                "text": torch.zeros(self.batch_size, 4, dtype=torch.bool),
            },
        )
        self.assertTrue(bool(torch.isfinite(masked_output["ledger_context"]).all()))


if __name__ == "__main__":
    unittest.main()
