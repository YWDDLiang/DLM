import unittest

from crystal_dlm.h1a2_planner_objective import (
    FIELD_GROUP_IDS,
    FIELD_GROUP_WEIGHTS,
    LOOKAHEAD_HEAD_SPECS,
    build_lookahead_heads,
    build_lookahead_vocabs,
    encode_field_group_ids,
    encode_lookahead_labels,
    field_balanced_mean,
    lookahead_head_name,
    parse_seven_line_plan,
    token_fields_from_offsets,
    torch_field_balanced_loss,
    torch_lookahead_loss,
)

try:
    import torch
except ImportError:
    torch = None


PLAN = (
    "formula: Li2O\n"
    "anion: oxide\n"
    "charge: neutral_plausible\n"
    "lattice: hexagonal\n"
    "spacegroup: sg_168_194\n"
    "volume: volpa_005_009\n"
    "end: plan"
)


class H1A2PlannerObjectiveTests(unittest.TestCase):
    def test_exact_schema_and_spans(self):
        spans = parse_seven_line_plan(PLAN)
        self.assertEqual(
            [span.field for span in spans],
            [
                "formula",
                "anion",
                "charge",
                "lattice",
                "spacegroup",
                "volume",
                "end",
            ],
        )
        self.assertEqual(spans[0].value, "Li2O")
        self.assertEqual(spans[-1].value, "plan")
        self.assertEqual(spans[0].start, 0)
        self.assertEqual(spans[-1].end, len(PLAN))

    def test_schema_rejects_missing_extra_reordered_and_empty_lines(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_seven_line_plan("\n".join(PLAN.splitlines()[:-1]))
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_seven_line_plan(PLAN + "\nextra: no")
        with self.assertRaisesRegex(ValueError, "expected line"):
            parse_seven_line_plan(PLAN.replace("anion: oxide", "charge: oxide"))
        with self.assertRaisesRegex(ValueError, "empty"):
            parse_seven_line_plan(PLAN.replace("anion: oxide", "anion: "))

    def test_offset_mapping_assigns_lines_and_terminal_special(self):
        offsets = []
        cursor = 0
        for line in PLAN.splitlines(keepends=True):
            offsets.append((cursor, cursor + len(line)))
            cursor += len(line)
        offsets.append((0, 0))
        self.assertEqual(
            token_fields_from_offsets(PLAN, offsets),
            (
                "formula",
                "anion",
                "charge",
                "lattice",
                "spacegroup",
                "volume",
                "end",
                "end",
            ),
        )

    def test_field_balanced_mean_is_length_invariant_within_groups(self):
        fields = (
            "formula",
            "formula",
            "anion",
            "charge",
            "lattice",
            "spacegroup",
            "volume",
            "end",
        )
        losses = (2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 3.0, 4.0)
        total, means = field_balanced_mean(losses, fields)
        expected = (
            FIELD_GROUP_WEIGHTS["formula"] * 2.0
            + FIELD_GROUP_WEIGHTS["chemistry"] * 1.0
            + FIELD_GROUP_WEIGHTS["geometry"] * 3.0
            + FIELD_GROUP_WEIGHTS["terminator"] * 4.0
        )
        self.assertAlmostEqual(total, expected)
        self.assertEqual(
            means,
            {
                "formula": 2.0,
                "chemistry": 1.0,
                "geometry": 3.0,
                "terminator": 4.0,
            },
        )

    def test_field_balanced_mean_fails_closed_on_missing_group(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            field_balanced_mean([1.0], ["formula"])

    def test_field_group_ids_match_registered_groups(self):
        self.assertEqual(
            encode_field_group_ids(
                ["formula", "anion", "charge", "lattice", "end"]
            ),
            (
                FIELD_GROUP_IDS["formula"],
                FIELD_GROUP_IDS["chemistry"],
                FIELD_GROUP_IDS["chemistry"],
                FIELD_GROUP_IDS["geometry"],
                FIELD_GROUP_IDS["terminator"],
            ),
        )

    def test_lookahead_vocabularies_are_sorted_and_fail_on_unknown(self):
        other = PLAN.replace("anion: oxide", "anion: sulfide").replace(
            "charge: neutral_plausible",
            "charge: all_metal",
        )
        vocabs = build_lookahead_vocabs([PLAN, other])
        self.assertEqual(vocabs["anion"], ("oxide", "sulfide"))
        encoded = encode_lookahead_labels(PLAN, vocabs)
        self.assertEqual(encoded["anion"], 0)
        with self.assertRaisesRegex(ValueError, "unknown"):
            encode_lookahead_labels(
                PLAN.replace("anion: oxide", "anion: halide"),
                vocabs,
            )


@unittest.skipUnless(torch is not None, "optional torch dependency is absent")
class H1A2PlannerTorchObjectiveTests(unittest.TestCase):
    def test_torch_field_loss_matches_reference(self):
        losses = torch.tensor(
            [[2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 3.0, 4.0]],
            requires_grad=True,
        )
        fields = (
            "formula",
            "formula",
            "anion",
            "charge",
            "lattice",
            "spacegroup",
            "volume",
            "end",
        )
        group_ids = torch.tensor([encode_field_group_ids(fields)])
        mask = torch.ones_like(group_ids, dtype=torch.bool)
        result, diagnostics = torch_field_balanced_loss(
            losses,
            group_ids,
            mask,
        )
        expected, _ = field_balanced_mean(losses.detach()[0].tolist(), fields)
        self.assertAlmostEqual(float(result.detach()), expected, places=6)
        self.assertEqual(set(diagnostics), {
            "formula",
            "chemistry",
            "geometry",
            "terminator",
            "field_balanced",
        })
        result.backward()
        self.assertTrue(torch.isfinite(losses.grad).all())

    def test_lookahead_heads_are_seeded_and_loss_covers_seven_heads(self):
        sizes = {
            "anion": 2,
            "charge": 3,
            "lattice": 2,
            "spacegroup": 4,
            "volume": 2,
        }
        heads1 = build_lookahead_heads(8, sizes, seed=17)
        heads2 = build_lookahead_heads(8, sizes, seed=17)
        self.assertEqual(len(heads1), len(LOOKAHEAD_HEAD_SPECS))
        for boundary, target in LOOKAHEAD_HEAD_SPECS:
            name = lookahead_head_name(boundary, target)
            self.assertTrue(torch.equal(heads1[name].weight, heads2[name].weight))

        hidden = torch.randn(2, 5, 8, requires_grad=True)
        positions = {
            "formula": torch.tensor([1, 1]),
            "lattice": torch.tensor([3, 3]),
        }
        labels = {
            field: torch.tensor([0, size - 1])
            for field, size in sizes.items()
        }
        loss, diagnostics = torch_lookahead_loss(
            hidden,
            positions,
            labels,
            heads1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            len([key for key in diagnostics if key.endswith("_loss")]),
            7,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())


if __name__ == "__main__":
    unittest.main()
