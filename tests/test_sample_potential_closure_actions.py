import unittest

try:
    import torch
    from crystal_dlm.spad_generation import _transaction_candidate_tokens
    from scripts.sample_potential_closure_actions import (
        MAX_PROPOSAL_ATTEMPTS,
        active_block_escape_positions,
        first_unique_legal_attempts,
        request_seed,
    )
except ModuleNotFoundError:
    torch = None
    _transaction_candidate_tokens = None
    MAX_PROPOSAL_ATTEMPTS = 8
    active_block_escape_positions = None
    first_unique_legal_attempts = None
    request_seed = None


def attempt(action, *, valid=True, name="dlm_proposal"):
    return {
        "attempt_kind": name,
        "action_token_ids": list(action),
        "valid_action": bool(valid),
    }


@unittest.skipIf(
    first_unique_legal_attempts is None,
    "sampler runtime dependencies unavailable",
)
class PotentialClosureActionSelectionTest(unittest.TestCase):
    def test_first_unique_legal_actions_follow_request_order(self):
        retained, audit = first_unique_legal_attempts(
            [
                attempt([11, 12, 13]),
                attempt([11, 12, 13]),
                attempt([21, 22, 23], valid=False),
                attempt([31, 32, 33]),
                attempt([41, 42, 43]),
            ],
            existing_signatures=[[1, 2, 3]],
            limit=2,
        )
        self.assertEqual(
            [row["action_token_ids"] for row in retained],
            [[11, 12, 13], [31, 32, 33]],
        )
        self.assertEqual(
            [row["retention_status"] for row in audit],
            ["retained", "duplicate", "invalid", "retained", "not_needed"],
        )

    def test_duplicate_of_existing_noop_is_merged_into_audit_only(self):
        retained, audit = first_unique_legal_attempts(
            [attempt([7, 8, 9], name="clean_teacher")],
            existing_signatures=[[7, 8, 9]],
            limit=2,
        )
        self.assertEqual(retained, [])
        self.assertEqual(audit[0]["retention_status"], "duplicate")
        self.assertFalse(audit[0]["retained"])

    def test_only_first_eight_proposal_attempts_are_examined(self):
        attempts = [attempt([index, index + 1]) for index in range(10)]
        retained, audit = first_unique_legal_attempts(attempts, limit=10)
        self.assertEqual(len(audit), MAX_PROPOSAL_ATTEMPTS)
        self.assertEqual(len(retained), MAX_PROPOSAL_ATTEMPTS)
        self.assertEqual(retained[-1]["action_token_ids"], [7, 8])

    def test_active_block_escape_is_rejected(self):
        source = [10, 11, 12, 13, 14, 15, 16]
        legal = [10, 21, 22, 23, 14, 15, 16]
        escaped = [10, 21, 22, 23, 99, 15, 16]
        self.assertEqual(active_block_escape_positions(source, legal, [1, 2, 3]), ())
        self.assertEqual(
            active_block_escape_positions(source, escaped, [1, 2, 3]),
            (4,),
        )


@unittest.skipIf(torch is None, "torch unavailable")
class PotentialClosureRequestSeedTest(unittest.TestCase):
    def test_request_keyed_sampling_is_batch_order_invariant(self):
        group_ids = [17, 901]
        seeds = [request_seed(20260904, group_idx, 3) for group_idx in group_ids]
        logits = torch.zeros((2, 1, 128), dtype=torch.float32)
        logits[:, :, 20:60] = 1.0
        first = _transaction_candidate_tokens(
            logits,
            active_absolute_positions={0: 0, 1: 0},
            temperature=0.7,
            remasking="low_confidence",
            sampling_seeds_by_batch=seeds,
            salt=1009,
        )
        swapped = _transaction_candidate_tokens(
            logits.flip(0),
            active_absolute_positions={0: 0, 1: 0},
            temperature=0.7,
            remasking="low_confidence",
            sampling_seeds_by_batch=list(reversed(seeds)),
            salt=1009,
        )
        self.assertEqual(int(first[0, 0]), int(swapped[1, 0]))
        self.assertEqual(int(first[1, 0]), int(swapped[0, 0]))

    def test_request_seed_rejects_attempt_nine(self):
        with self.assertRaisesRegex(ValueError, "1..8"):
            request_seed(20260904, 0, 9)


if __name__ == "__main__":
    unittest.main()
