"""The cache-only collation must mask every single-pass field and every invalid teacher path."""

from __future__ import annotations

import json
import unittest

from sdsafe_v6 import aux_data


class AuxPolicyTest(unittest.TestCase):
    def test_single_pass_is_not_trainable(self):
        _, manifest, _ = aux_data.build(write=False)
        s = manifest["stats"]
        self.assertGreater(s["single"], 0)
        # strict PROTOCOL §4: trainable == both-pass agreement, never agreement + single
        self.assertEqual(s["trainable_rows"]["action"], s["field_agree"]["action"])
        self.assertNotEqual(s["trainable_rows"]["action"],
                            s["field_agree"]["action"] + s["single"])
        for field in ("action", "rel", "align"):
            self.assertLessEqual(s["trainable_rows"][field], s["both"])

    def test_invalid_teacher_paths_are_not_usable_arg_supervision(self):
        _, manifest, _ = aux_data.build(write=False)
        s = manifest["stats"]
        self.assertGreater(s["arg_path_invalid"], 0)
        self.assertLessEqual(s["arg_event_usable"], s["arg_path_valid"])

    def test_manifest_pins_all_three_caches_and_zero_network_policy(self):
        _, manifest, _ = aux_data.build(write=False)
        self.assertEqual(set(manifest["inputs"]),
                         {"aux_calls_v2.jsonl", "aux2_calls.jsonl", "pilot400_calls.jsonl"})
        self.assertIn("strict both-pass", manifest["policy"])
        self.assertIn("no network", manifest["policy"])
        self.assertTrue(manifest["stats"]["excluded_forged_trusted_role_rows"] > 0)


if __name__ == "__main__":
    unittest.main()
