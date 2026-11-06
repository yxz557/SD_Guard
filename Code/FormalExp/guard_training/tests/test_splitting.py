from __future__ import annotations

import unittest

from guard_training.common.splitting import find_group_stratified_split


class GroupSplitTest(unittest.TestCase):
    def test_groups_never_cross_splits(self) -> None:
        attacks = ["direct", "ignore_previous", "injecagent", "important", "tool"]
        suites = ["workspace", "slack", "travel", "banking"]
        rows = []
        for group_index in range(80):
            suite = suites[group_index % len(suites)]
            for attack_index, attack in enumerate(attacks):
                rows.append(
                    {
                        "sample_id": f"s-{group_index}-{attack_index}",
                        "group_id": f"g-{group_index}",
                        "suite_name": suite,
                        "attack_type": attack,
                        "label": int((group_index + attack_index) % 7 == 0),
                    }
                )
        result = find_group_stratified_split(rows, seed=7, trials=200)
        self.assertEqual(set(result.group_to_split), {f"g-{index}" for index in range(80)})
        self.assertEqual(set(result.group_to_split.values()), {"train", "validation", "test"})
        for row in rows:
            self.assertIn(result.group_to_split[row["group_id"]], {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()

