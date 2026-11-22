"""Fail-closed annotation guard tests.  They must prove zero network calls before any key read."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from sdsafe_v6 import annotation_guard, teacher


class AnnotationGuardTest(unittest.TestCase):
    def test_constants_are_zero(self):
        self.assertTrue(annotation_guard.ANNOTATIONS_LOCKED)
        self.assertEqual(annotation_guard.ANNOTATION_REQUESTS_ALLOWED_PER_RUN, 0)

    def test_call_refuses_before_key_and_network(self):
        calls = {"urlopen": 0, "key_read": 0}

        class FakeAuth:
            def read_text(self, *a, **k):
                calls["key_read"] += 1
                raise AssertionError("credential file must not be read")

        def fake_urlopen(*a, **k):
            calls["urlopen"] += 1
            raise AssertionError("network must not be touched")

        with mock.patch.object(teacher, "AUTH_PATH", FakeAuth()), \
                mock.patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaises(annotation_guard.AnnotationLocked):
                teacher.call("not sent", pass_id="A")
            with self.assertRaises(annotation_guard.AnnotationLocked):
                teacher._key()
            with self.assertRaises(annotation_guard.AnnotationLocked):
                teacher.label_events([], cache_path=Path("/tmp/never-written.jsonl"))
        self.assertEqual(calls, {"urlopen": 0, "key_read": 0})

    def test_frozen_manifest_matches_current_caches(self):
        got = annotation_guard.verify_frozen_manifest()
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["requests_allowed"], 0)
        frozen = json.loads(annotation_guard.FROZEN_MANIFEST.read_text())
        self.assertEqual(frozen["network_requests_allowed"], 0)
        self.assertEqual(set(frozen["frozen_teacher_caches"]),
                         {"aux2_calls.jsonl", "aux_calls_v2.jsonl", "pilot400_calls.jsonl"})


if __name__ == "__main__":
    unittest.main()
