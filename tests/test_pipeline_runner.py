import unittest

from src.pipeline_runner import PipelineConfig, safe_slug, _write_uploads


class PipelineRunnerTests(unittest.TestCase):
    def test_safe_slug(self):
        self.assertEqual(safe_slug("Kuku Technologies / Show 1"), "Kuku_Technologies_Show_1")
        self.assertEqual(safe_slug("!!!"), "run")

    def test_config_defaults_validate(self):
        PipelineConfig().validate()

    def test_invalid_chunk_overlap_rejected(self):
        with self.assertRaises(ValueError):
            PipelineConfig(story_chunk_size=6, story_chunk_overlap=6).validate()

    def test_config_candidate_limit_rejected(self):
        with self.assertRaises(ValueError):
            PipelineConfig(max_candidates=50).validate()

    def test_write_uploads_sanitizes_and_deduplicates(self,):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            saved = _write_uploads(
                [("../Episode 1.txt", b"hello"), ("Episode 1.txt", b"world")],
                Path(tmp) / "in",
            )
            self.assertEqual([p.name for p in saved], ["Episode_1.txt", "Episode_1_2.txt"])
            self.assertEqual(saved[0].read_bytes(), b"hello")

    def test_write_uploads_rejects_non_utf8(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _write_uploads([("bad.txt", b"\xff\xfe")], Path(tmp) / "in")


if __name__ == "__main__":
    unittest.main()
