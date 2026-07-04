import tempfile
import unittest
from pathlib import Path

from xquik_export import load_xquik_export


class XquikExportTests(unittest.TestCase):
    def test_loads_wrapped_json_text(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tweets.json"
            path.write_text('{"data":[{"text":"Great launch","username":"ada"}]}', encoding="utf-8")

            rows = load_xquik_export(path)

        self.assertEqual(rows, [{"tweet": "Great launch", "created_at": "", "username": "ada"}])

    def test_loads_csv_tweet_field(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tweets.csv"
            path.write_text("tweet,user\nUseful update,dev\n", encoding="utf-8")

            rows = load_xquik_export(path)

        self.assertEqual(rows[0]["tweet"], "Useful update")
        self.assertEqual(rows[0]["username"], "dev")

    def test_rejects_exports_without_text(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tweets.jsonl"
            path.write_text('{"id":"1"}\n', encoding="utf-8")

            with self.assertRaises(ValueError):
                load_xquik_export(path)


if __name__ == "__main__":
    unittest.main()
