"""Mock HTTP download edge cases without fetching external ETH3D archives."""
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_eth3d


class FakeHTTP(io.BytesIO):
    def __init__(self, payload, status=200, mime="application/x-7z-compressed"):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload)), "Content-Type": mime}


class TestDownloadHTTP(unittest.TestCase):
    def test_new_download_and_cached(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "fixture.7z"
            with patch.object(download_eth3d.urllib.request, "urlopen",
                              return_value=FakeHTTP(b"7z-fixture")):
                download_eth3d.download_archive("https://example.org/file.7z", dest, 100)
            self.assertEqual(dest.read_bytes(), b"7z-fixture")
            with patch.object(download_eth3d.urllib.request, "urlopen",
                              side_effect=AssertionError("should use cache")):
                download_eth3d.download_archive("https://example.org/file.7z", dest, 100)

    def test_resume_range_206(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "fixture.7z"
            dest.with_suffix(".7z.part").write_bytes(b"abc")
            with patch.object(download_eth3d.urllib.request, "urlopen",
                              return_value=FakeHTTP(b"def", status=206)) as fetch:
                download_eth3d.download_archive("https://example.org/file.7z", dest, 100)
            self.assertEqual(dest.read_bytes(), b"abcdef")
            self.assertEqual(fetch.call_args.args[0].get_header("Range"), "bytes=3-")

    def test_range_ignored_restarts(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "fixture.7z"
            dest.with_suffix(".7z.part").write_bytes(b"old")
            with patch.object(download_eth3d.urllib.request, "urlopen",
                              return_value=FakeHTTP(b"new", status=200)):
                download_eth3d.download_archive("https://example.org/file.7z", dest, 100)
            self.assertEqual(dest.read_bytes(), b"new")

    def test_reject_html(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "fixture.7z"
            with patch.object(download_eth3d.urllib.request, "urlopen",
                              return_value=FakeHTTP(b"<html/>", mime="text/html")):
                with self.assertRaises(RuntimeError):
                    download_eth3d.download_archive("https://example.org/file.7z", dest, 100)
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
