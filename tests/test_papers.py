"""Offline tests for the on-demand research paper PDF downloader."""
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_papers


class MockResponse(io.BytesIO):
    def __init__(self, data: bytes, content_type: str = "application/pdf"):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data)), "Content-Type": content_type}


class TestPaperDownloader(unittest.TestCase):
    def test_catalog_has_research_core(self):
        self.assertTrue({"vggt", "directuv", "matlat", "gaussian2d"} <= set(download_papers.PAPERS))

    def test_real_pdf_magic_and_sha_cache(self):
        body = b"%PDF-1.7\nminimal-fixture"
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            with patch.object(download_papers.urllib.request, "urlopen",
                              return_value=MockResponse(body)) as remote:
                row = download_papers.download_one("vggt", folder, 1024)
            self.assertEqual(row["bytes"], len(body))
            self.assertEqual(len(row["sha256"]), 64)
            self.assertTrue((folder / row["filename"]).is_file())
            with patch.object(download_papers.urllib.request, "urlopen",
                              side_effect=AssertionError("cache unexpectedly downloaded")):
                self.assertEqual(download_papers.download_one("vggt", folder, 1024), row)
            remote.assert_called_once()

    def test_reject_html_and_size_cap(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            with patch.object(download_papers.urllib.request, "urlopen",
                              return_value=MockResponse(b"<html>bad</html>", "text/html")):
                with self.assertRaises(RuntimeError):
                    download_papers.download_one("directuv", folder, 1024)
            self.assertFalse(list(folder.glob("*.part")))
            with patch.object(download_papers.urllib.request, "urlopen",
                              return_value=MockResponse(b"%PDF-12345")):
                with self.assertRaises(RuntimeError):
                    download_papers.download_one("directuv", folder, 5)
            self.assertFalse(list(folder.glob("*.pdf")))
            self.assertFalse(list(folder.glob("*.part")))

    def test_reject_non_pdf_signature(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(download_papers.urllib.request, "urlopen",
                              return_value=MockResponse(b"not-a-pdf")):
                with self.assertRaises(ValueError):
                    download_papers.download_one("matlat", Path(d), 512)


if __name__ == "__main__":
    unittest.main()
