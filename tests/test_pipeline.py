import struct,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_scene import prepare
from scene_scale import robust_extent
from common import scene_path
class TestPipeline(unittest.TestCase):
 def test_scene_names(self):
  for n in ("..","../x","a/b",""):
   with self.assertRaises(ValueError):scene_path(n)
 def test_bridge(self):
  with tempfile.TemporaryDirectory() as d:
   s=Path(d);(s/"images").mkdir();(s/"sparse").mkdir()
   for i in range(3):(s/"images"/(str(i)+".png")).write_bytes(b"x")
   for n in ("cameras.bin","images.bin","points3D.bin"):(s/"sparse"/n).write_bytes(b"x")
   self.assertTrue((prepare(s)/"cameras.bin").is_file())
 def test_scale(self):
  with tempfile.TemporaryDirectory() as d:
   with (Path(d)/"points3D.bin").open("wb") as f:
    f.write(struct.pack("<Q",32))
    for i in range(32):f.write(struct.pack("<Q3d3BdQ",i+1,i,i*.1,i*.2,128,128,128,0,0))
   self.assertGreater(robust_extent(Path(d)),20)
