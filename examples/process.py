from stempy import io
import glob

for i, f in enumerate(glob.glob('/home/zhe/Downloads/smallScanningDiffraction/data*.dat')):
  print(i,f)
  reader = io.reader(f)
  print("reader ok")
  reader.process(url='http://localhost:5000', stream_id=i, concurrency=16)

