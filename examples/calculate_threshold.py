import os
from stempy import io, image


blocks = []

node_id = os.environ['PMI_RANK']

path = '/data/4dstem/smallScanningDiffraction/data00%s.dat' % node_id.zfill(2)
reader = io.reader(path)
for b in reader:
    blocks.append(b)

dark = image.create_dark_field_reference(blocks, 160, 160)
image.calculate_counting_threshhold(blocks, 160, 160, dark)
