import glob
from stempy import io, image
import numpy as np
from PIL import Image
from mpi4py import MPI

width = 513
height = 513

def save_img(stem_image_data, name):
    min = np.min(stem_image_data)
    max = np.max(stem_image_data)

    stem_image_data = stem_image_data.reshape((width, width))
    stem_image_data = np.interp(stem_image_data, [min, max], [0, 256])
    stem_image_data = stem_image_data.astype(np.uint8)
    img = Image.fromarray(stem_image_data)
    img.save(name)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
comm.Barrier()
start = MPI.Wtime()

files = glob.glob('/global/cscratch1/sd/cjh1/scan_0000000308/*.dat')
files_per_rank = len(files) // world_size
left_over = len(files) % world_size
if rank < left_over:
    offset = rank*(files_per_rank+1)
    files = files[offset:offset+files_per_rank+1]
else:
    offset = rank*files_per_rank+left_over
    files = files[offset:offset+files_per_rank]
# Create local sum
reader = io.reader(files, version=io.FileVersion.VERSION2)
local_stem = image.create_stem_image(reader, width, height, 40, 288)

# Now reduce to root
global_stem_bright = np.zeros(width*height, dtype='uint64')
global_stem_dark = np.zeros(width*height, dtype='uint64')
comm.Reduce(local_stem.bright, global_stem_bright, op=MPI.SUM)
comm.Reduce(local_stem.dark, global_stem_dark, op=MPI.SUM)

if rank == 0:
    save_img(global_stem_bright, 'bright.png')
    save_img(global_stem_dark, 'dark.png')

comm.Barrier()
end = MPI.Wtime()

if rank == 0:
    print('time: %s' % (end - start))
