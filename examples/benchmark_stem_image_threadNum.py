import click
from stempy import io, image
import sys
import time
import glob
import numpy as np
from PIL import Image

def save_img(stem_image_data, name):
    min = np.min(stem_image_data)
    max = np.max(stem_image_data)

    stem_image_data = stem_image_data.reshape((160, 160))
    stem_image_data = np.interp(stem_image_data, [min, max], [0, 256])
    stem_image_data = stem_image_data.astype(np.uint8)
    img = Image.fromarray(stem_image_data)
    img.save(name)

# Beware of disk caching when using this benchmark
@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-n', '--num-runs', default=1, help='Number of runs to perform')
def run_benchmarks(files, num_runs):
    """Run benchmarks using the files given in the arguments
    Example: "python benchmark_stem_image.py /path/to/data/data*.dat"
    """

    if len(files) == 0:
        sys.exit('No files found')

    times = []
    threadNumTotal = 32
    #inner_radii = [0, 40]
    #outer_radii = [288, 288]   
    for i in range(num_runs):
        print("round: " + str(i+1))
        threadNum = 1
        while(threadNum<=threadNumTotal):
            start = time.time()
            reader = io.reader(files)
            img = image.create_stem_image(reader, 40, 288, width=160, height=160, threadNum=threadNum)

            #reader = io.reader(files, version=io.FileVersion.VERSION2)
            #imgs = image.create_stem_image(reader, 40, 288, width=40, height=40, threadNum=threadNum)

            end = time.time()
            times.append(end - start)
            print('Thread number ' + str(threadNum) + ' : {:0.2f} seconds'.format(times[-1]))
            threadNum = threadNum*2

            
            #suffix = str(40) + '_' + str(288) + '.png'
            #save_img(img, 'img_' + suffix)

    print('Number of round was:', num_runs)


if __name__ == '__main__':
    run_benchmarks()