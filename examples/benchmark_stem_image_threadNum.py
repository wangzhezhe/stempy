import click
from stempy import io, image
import sys
import time

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
    for i in range(num_runs):
        print("round: " + str(i+1))
        threadNum = 1
        while(threadNum<=threadNumTotal):
            start = time.time()
            reader = io.reader(files)
            img = image.create_stem_image(reader, 40, 288, width=160, height=160, threadNum=threadNum)

            end = time.time()
            times.append(end - start)
            print('Thread number ' + str(threadNum) + ' : {:0.2f} seconds'.format(times[-1]))
            threadNum = threadNum*2

    print('Number of round was:', num_runs)


if __name__ == '__main__':
    run_benchmarks()
