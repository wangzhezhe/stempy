#include "image.h"
#include "mask.h"

#include <memory>
#include <cmath>
#include <mpi.h>

using namespace std;

namespace stempy {

Image::Image(uint32_t width, uint32_t height) :
    width(width), height(height),
    data(new uint64_t[width * height], std::default_delete<uint64_t[]>())
{ }

STEMImage::STEMImage(uint32_t width, uint32_t height)
{
  this->bright = Image(width, height);
  this->dark = Image(width, height);
}

STEMValues calculateSTEMValues(uint16_t data[], int offset,
                               int numberOfPixels,
                               uint16_t brightFieldMask[],
                               uint16_t darkFieldMask[],
                               uint32_t imageNumber)
{
  STEMValues stemValues;
  stemValues.imageNumber = imageNumber;
  for (int i=0; i<numberOfPixels; i++) {
    auto value = data[offset + i];

    stemValues.bright += value & brightFieldMask[i];
    stemValues.dark  += value & darkFieldMask[i];
  }

  return stemValues;
}

DarkFieldReference::DarkFieldReference(uint32_t size) :
    size(size),
    referenceFrame(new uint64_t[size], std::default_delete<uint64_t[]>())
{ }

STEMImage createSTEMImage(std::vector<Block>& blocks, int rows, int columns,  int innerRadius, int outerRadius)
{
  STEMImage image(rows, columns);

  // Get image size from first block
  auto detectorImageRows = blocks[0].header.rows;
  auto detectorImageColumns = blocks[0].header.columns;
  auto numberOfPixels = detectorImageRows * detectorImageRows;

  auto brightFieldMask = createAnnularMask(detectorImageRows, detectorImageColumns, 0, outerRadius);
  auto darkFieldMask = createAnnularMask(detectorImageRows, detectorImageColumns, innerRadius, outerRadius);

  for(const Block &block: blocks) {
    auto data = block.data.get();
    for (int i=0; i<block.header.imagesInBlock; i++) {
      auto stemValues = calculateSTEMValues(data, i*numberOfPixels, numberOfPixels,
                                            brightFieldMask, darkFieldMask);
      image.bright.data[block.header.imageNumbers[i]-1] = stemValues.bright;
      image.dark.data[block.header.imageNumbers[i]-1] = stemValues.dark;
    }
  }

  delete[] brightFieldMask;
  delete[] darkFieldMask;

  return image;
}

DarkFieldReference createDarkFieldReference(std::vector<Block>& blocks, int rows, int columns,
    int numberOfSamples, int sampleStripWidth) {

  DarkFieldReference darkReference(columns);

  MPI_Init(NULL, NULL);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int numberOfRanks;
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);

  cout << "rank: " << rank  << endl;

  auto detectorImageRows = blocks[0].header.rows;
  auto detectorImageColumns = blocks[0].header.columns;
  auto numberOfPixels = detectorImageRows * detectorImageRows;

  auto referenceFrame = darkReference.referenceFrame.get();
  double mean = 0.0;
  double variance = 0.0;
  for(int i=0; i< numberOfSamples; i++) {
    auto randIndex = rand() % numberOfPixels;
    double sum = 0;
    uint64_t count = 0;
    for(const Block &block: blocks) {
      auto data = block.data.get();
      for (int j=0; j<block.header.imagesInBlock; j++) {
        auto imageIndex = j*numberOfPixels+randIndex;
        auto imageNumber = block.header.imageNumbers[j];
        // We only need the value if it appears in our sample strip
        if (imageNumber % columns <= sampleStripWidth) {
          auto referenceIndex = static_cast<int>(ceil(imageNumber / static_cast< float >(columns)))-1;
          referenceFrame[referenceIndex] += data[imageIndex];
        }
        sum += data[imageIndex];
        ++count;
      }
    }

    auto sampleMean = sum/count;
    mean += sampleMean;

    // Now calculate the variance for the sample
    count = 0;
    double sampleVariance = 0.0;
    for(const Block &block: blocks) {
      auto data = block.data.get();
      for (int j=0; j<block.header.imagesInBlock; j++) {
        auto imageIndex = j*numberOfPixels+randIndex;
        sampleVariance += pow(data[imageIndex] - sampleMean, 2.0);
        ++count;
      }
    }

    sampleVariance /= count;
    variance += sampleVariance;
  }

  mean /= numberOfSamples;
  variance /= numberOfSamples;

  // Now we need to reduce across nodes
  if (numberOfRanks > 1) {
    double stats[2];
    stats[0] = mean;
    stats[1] = variance;

    MPI_Allreduce(MPI_IN_PLACE, stats, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mean = stats[0] / numberOfRanks;
    variance = stats[1] / numberOfRanks;

    MPI_Allreduce(MPI_IN_PLACE, referenceFrame, columns, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

  }

  for(int i=0; i<columns; i++) {
    referenceFrame[i] /= (numberOfRanks*numberOfSamples*sampleStripWidth);
  }

  darkReference.mean = mean;
  darkReference.variance = variance;

  return darkReference;
}


}
