#include "image.h"
#include "mask.h"

#include <memory>
#include <cmath>
#include <mpi.h>
#include <lsq/lsqcpp.h>

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

  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(NULL, NULL);
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int numberOfRanks;
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);

  cout << "rank: " << rank  << endl;

  auto scanSize = rows * columns;

  // First calculate the fraction of the samples we are going to sample
  auto sampleFraction = numberOfSamples/static_cast<double>(scanSize);

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

  cout << "dark mean: " << mean << endl;

  return darkReference;
}

double calculateMean(uint64_t values[], int size) {
  double sum = 0;
  for (int i=0; i<size; i++) {
    sum += values[i];
  }

  return sum / size;
}

double calculateVariance(uint64_t values[], int size, double mean) {
  double variance = 0;

  for (int i=0; i<size; i++) {
    variance += pow(values[i] - mean, 2.0);
  }

  return variance;
}

std::pair<uint64_t, uint64_t> findMinMax(uint64_t values[], int size) {

  auto min = values[0], max = values[0];

  for (int i=0; i< size; i++) {
    if (values[i] < min) {
      min = values[i];
    }
    if (values[i] > max) {
      max= values[i];
    }
  }

  return std::make_pair(min, max);
}

uint64_t findMax(uint64_t values[], int size) {

  auto max = values[0];

  for (int i=0; i< size; i++) {
    if (values[i] > max) {
      max = values[i];
    }
  }

  return max;
}

class GaussianErrorFunction : public lsq::ErrorFunction<double>
{

public:
    GaussianErrorFunction(uint64_t* bins, uint32_t* histogram, int size)
          : size(size), bins(bins), histogram(histogram)
      {}
    void _evaluate(
        const lsq::Vectord &state,
        lsq::Vectord& outVal,
        lsq::Matrixd& outJac) override
    {
      outVal.resize(size);

      for(Eigen::Index i = 0; i < size; ++i) {
        outVal[i] = state[0]*exp(-0.5*pow((bins[i]-state[1])/state[2], 2)) - histogram[i];
      }
    }
private:
    int size;
    uint64_t* bins;
    uint32_t* histogram;

};

uint64_t calculateCountingThreshhold(std::vector<Block>& blocks, int rows, int columns,
    const DarkFieldReference& darkReference, int sigmaThreshold, int numberOfSamples,
    int upperLimit)
{
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(NULL, NULL);
  }
  int numberOfRanks;
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);

  auto detectorImageRows = blocks[0].header.rows;
  auto detectorImageColumns = blocks[0].header.columns;
  auto numberOfPixels = detectorImageRows * detectorImageRows;

  int numberScanImageSamples = rows*columns*numberOfSamples;
  uint64_t samples[numberScanImageSamples] = {0};
  for(int i=0; i< numberOfSamples; i++) {
    auto randIndex = rand() % numberOfPixels;
    double sum = 0;
    uint64_t count = 0;
    for(const Block &block: blocks) {
      auto data = block.data.get();
      for (int j=0; j<block.header.imagesInBlock; j++) {
        auto imageIndex = j*numberOfPixels+randIndex;
        auto imageNumber = block.header.imageNumbers[j];
        //cout << "value: " << data[imageIndex] << endl;
        auto referenceIndex = static_cast<int>(ceil(imageNumber / static_cast< float >(columns)))-1;
        //cout << "dark: " << darkReference.referenceFrame[referenceIndex] << endl;
        samples[imageNumber-1] = data[imageIndex] - darkReference.referenceFrame[referenceIndex];
      }
    }
  }

  auto mean = calculateMean(samples, numberScanImageSamples);
  auto variance = calculateVariance(samples, numberScanImageSamples, mean);

  // Now we need to reduce across nodes
  if (numberOfRanks > 1) {
    double stats[2];
    stats[0] = mean;
    stats[1] = variance;

    MPI_Allreduce(MPI_IN_PLACE, stats, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mean = stats[0] / numberOfRanks;
    variance = stats[1] / numberOfRanks;

    MPI_Allreduce(MPI_IN_PLACE, samples, numberOfSamples, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
  }

  auto stdDev = sqrt(variance);

  // zero all X-ray counts
  for (int i=0; i< numberScanImageSamples; i++) {
    if (samples[i]>(mean+upperLimit*stdDev)) {
      samples[i] = mean;
    }
  }

  // Now generate a histograms
  auto minMax = findMinMax(samples, numberScanImageSamples);
  int min = floor(std::get<0>(minMax));
  int max = ceil(std::get<1>(minMax));
  auto numberOfBins = max - min;
  uint32_t histogram[numberOfBins];
  uint64_t bins[numberOfBins];

  auto binEdge = min+1;
  for (int i=0; i<numberOfBins; i++) {
    bins[i] = binEdge++;
  }

  for (int i=0; i< numberScanImageSamples; i++) {
    auto binIndex = static_cast<int>(samples[i] - min);
    histogram[binIndex] += 1;
  }

  // Now optimize
  lsq::GaussNewton<double> optAlgo;
  optAlgo.setLineSearchAlgorithm(nullptr);
  optAlgo.setMaxIterations(20);
  optAlgo.setVerbose(true);
  GaussianErrorFunction errorFunction(bins, histogram, numberOfBins);
  optAlgo.setErrorFunction(&errorFunction);
  lsq::Vectord initialState(3);
  auto indexOfMaxElement = std::max_element(histogram, histogram+numberOfBins) - histogram;
  initialState[0] =  static_cast<double>(histogram[indexOfMaxElement]);
  initialState[1] = (bins[indexOfMaxElement+1]-bins[indexOfMaxElement])/2.0;
  initialState[2] = stdDev;
  // optimize
  auto result = optAlgo.optimize(initialState);






}

}
