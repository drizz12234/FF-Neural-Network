#include "Flatten.h"

#include <iostream>
#include <math.h>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {

void FlattenLayer::computeNaive(const LayerData& dataIn) const {

    Timer timer_flatten("timer_flatten");
    timer_flatten.start();

    LayerParams inputParams = getInputParams();
    LayerParams outputParams = getOutputParams();

    int Q = outputParams.dims[0];

    for(int i =0; i < Q; i++){
        getOutputData().get<fp32>(i) = dataIn.get<fp32>(i);
    }

    timer_flatten.stop();
}

// Compute the convolution using threads
void FlattenLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using a tiled approach
void FlattenLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using SIMD
void FlattenLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

}  // namespace ML