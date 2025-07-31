#include "Softmax.h"

#include <iostream>
#include <math.h>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {

// Compute the convultion for the layer data
void SoftmaxLayer::computeNaive(const LayerData& dataIn) const {

    Timer timer_softmax("timer_softmax");
    timer_softmax.start();

    LayerParams inputParams = getInputParams();
    LayerParams outputParams = getOutputParams();

    int C = inputParams.dims[0];  // Number of channels of Ifmap (input  channels)

    fp32 gi = 0;
    fp32 gl = 0;
    fp64 den = 0;

    for(int l = 0; l < C; l++){
        gl = dataIn.get<fp32>(l);
        den += std::exp((fp64)gl);
    }

    for(int i = 0; i < C; i++){
        gi = dataIn.get<fp32>(i);
        fp64 num = std::exp((fp64)gi);
        if(den == 0){
            // fp32 max = pow(1.18, -38);
            // den += max;
            exit(0);
            //getOutputData().get<fp32>(i) = (num/den);
        } else {
            fp32 value_for_output = num/den;
            getOutputData().get<fp32>(i) = value_for_output;
        }
    }
    timer_softmax.stop();
}

// Compute the convolution using threads
void SoftmaxLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using a tiled approach
void SoftmaxLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using SIMD
void SoftmaxLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

}  // namespace ML