#include "Dense.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
fp32 selu(fp32 x);

// Compute the convultion for the layer data
void DenseLayer::computeNaive(const LayerData& dataIn) const {

    Timer timer_dense("timer_dense");
    timer_dense.start();

    LayerParams weightParams = getWeightParams();
    LayerParams biasParams = getBiasParams();
    LayerParams inputParams = getInputParams();
    LayerParams outputParams = getOutputParams();

    int M = outputParams.dims[0]; // Number of channels of ofmap (output channels)
    int S = inputParams.dims[0];  // Set to W of input --> Width

    for(int m = 0; m < M; m++){
        fp32 value_for_output = 0; 
        for(int s = 0; s < S; s++){
            int filter_index = s*M + m;
            int input_index = s;

            fp32 filter_val = getWeightData().get<fp32>(filter_index);
            fp32 input_val = dataIn.get<fp32>(input_index);

            value_for_output += input_val * filter_val;
        }
        fp32 bias_val = getBiasData().get<fp32>(m);
        value_for_output = value_for_output + bias_val;
        
        //SeLU
        if((M == 128)){
            value_for_output = selu(value_for_output);
        }
        int output_index = m;
        getOutputData().get<fp32>(output_index) = value_for_output;
    }
    timer_dense.stop();
}

fp32 selu(fp32 x){
    const fp32 alpha = 1.67326324;
    const fp32 scale = 1.05070099;

    if (x > 0) {
        return scale * x;
    } else {
        return scale * alpha * (std::exp(x) - 1);
    }
}

// Compute the convolution using threads
void DenseLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using a tiled approach
void DenseLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using SIMD
void DenseLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
}  // namespace ML