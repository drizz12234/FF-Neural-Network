#include "Neurons.h"

#include <iostream>
#include <math.h>

#include "../Types.h"
#include "../Utils.h"

namespace ML {

float Neuron::computeNaive(dataIn, weightIn, bias) const {

    LayerParams inputParams = getInputParams();
    LayerParams outputParams = getOutputParams();

    int Q = dataIn.size();
    int P = weightIn.size();

    float val = 0;
    float sum = 0;
    for(int i=0; i < Q; i++){  // inputs
        for(int j=0; j < P; j++){  // weights
            sum = dataIn[i] * weightIn[j];
            val = val + sum;
        }
    }
    val = val + bias;

    // RELU
    if(val < 0){
        val = 0;
    }

    return val;
}

}  // namespace ML