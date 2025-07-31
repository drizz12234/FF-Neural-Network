#include <iostream>
#include <sstream>
#include <vector>

#include "Config.h"
#include "Model.h"
#include "Types.h"
#include "Utils.h"
#include "layers/HiddenOne.h"

#ifdef ZEDBOARD
#include <file_transfer/file_transfer.h>
#endif

namespace ML {

// Build our ML toy model
Model buildToyModel(const Path modelPath) {
    Model model;
    logInfo("--- Building Toy Model ---");

    // --- flatten: L0 ---
    // Input shape: 8x64
    // Output shape: 2048

    model.addLayer<FlattenLayer>(
        LayerParams{sizeof(fp32), {1, 1, 20}},                                    // Input Data
        LayerParams{sizeof(fp32), {20}}                                          // Output Data
    );

    // --- dense: L1 
    // Input shape: 512
    // Output shape: 128

    model.addLayer<DenseLayer>(
        LayerParams{sizeof(fp32), {20}},                                    // Input Data
        LayerParams{sizeof(fp32), {128}},                                     // Output Data
        LayerParams{sizeof(fp32), {20, 128}, modelPath / "dense_weights.bin"},   // Weights
        LayerParams{sizeof(fp32), {128}, modelPath / "dense_biases.bin"}           // Bias
    );

    // --- dense_1: L2 
    // Input shape: 128
    // Output shape: 128

    model.addLayer<DenseLayer>(
        LayerParams{sizeof(fp32), {128}},                                    // Input Data
        LayerParams{sizeof(fp32), {128}},                                   // Output Data
        LayerParams{sizeof(fp32), {128, 128}, modelPath / "dense_1_weights.bin"}, // Weights
        LayerParams{sizeof(fp32), {128}, modelPath / "dense_1_biases.bin"}            // Bias
    );


    // --- dense_2: L3 
    // Input shape: 128
    // Output shape: 24

    model.addLayer<DenseLayer>(
        LayerParams{sizeof(fp32), {128}},                                    // Input Data
        LayerParams{sizeof(fp32), {10}},                                   // Output Data
        LayerParams{sizeof(fp32), {128, 10}, modelPath / "dense_2_weights.bin"}, // Weights
        LayerParams{sizeof(fp32), {10}, modelPath / "dense_2_biases.bin"}            // Bias
    );

    // --- softmax: L4 
    // Input shape: 24
    // Output shape: 24

    model.addLayer<SoftmaxLayer>(
        LayerParams{sizeof(fp32), {10}},                                    // Input Data
        LayerParams{sizeof(fp32), {10}}                                     // Output Data
    );

    return model;
}

// TODO

void runBasicTest(const Model& model, const Path& basePath) {
    logInfo("--- Running Basic Test ---");

    // Load an image
    LayerData img = {{sizeof(fp32), {64, 64, 3}, "./data/image_0.bin"}};
    img.loadData();

    // Compare images
    std::cout << "Comparing image 0 to itself (max error): " << img.compare<fp32>(img) << std::endl
              << "Comparing image 0 to itself (T/F within epsilon " << ML::Config::EPSILON << "): " << std::boolalpha
              << img.compareWithin<fp32>(img, ML::Config::EPSILON) << std::endl;

    // Test again with a modified copy
    std::cout << "\nChange a value by 0.1 and compare again" << std::endl;
    
    LayerData imgCopy = img;
    imgCopy.get<fp32>(0) += 0.1;

    // Compare images
    img.compareWithinPrint<fp32>(imgCopy);

    // Test again with a modified copy
    log("Change a value by 0.1 and compare again...");
    imgCopy.get<fp32>(0) += 0.1;

    // Compare Images
    img.compareWithinPrint<fp32>(imgCopy);
}

// TODO

void runLayerTest(const std::size_t layerNum, const Model& model, const Path& basePath) {
    // Load an image
    logInfo("--- Running Layer Test ---");
    dimVec inDims = {1, 256, 64};

    // Construct a LayerData object from a LayerParams one
    LayerData img({sizeof(fp32), inDims, basePath / "sample_0_data" / "layer_3_output.bin"});
    //"sample_0.bin"
    //"sample_0_data" / "layer_2_output.bin"
    img.loadData();

    Timer timer("Layer Inference");

    // Run inference on the model
    timer.start();
    const LayerData output = model.inferenceLayer(img, layerNum, Layer::InfType::NAIVE);
    timer.stop();

    // Compare the output
    // Construct a LayerData object from a LayerParams one
    dimVec outDims = model[layerNum].getOutputParams().dims;
    LayerData expected({sizeof(fp32), outDims, basePath / "sample_0_data" / "layer_4_output.bin"});
    expected.loadData();
    output.compareWithinPrint<fp32>(expected);
}

// TODO

void runInferenceTest(const Model& model, const Path& basePath) {
    // Load an image
    logInfo("--- Running Inference Test ---");
    dimVec inDims = {1,1024, 2};

    // Construct a LayerData object from a LayerParams one
    LayerData img({sizeof(fp32), inDims, basePath / "sample_0.bin"});
    img.loadData();

    Timer timer("Full Inference");

    // Run inference on the model
    timer.start();
    const LayerData output = model.inference(img, Layer::InfType::NAIVE);
    timer.stop();

    // Compare the output
    // Construct a LayerData object from a LayerParams one
    dimVec outDims = model.getOutputLayer().getOutputParams().dims;
    LayerData expected({sizeof(fp32), outDims, basePath / "sample_0_data" / "layer_18_output.bin"});
    expected.loadData();
    output.compareWithinPrint<fp32>(expected);
}

// TODO

void runTests() {
    // Base input data path (determined from current directory of where you are running the command)
    Path basePath("data");  // May need to be altered for zedboards loading from SD Cards

    // Build the model and allocate the buffers
    Model model = buildToyModel(basePath / "model");
    model.allocLayers();

    // Run some framework tests as an example of loading data
    //runBasicTest(model, basePath);

    // Run a layer inference test
    // runLayerTest(4, model, basePath);

    // Run an end-to-end inference test
    runInferenceTest(model, basePath);

    // Clean up
    model.freeLayers();
    std::cout << "\n\n----- ML::runTests() COMPLETE -----\n";
}

} // namespace ML

#ifdef ZEDBOARD
extern "C"
int main() {
    try {
        static FATFS fatfs;
        if (f_mount(&fatfs, "/", 1) != FR_OK) {
            throw std::runtime_error("Failed to mount SD card. Is it plugged in?");
        }
        ML::runTests();
    } catch (const std::exception& e) {
        std::cerr << "\n\n----- EXCEPTION THROWN -----\n" << e.what() << '\n';
    }
    std::cout << "\n\n----- STARTING FILE TRANSFER SERVER -----\n";
    FileServer::start_file_transfer_server();
}
#else
int main() {
    ML::runTests();
}
#endif