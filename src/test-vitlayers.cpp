/* This code is ONNX execution with "deit_{type}_patch16_224_layer_#.onnx" 
 * by Jiwon Kim, MOBED, Yonsei Univ.
 * {type}: tiny, small, base; #: 0 - 11
 *      tiny  (input: "input.1"/float32[1,197,192], output: "100"/float32[1,197,192])
 *      small (input: "input.1"/float32[1,197,384], output: "100"/float32[1,197,384])
 *      base  (input: "input.1"/float32[1,197,768], output: "100"/float32[1,197,768])
 */

#include <iostream>
#include <vector>
#include <algorithm>

#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace std;

int main(void) {
    /* Initialize ONNX Runtime */
	cout << "Initializing ONNX Runtime..." << endl;
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelInference");
    
	/* Initialize session options */
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);

	/* Set input and output information */
    cout << "Setting model information..." << endl;
    constexpr int64_t height  = 197;
    constexpr int64_t t_width = 192; // tiny
    constexpr int64_t s_width = 384; // small
    constexpr int64_t b_width = 768; // base
    constexpr int64_t MaxNumInputElements  = height * b_width;
    constexpr int64_t MaxNumOutputElements = height * b_width;

    vector<const char*> vit_types = {"tiny", "small", "base"};
    const char* input_node_name = "input.1";
    const char* output_node_name = "100";
    vector<const char*> input_node_names = { input_node_name };
    vector<const char*> output_node_names = { output_node_name };

	/* Load and run the stitch layer models */
	const string pretrained_dir = "./pretrained/onnx/";
    string model_path;
    int64_t numInputElements, numOutputElements;
    int64_t i_width, o_width;
    
    // Memory assign to avoid performance degradation
    vector<float> input_tensor_values;
    input_tensor_values.reserve(MaxNumInputElements);
    vector<float> output_tensor_values;
    output_tensor_values.reserve(MaxNumOutputElements);

    for (int i = 0; i <= 2; i ++) {     
        // set model shape based on type   
        if (i == 0) { // tiny
            i_width = t_width; o_width = t_width;
        } else if (i == 1) { // small
            i_width = s_width; o_width = s_width;
        } else { // base
            i_width = b_width; o_width = b_width;
        }
        numInputElements  = height * i_width;
        numOutputElements = height * o_width;

        for (int j = 0; j <= 11; j ++) {
            // Load model
            model_path = pretrained_dir + "deit_" + vit_types[i] + "_patch16_224_layer_" + to_string(j) + ".onnx";
            
            cout << "\nLoading ONNX model " + model_path + "..." << endl;
            Ort::Session onnx_session(env, model_path.c_str(), session_options);

            // Set input tensor
            cout << "Creating input and output tensors..." << endl;
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            input_tensor_values.assign(numInputElements, 0.5);  // Fill with 0.5 as an example
            vector<int64_t> input_tensor_shape = {1, height, i_width};  // batch size of 1
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());
            vector<Ort::Value> input_tensors;
            input_tensors.emplace_back(std::move(input_tensor));

            // Set output tensor
            output_tensor_values.assign(numOutputElements, 0.0);
            vector<int64_t> output_tensor_shape = {1, height, o_width};  // batch size of 1
            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info, output_tensor_values.data(), output_tensor_values.size(), output_tensor_shape.data(), output_tensor_shape.size());
            vector<Ort::Value> output_tensors;
            output_tensors.emplace_back(std::move(output_tensor));

            // Run inference
            cout << "Running inference..." << endl;
            onnx_session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), output_tensors.data(), 1);

            // Get pointer to output tensor float values
            cout << "Getting output tensor data..." << endl;
            float* floatarr = output_tensors.front().GetTensorMutableData<float>();

            // Use output for the application
            cout << model_path + " Finished!" << endl;
        }
    }

}