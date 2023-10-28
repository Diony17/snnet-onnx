/* This code is ONNX execution with "resnet50v2.onnx"
 * by Jiwon Kim
 * resnet50v2.onnx (input: "data"/float32[1,3,224,224], output: "resnetv24_dense0_fwd"/float32[1,1000])
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

	/* Input information */
    constexpr int64_t numChannels = 3;
    constexpr int64_t width = 224;
    constexpr int64_t height = 224;
    constexpr int64_t numClasses = 1000;
    constexpr int64_t numInputElements = numChannels * height * width;

	/* Load onnx model */
	const string pretrained_dir = "./pretrained/onnx/";
	string model_path = pretrained_dir + "resnet50v2.onnx"; // resnet50v2.onnx
	cout << "Loading ONNX model..." << endl;
	Ort::Session onnx_session(env, model_path.c_str(), session_options);

	/* Input and output setting */
	// Make dummy input
	cout << "Creating dummy input tensor..." << endl;
	vector<float> input_tensor_values(numInputElements, 0.5);  // Fill with 0.5 as an example
	vector<int64_t> input_tensor_shape = {1, numChannels, height, width};  // batch size of 1 (input: float32[1,3,224,224])
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());

	// Set input tensor
	cout << "Setting input tensor..." << endl;
	const char* input_node_name = "data";
	vector<const char*> input_node_names = { input_node_name };

	// Set output tensor
	cout << "Setting output tensor..." << endl;
	const char* output_node_name = "resnetv24_dense0_fwd";
	vector<const char*> output_node_names = { output_node_name };

	// Create a placeholder for the output tensor
	vector<float> output_tensor_values(numClasses);
	vector<int64_t> output_tensor_shape = {1, numClasses};  // Assuming batch size of 1 (output: float32[1,1000]) 
	Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
		memory_info, output_tensor_values.data(), output_tensor_values.size(), output_tensor_shape.data(), output_tensor_shape.size());

	// Use std::move to add Ort::Value objects to the vectors
	vector<Ort::Value> input_tensors;
	input_tensors.emplace_back(std::move(input_tensor));

	vector<Ort::Value> output_tensors;
	output_tensors.emplace_back(std::move(output_tensor));

	// Run inference
	cout << "Running inference..." << endl;
	onnx_session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), output_tensors.data(), 1);

	// Get pointer to output tensor float values
	cout << "Getting output tensor data..." << endl;
	float* floatarr = output_tensors.front().GetTensorMutableData<float>();
	int predicted_class = std::distance(floatarr, std::max_element(floatarr, floatarr + 1000));

    // Use output for the application
	cout << "Finished!" << endl;
    return 0;

}
