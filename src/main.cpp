/* This code is ONNX execution with the partitioned SN-Net (CVPR, 2023)
 * by Jiwon Kim, MOBED, Yonsei Univ.
 * Anchors: "deit_{type}_patch16_224_layer_#.onnx" 
 * {type}: tiny, small, base; #: 0 - 11
 *      tiny  (input: "input.1"/float32[1,197,192], output: "100"/float32[1,197,192])
 *      small (input: "input.1"/float32[1,197,384], output: "100"/float32[1,197,384])
 *      base  (input: "input.1"/float32[1,197,768], output: "100"/float32[1,197,768])
 * Stitch Layers: 
 * deit_sl_3.onnx - deit_sl_36.onnx: tiny to small
 *      (input: "onnx::MatMul_0"/float32[1,197,192], output: "5"/float32[1,197,384])
 * deit_sl_37.onnx - deit_sl_70.onnx: small to base 
 *      (input: "onnx::MatMul_0"/float32[1,197,384], output: "5"/float32[1,197,768])
 */

#include <iostream>
#include <vector>
#include <algorithm>

#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include "stitch_config.h"
#include "constants.h"
#include "image_loader.h"

using namespace std;

class StitchLayer {
	private:
		int stitch_id, stitch_config_id; // stitch_id: 3-70, stitch_config_id: 0-33
		int stitch_code; // 0: tiny, 1: small, 2: base, 3: tiny-small, 4: small-base
		pair<int, int> anchor_layer_num; // <front, back> anchors' layer to stich
		int num_stitchies;
		StitchConfig config;

	public:
    StitchLayer(int s_id) : stitch_id(s_id), config(12, 2, 1) {		
		if (stitch_id < 3) {
			stitch_code = stitch_id, stitch_config_id = -1;
			anchor_layer_num.first = 11, anchor_layer_num.second = 12;
		} else if (stitch_id > 36) {
			stitch_config_id = stitch_id - 34 - 3;
			stitch_code = 4;
			anchor_layer_num = config.getStitchConfig(stitch_config_id);
		} else {
			stitch_config_id = stitch_id - 34 - 3;
			stitch_code = 3;
			anchor_layer_num = config.getStitchConfig(stitch_config_id);
		}
		num_stitchies = config.getNumStitches();
	}
	pair<int, int> getAnchorLayerNum() const {
		return anchor_layer_num;
	}
	int getStitchCode() const {
		return stitch_code;
	}
	int getNumStitchies() const {
		return num_stitchies;
	}
};

int main(int argc, char* argv[]) {
	/* Setting stitch layer information*/
	cout << "Setting stitch layer information..." << endl;
	if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <stitch layer number>" << endl;
        exit(1);
    }
	
	int stitch_id;
    try {
        stitch_id = stoi(argv[1]);
    } catch (const exception& e) {
        cerr << "Error: Invalid number provided." << endl;
        exit(1);
    }

    cout << "Stitch layer number: " << stitch_id << endl;
	StitchLayer stitch_layer(stitch_id);

    /* Initialize ONNX Runtime environment */
	cout << "Initializing ONNX Runtime..." << endl;
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelInference");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);

	const string pretrained_dir = "./pretrained/onnx/";
	const string assets_dir = "./assets/";
    string model_path, image_path, label_path;
    int64_t numInputElements, numOutputElements;
    int64_t i_width, o_width;
	vector<const char*> input_node_names, output_node_names;

	vector<int64_t> input_tensor_shape, output_tensor_shape;
	vector<Ort::Value> input_tensors, output_tensors;
    vector<float> input_tensor_values;
    input_tensor_values.reserve(maxNumInputElements);
    vector<float> output_tensor_values;
    output_tensor_values.reserve(maxNumOutputElements);

	// Anchor information setting
	pair<int, int> anchor_num = stitch_layer.getAnchorLayerNum();
	int frontAnchorNum = anchor_num.first, backAnchorNum = anchor_num.second;
	int stitchCode = stitch_layer.getStitchCode();
	int numStitchies = stitch_layer.getNumStitchies();
	string front_anchor_name, back_anchor_name;

	// load image
	image_path = assets_dir + image_name;
    vector<float> imageVec = ImageHelpers::loadImage(image_path);
    if (imageVec.empty()) {
        cout << "Failed to load image: " << image_path << endl;
        return 1;
    }

	//load labels
	label_path = assets_dir + label_name;
    vector<string> labels = ImageHelpers::loadLabels(label_path);
    if (labels.empty()) {
        cout << "Failed to load labels: " << label_path << endl;
        return 1;
    }

	try {
		/* Embed layer execution */
		input_node_names = { input_node_name_vit_embed };
		output_node_names = { output_node_name_vit_embed };

		if (stitchCode == 0 || stitchCode == 3) { // tiny only or front tiny
			i_width = tr_width_t; o_width = tr_width_t;
			front_anchor_name = vit_types[0];
			if (stitchCode == 3) back_anchor_name = vit_types[1];
		} else if (stitchCode == 1 || stitchCode == 4) { // small only or front small
			i_width = tr_width_s; o_width = tr_width_s;
			front_anchor_name = vit_types[1];
			if (stitchCode == 4) back_anchor_name = vit_types[2];
		} else { // stitchCode == 2 (single base model)
			i_width = tr_width_b; o_width = tr_width_b;
			front_anchor_name = vit_types[2];
		}

		numInputElements  = in_numChannels * in_height * in_width;
		numOutputElements = tr_height * i_width; // Embed output shape is equal to the input shape of transformer layers

		if (imageVec.size() != numInputElements) {
			cout << "Invalid image format. Must be 224x224 RGB image." << endl;
			return 1;
		}

		model_path = pretrained_dir + "deit_" + front_anchor_name + "_patch16_224_embed.onnx";
		cout << "\nLoading ONNX model " + model_path + "..." << endl;
		Ort::Session onnx_session_embed(env, model_path.c_str(), session_options);
		Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		// Set input and output tensor
		input_tensor_values.assign(numInputElements, 0.5);  // Fill with 0.5 as an example
		copy(imageVec.begin(), imageVec.end(), input_tensor_values.begin());
		input_tensor_shape = {1, in_numChannels, in_height, in_width};  // batch size of 1 (input: float32[1,3,224,224])
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
			memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());
		input_tensors.emplace_back(move(input_tensor));

		output_tensor_values.assign(numOutputElements, 0.0);
		output_tensor_shape = {1, tr_height, o_width};  // batch size of 1
		Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
			memory_info, output_tensor_values.data(), output_tensor_values.size(), output_tensor_shape.data(), output_tensor_shape.size());
		output_tensors.emplace_back(move(output_tensor));

		// Run inference
		cout << "Running Embeding Layer inference..." << endl;
		onnx_session_embed.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), output_tensors.data(), 1);

		/*  ViT layers execution (front anchor) */
		input_node_names = { input_node_name_vit_layers };
		output_node_names = { output_node_name_vit_layers };

		numInputElements  = tr_height * i_width;
		numOutputElements = tr_height * o_width;
		
		for (int i = 0; i < frontAnchorNum + 1; i ++) { 
			// Load model
			model_path = pretrained_dir + "deit_" + front_anchor_name + "_patch16_224_layer_" + to_string(i) + ".onnx";
			cout << "\nLoading ONNX model " + model_path + "..." << endl;
			Ort::Session onnx_session(env, model_path.c_str(), session_options);
			memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

			copy(output_tensor_values.begin(), output_tensor_values.end(), input_tensor_values.begin());
			input_tensor_shape = {1, tr_height, i_width};  // batch size of 1		
			input_tensor = Ort::Value::CreateTensor<float>(
				memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());
			input_tensors.clear();
			input_tensors.emplace_back(move(input_tensor));

			// Set output tensor
			output_tensor_values.clear();
			output_tensor_values.assign(numOutputElements, 0.0);
			output_tensor_shape = {1, tr_height, o_width};  // batch size of 1
			output_tensor = Ort::Value::CreateTensor<float>(
				memory_info, output_tensor_values.data(), output_tensor_values.size(), output_tensor_shape.data(), output_tensor_shape.size());
			output_tensors.emplace_back(move(output_tensor));

			// Run inference
			cout << "Running inference..." << endl;
			onnx_session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), output_tensors.data(), 1);
		}

		/*  Stitch layer execution */
		input_node_names = { input_node_name_stitch_layers };
		output_node_names = { output_node_name_stitch_layers };

		bool stitch_flag = true;

		// set model shape based on anchor stitch code  
		if (stitchCode == 3) { // tiny - small
			i_width = tr_width_t; o_width = tr_width_s;
		} else if (stitchCode == 4) { // small - base
			i_width = tr_width_s; o_width = tr_width_b;
		} else { // base (stitchCode == 2|| stitchCode == 4) // base only or back base
			stitch_flag = false;
		}
		numInputElements  = tr_height * i_width;
		numOutputElements = tr_height * o_width;	

		if (stitch_flag) { // if we use a single anchor, skip this phase.
			model_path = pretrained_dir + "deit_sl_" + to_string(stitch_id) + ".onnx";
			cout << "\nLoading ONNX model " + model_path + "..." << endl;
			Ort::Session onnx_session_stitch(env, model_path.c_str(), session_options);
			memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

			copy(output_tensor_values.begin(), output_tensor_values.end(), input_tensor_values.begin());
			input_tensor_shape = {1, tr_height, i_width};  // batch size of 1
			input_tensor = Ort::Value::CreateTensor<float>(
				memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());
			input_tensors.clear();
			input_tensors.emplace_back(move(input_tensor));

			// Set output tensor
			output_tensor_values.assign(numOutputElements, 0.0);
			output_tensor_shape = {1, tr_height, o_width};  // batch size of 1
			output_tensor = Ort::Value::CreateTensor<float>(
				memory_info, output_tensor_values.data(), output_tensor_values.size(), output_tensor_shape.data(), output_tensor_shape.size());
			output_tensors.clear();
			output_tensors.emplace_back(move(output_tensor));

			// Run inference
			cout << "Running inference..." << endl;
			onnx_session_stitch.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), output_tensors.data(), 1);
		}

		/*  ViT layers execution (back anchor) */
		input_node_names = { input_node_name_vit_layers };
		output_node_names = { output_node_name_vit_layers };
		
		// set model shape based on anchor stitch code  
		if (stitchCode == 0) { // tiny only
			i_width = tr_width_t; o_width = tr_width_t;
		} else if (stitchCode == 1 || stitchCode == 3) { // small only or back small
			i_width = tr_width_s; o_width = tr_width_s;
		} else { // base (stitchCode == 2|| stitchCode == 4) // base only or back base
			i_width = tr_width_b; o_width = tr_width_b;
		}
		numInputElements  = tr_height * i_width;
		numOutputElements = tr_height * o_width;	

		for (int i = backAnchorNum; i < numStitchies; i ++) {
			// Load model
			model_path = pretrained_dir + "deit_" + back_anchor_name + "_patch16_224_layer_" + to_string(i) + ".onnx";
			cout << "\nLoading ONNX model " + model_path + "..." << endl;
			Ort::Session onnx_session(env, model_path.c_str(), session_options);
			memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

			input_tensor_values.assign(numInputElements, 0.0);
			copy(output_tensor_values.begin(), output_tensor_values.end(), input_tensor_values.begin());
			input_tensor_shape = {1, tr_height, i_width};  // batch size of 1
			input_tensor = Ort::Value::CreateTensor<float>(
				memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());
			input_tensors.clear();
			input_tensors.emplace_back(move(input_tensor));

			// Set output tensor
			output_tensor_values.assign(numOutputElements, 0.0);
			output_tensor_shape = {1, tr_height, o_width};  // batch size of 1
			output_tensor = Ort::Value::CreateTensor<float>(
				memory_info, output_tensor_values.data(), output_tensor_values.size(), output_tensor_shape.data(), output_tensor_shape.size());
			output_tensors.clear();
			output_tensors.emplace_back(move(output_tensor));
			
			// Run inference
			cout << "Running inference..." << endl;
			onnx_session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), output_tensors.data(), 1);
		}		

		/*  Head layer execution */
		input_node_names = { input_node_name_vit_head };
		output_node_names = { output_node_name_vit_head };
		
		numInputElements  = tr_height * o_width;
		numOutputElements = out_numClasses;
		
		model_path = pretrained_dir + "deit_" + back_anchor_name + "_patch16_224_head.onnx";
		cout << "\nLoading ONNX model " + model_path + "..." << endl;
		Ort::Session onnx_session_head(env, model_path.c_str(), session_options);
		memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		copy(output_tensor_values.begin(), output_tensor_values.end(), input_tensor_values.begin());
		input_tensor_shape = {1, tr_height, o_width};  // batch size of 1
		input_tensor = Ort::Value::CreateTensor<float>(
			memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());
		input_tensors.clear();
		input_tensors.emplace_back(move(input_tensor));

		// Set output tensor
		output_tensor_values.assign(numOutputElements, 0.0);
		output_tensor_shape = {1, out_numClasses};  // batch size of 1
		output_tensor = Ort::Value::CreateTensor<float>(
			memory_info, output_tensor_values.data(), output_tensor_values.size(), output_tensor_shape.data(), output_tensor_shape.size());
		output_tensors.clear();
		output_tensors.emplace_back(move(output_tensor));

		// Run inference
		cout << "Running inference..." << endl;
		onnx_session_head.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), 1, output_node_names.data(), output_tensors.data(), 1);

	} catch (const Ort::Exception& e) {
        cerr << "ONNX Runtime Error: " << e.what() << endl;
        return -1;
    } catch (const exception& e) {
        cerr << "Standard exception: " << e.what() << endl;
        return -1;
    } catch (...) {
        cerr << "Unknown exception caught!" << endl;
        return -1;
    }

	/* Processing the result */
	float* floatarr = output_tensors.front().GetTensorMutableData<float>();
	int predicted_class = distance(floatarr, max_element(floatarr, floatarr + out_numClasses));

	if (predicted_class < labels.size()) {
        cout << "Predicted label is: " << labels[predicted_class] << endl;
    } else {
        cout << "Invalid predicted class index!" << endl;
    }

	cout << "Finished!" << endl;

	return 0;
}
