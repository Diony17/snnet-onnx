#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <vector>
#include <cstdint>

/* Input image shape and number of classes */
constexpr int64_t in_numChannels = 3;
constexpr int64_t in_width = 224;
constexpr int64_t in_height = 224;
constexpr int64_t out_numClasses = 1000;

/* Input image */
const char* image_name = "n01443537_goldfish.JPEG";
const char* label_name = "imagenet_classes.txt";

/* Transformer layer shape */
constexpr int64_t tr_height  = 197;
constexpr int64_t tr_width_t = 192; // tiny
constexpr int64_t tr_width_s = 384; // small
constexpr int64_t tr_width_b = 768; // base

constexpr int64_t maxNumInputElements = tr_height * tr_width_b; // for memory assign
constexpr int64_t maxNumOutputElements = tr_height * tr_width_b; // for memory assign

/* Transformer layer shape */
std::vector<const char*> vit_types = {"tiny", "small", "base"};
const char* input_node_name_vit_head = "input";
const char* output_node_name_vit_head = "18";
const char* input_node_name_vit_embed = "input.1";
const char* output_node_name_vit_embed = "input"; 
const char* input_node_name_vit_layers = "input.1";
const char* output_node_name_vit_layers = "100";
const char* input_node_name_stitch_layers = "onnx::MatMul_0";
const char* output_node_name_stitch_layers = "5";



#endif // CONSTANTS_H