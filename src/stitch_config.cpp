#include "stitch_config.h"
#include <algorithm>

StitchConfig::StitchConfig(int d, int k, int s) : depth(d), kernel_size(k), stride(s) { // 12, 2, 1
    for (int i = 0; i < depth; ++i) {
        blk_id.push_back(i);
    }

    int i = 0;
    int stitch_id = -1;
    while (i < depth) {
        bool has_new_stitches = false;
        for (int j = i; j < i + kernel_size; ++j) {
            for (int k = i; k < i + kernel_size; ++k) {
                std::pair<int, int> stitch_pair(j, k);
                if (std::find(stitch_cfgs.begin(), stitch_cfgs.end(), stitch_pair) == stitch_cfgs.end()) {
                    has_new_stitches = true;
                    stitch_cfgs.push_back(stitch_pair);
                    stitching_layers_mappings.push_back(stitch_id + 1);
                }
            }
        }
        if (has_new_stitches) {
            stitch_id++;
        }
        i += stride;
    }
    num_stitches = stitch_id + 1;
}

void StitchConfig::printStitchConfig() {
    for (auto& pair : stitch_cfgs) {
        std::cout << "(" << pair.first << ", " << pair.second << ") ";
    }
    std::cout << std::endl;
}
