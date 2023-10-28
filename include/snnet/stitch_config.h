#ifndef STITCHCONFIG_H
#define STITCHCONFIG_H

#include <iostream>
#include <vector>
#include <set>
#include <utility>

class StitchConfig {
private:
    int depth;
    int kernel_size;
    int stride;
    int num_stitches;
    std::vector<int> blk_id;
    std::vector<std::pair<int, int>> stitch_cfgs;
    std::vector<int> stitching_layers_mappings;

public:
    StitchConfig(int d, int k, int s);
    void printStitchConfig();

    int getNumStitches() const {
        return num_stitches;
    }

    std::pair<int, int> getStitchConfig(int s_config_id) const {
        return stitch_cfgs[s_config_id];
    }

    std::vector<int> getLayerMappings() const {
        return stitching_layers_mappings;
    }
};

#endif // STITCHCONFIG_H