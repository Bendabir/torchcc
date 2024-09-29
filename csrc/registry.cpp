// Copyright (c) 2024 Bendabir.
#include <torch/extension.h>
#include <torchcc.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ccl2d", &torchcc::ccl2d, "Compute Connected Components Labeling of a 2D image (or a batch of 2D images).");
    m.def("ccl3d", &torchcc::ccl3d, "Compute Connected Components Labeling of a 3D volume (or a batch of 3D volumes).");
}
