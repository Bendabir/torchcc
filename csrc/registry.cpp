#include <torch/extension.h>
#include <torchcc.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cc2d", &torchcc::cc2d, "Compute Connected Components of a 2D image (or a batch of 2D images).");
    m.def("cc3d", &torchcc::cc3d, "Compute Connected Components of a 3D volume (or a batch of 3D volumes).");
}
