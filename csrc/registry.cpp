#include <torch/extension.h>
#include <torchcc.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("identity", &torchcc::identity); }
