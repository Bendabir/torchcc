// Copyright (c) 2024 Bendabir.
#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torchcc.h>

// Empty module so Python can import the shared library (it registers the ops on load).
extern "C"
{
    PyObject *PyInit__cuda(void)
    {
        static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_cuda", nullptr, -1, nullptr};
        return PyModule_Create(&module_def);
    }
}

STABLE_TORCH_LIBRARY(torchcc, m)
{
    // Compute Connected Components Labeling of a 2D image (or a batch of 2D images).
    m.def("ccl2d(Tensor x, int connectivity) -> Tensor");
    // Compute Connected Components Labeling of a 3D volume (or a batch of 3D volumes).
    m.def("ccl3d(Tensor x, int connectivity) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(torchcc, CUDA, m)
{
    m.impl("ccl2d", TORCH_BOX(&torchcc::ccl2d));
    m.impl("ccl3d", TORCH_BOX(&torchcc::ccl3d));
}
