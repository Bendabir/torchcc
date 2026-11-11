// Copyright (c) 2024 Bendabir.
#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torchcc.h>

// Empty module so Python can import the shared library (it registers the ops on load).
extern "C"
{
    PyObject *PyInit__C(void)
    {
        static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
        return PyModule_Create(&module_def);
    }
}

STABLE_TORCH_LIBRARY(torchcc, m)
{
    m.def("ccl2d(Tensor x, int connectivity) -> Tensor");
    m.def("ccl3d(Tensor x, int connectivity) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(torchcc, CUDA, m)
{
    m.impl("ccl2d", TORCH_BOX(&torchcc::ccl2d));
    m.impl("ccl3d", TORCH_BOX(&torchcc::ccl3d));
}
