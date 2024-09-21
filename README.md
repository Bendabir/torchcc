# Torch Connected Components

## Compatibility

| Torch Version | CUDA 9.2 | CUDA 10.1 | CUDA 10.2 | CUDA 11.0 | CUDA 11.1 | CUDA 11.3 | CUDA 11.6 | CUDA 11.7 | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 | Min. Python Version | Max. Python Version |
| ------------- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ------------------- | ------------------- |
| 1.7.1         | ☑️       | ☑️        | ✅        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.8.0         | ❌       | ☑️        | ✅        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.8.1         | ❌       | ☑️        | ✅        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.9.0         | ❌       | ❌        | ✅        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.9.1         | ❌       | ❌        | ✅        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.10.0        | ❌       | ❌        | ✅        | ❌        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.10.1        | ❌       | ❌        | ✅        | ❌        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.10.2        | ❌       | ❌        | ✅        | ❌        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.10                |
| 1.11.0        | ❌       | ❌        | ✅        | ❌        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.7                 | 3.10                |
| 1.12.0        | ❌       | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | ❌        | ❌        | ❌        | 3.7                 | 3.10                |
| 1.12.1        | ❌       | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | ❌        | ❌        | ❌        | 3.7                 | 3.10                |
| 1.13.0        | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | ❌        | ❌        | 3.7                 | 3.10                |
| 1.13.1        | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | ❌        | ❌        | 3.7                 | 3.10                |
| 2.0.0         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | ❌        | 3.8                 | 3.11                |
| 2.0.1         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | ❌        | 3.8                 | 3.11                |
| 2.1.0         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.1.1         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.1.2         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.2.0         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.2.1         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.3.0         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.3.1         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.4.0         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ☑️        | ✅        | 🟦        | 3.8                 | 3.12                |
| 2.4.1         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ☑️        | ✅        | 🟦        | 3.8                 | 3.12                |
| 2.5.0         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ☑️        | ✅        | ☑️        | 3.9                 | 3.12                |

- ✅ : Default Stable CUDA
- ☑️ : Stable CUDA
- 🟦 : Experimental CUDA
- ❌ : Unsupported

More details : https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix

## References

This work is based on the following articles.

[1] Allegretti, S., Bolelli, F., & Grana, C. (2020). [Optimized Block-Based Algorithms to Label Connected Components on GPUs](https://federicobolelli.it/pub_files/2019tpds.pdf). IEEE Transactions on Parallel and Distributed Systems, 31(2), 423–438. https://doi.org/10.1109/TPDS.2019.2934683

[2] Grana, C., Bolelli, F., Baraldi, L., & Vezzani, R. (2016). [YACCLAB-Yet Another Connected Components Labeling Benchmark](https://federicobolelli.it/pub_files/2016icpr.pdf). https://doi.org/10.1109/ICPR.2016.7900112

[3] Bolelli, F., Allegretti, S., Lumetti, L., & Grana, C. (2024). [A State-of-the-Art Review with Code about Connected Components Labeling on GPUs](https://federicobolelli.it/pub_files/2024tpds.pdf).
