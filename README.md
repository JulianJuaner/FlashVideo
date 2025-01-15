# FlashVideo: My experiments on efficient video generation.

## Overview Log
### Multi-GPU Teacache Inference
This can achieve a nearly 1.5X inference speed-up conparing with vanilla 8-GPU inference (450 -> 298s)
```bash
cd HunyuanVideo
bash ../teacache_apply/run_sample_video_multigpu_tea.sh
```

### Profile
One denoising step with single card. Attention occupies 77.8% GPU time. 
| Name | Self CUDA | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
|------|-----------|-------------|------------|---------------|------------|
| FlashAttnVarlenFunc | 46.083s | 77.87% | 46.084s | 768.059ms | 60 |
| aten::linear | 9.354ms | 0.02% | 12.281s | 17.851ms | 688 |
| aten::addmm | 6.047s | 10.22% | 6.047s | 17.579ms | 344 |
| aten::copy_ | 2.638s | 4.46% | 2.638s | 2.334ms | 1130 |
| aten::to | 10.324ms | 0.02% | 2.110s | 1.381ms | 1528 |
| aten::_to_copy | 15.636ms | 0.03% | 2.079s | 1.942ms | 1081 |
| aten::layer_norm | 2.472ms | 0.00% | 1.626s | 6.454ms | 252 |
| aten::cat | 1.384s | 2.20% | 1.389s | 4.305ms | 304 |
| aten::mul | 1.187s | 2.01% | 1.187s | 1.460ms | 813 |
| aten::add | 754.235ms | 1.27% | 754.235ms | 1.155ms | 653 |
| aten::native_layer_norm | 171.323ms | 0.29% | 732.628ms | 5.815ms | 126 |
| aten::clone | 558.000us | 0.00% | 561.617ms | 12.196ms | 46 |
| aten::contiguous | 310.000us | 0.00% | 559.287ms | 13.644ms | 41 |
| aten::stack | 1.928ms | 0.00% | 549.823ms | 4.575ms | 120 |
| aten::type_as | 2.069ms | 0.00% | 442.590ms | 1.581ms | 280 |
| aten::pow | 200.642ms | 0.34% | 402.084ms | 1.257ms | 320 |
| aten::gelu | 373.665ms | 0.63% | 373.665ms | 4.671ms | 80 |
| aten::neg | 158.945ms | 0.27% | 158.945ms | 1.325ms | 120 |
| aten::mean | 130.268ms | 0.22% | 131.996ms | 804.854us | 164 |
| aten::slice | 10.287ms | 0.02% | 14.533ms | 10.846us | 1348 |