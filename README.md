# GPGPU_Project: Implement CUDA version of an image processing algorithm

:computer: CUDA, C++, CMake

* CPU and GPU (Cuda) implementation of binary opening/closing

* Library Needed: Opencv

* Run CPU: src/ Cmake ./gpgpu path_src

* Run GPU: cuda/ Makefile ./main path_src name_dst

* Test images: build/

Result: (1000x1000) image with 5x5 kernel

![one](https://user-images.githubusercontent.com/17318529/80267999-20a06e00-8672-11ea-8dc3-8b734c3b5a23.PNG)
![two](https://user-images.githubusercontent.com/17318529/80268004-2a29d600-8672-11ea-8f3c-f5ec0a97832a.PNG)
![three](https://user-images.githubusercontent.com/17318529/80268006-2b5b0300-8672-11ea-9dd4-fe4c6d9ad5e7.PNG)

Benchmark:

Kernel size: 5x5 - i7-8750H 6 coeurs/12 threads - Fréquence 2,20 GHz - 4,10 GHz et un GPU Nvidia 1050Ti 4Go 768 CUDA Cores

| Center-aligned | Center-aligned | Center-aligned | Center-aligned |
| :---:         |     :---:      |          :---: | :---: |
|   128x128  | 32 ms | 0.06 ms | 0.056 ms |
|   256x256 |  37 ms | 0.099 ms |0.1 ms |
| 600x600 |83 ms |0.359 ms | 0.386 ms |
|  800x800 |140 ms | 1.074 ms | 0.989 ms |
|  1000x1000 | 184 ms | 1.383 ms | 1.11 ms|
| 4k | 1278 ms |32.705 ms | 17.26 ms |
|  8k | 4848 ms | 123.230 ms | 62.111 ms|




