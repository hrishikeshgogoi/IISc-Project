OS: Ubuntu 24.10

Kernel version: 6.11.0-21-generic

oneAPI base toolkit: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

Issues faced: sycl-ls not showing GPU

Fix: Update drivers from https://dgpu-docs.intel.com/driver/client/overview.html

If issue persists,

forceprobe xe driver

check apt sources, update apt

Recommended resources before getting started:(DPC++ book) https://indianinstituteofscience.sharepoint.com/:b:/r/sites/distributed-systems/Shared%20Documents/Heterogeneous-ML/Resources/DPC++%20book%20-%202nd%20ed.pdf?csf=1&web=1&e=Ssydc5

TILE_SIZE * TILE_SIZE is the tiling size for shared memory blocks of matrices A and B and also the work-group size for matrix C

m_size * m_size is the size of the matrices A, B, and C(before padding)

Padding is used to overcome the issue where sycl does not allow TILE_SIZE that is not a multiple of m_size in GPU

local_accessor is used to allocate shared local memory(L1 cache) for faster access to tiles of A and B within each work-group

Looping over tiles(k) enables a single work-group to compute a tile of C by iteratively loading chunks of A and B into local memory

The first barrier ensures that all elements of tiles A and B are loaded into shared memory before any computation begins

The second barrier ensures all work-items have finished using the current tile before it is overwritten in the next iteration

After the loop completes, the final result is written to the corresponding element in matrix C (global memory)

Testing:

1st test: Initial testing for simple matrix multiplication(different matrix shapes)

cpu device:(Intel(R) Core(TM) Ultra 7 258V)

Problem size: c(2000,8000) = a(2000,4000) * b(4000,8000) Average: 7521.588 ms

Problem size: c(4000,4000) = a(4000,4000) * b(4000,4000) Average: 4301.173 ms

Problem size: c(8000,2000) = a(8000,4000) * b(4000,2000) Average: 3649.741 ms

Problem size: c(8000,4000) = a(8000,2000) * b(2000,4000) Average: 3680.396 ms

Problem size: c(4000,8000) = a(4000,2000) * b(2000,8000) Average: 5043.515 ms

gpu device:(Intel(R) Arc(TM) Graphics)

Problem size: c(2000,8000) = a(2000,4000) * b(4000,8000) Average: 2572.852 ms

Problem size: c(4000,4000) = a(4000,4000) * b(4000,4000) Average: 2452.428 ms

Problem size: c(8000,2000) = a(8000,4000) * b(4000,2000) Average: 1513.887 ms

Problem size: c(8000,4000) = a(8000,2000) * b(2000,4000) Average: 2141.063 ms

Problem size: c(4000,8000) = a(4000,2000) * b(2000,8000) Average: 2608.229 ms

Initial testing for simple matrix multiplication(different matrix sizes)

cpu device:(Intel(R) Core(TM) Ultra 7 258V)

Problem size: c(250,1000) = a(250,500) * b(500,1000) Average: 2.532338 ms

Problem size: c(500,2000) = a(500,1000) * b(1000,2000) Average: 25.51864 ms

Problem size: c(1000,4000) = a(1000,2000) * b(2000,4000) Average: 452.0946 ms

Problem size: c(2000,8000) = a(2000,4000) * b(4000,8000) Average: 7521.588 ms

Problem size: c(8000,8000) = a(8000,8000) * b(8000,8000) Average: 56200.837 ms

gpu device:(Intel(R) Arc(TM) Graphics)

Problem size: c(250,1000) = a(250,500) * b(500,1000) Average: 8.70561 ms

Problem size: c(500,2000) = a(500,1000) * b(1000,2000) Average: 38.66686 ms

Problem size: c(1000,4000) = a(1000,2000) * b(2000,4000) Average: 266.0434 ms

Problem size: c(2000,8000) = a(2000,4000) * b(4000,8000) Average: 2572.8522 ms

Problem size: c(8000,8000) = a(8000,8000) * b(8000,8000) Average: 20751.957 ms

2nd test: Problem size: c(2000,8000) = a(2000,4000) * b(4000,8000)

Initial testing for cpu(Intel(R) Core(TM) Ultra 7 258V):

local_range(2, 2):(~29.7s tt) Average: 28626.897 ms

local_range(2, 8):(~5.2s tt) Average: 4979.568 ms

local_range(4, 4):(~30s tt) Average: 29916.11 ms

local_range(2, 16):(~5.5s tt) Average: 5243.862 ms

local_range(8, 2):(~29s tt) Average: 28486.132

local_range(4, 8):(~4.5s tt) Average: 4170.435 ms

local_range(8, 4):(~30s tt)(higher total clock time for entire program execution than local_range(16, 8)) Average: 29085.89 ms

local_range(8, 8):(~4.2s tt) Average: 4042.656 ms

local_range(16, 8):(~4.3s tt) Average: 4030.052 ms

local_range(16, 16):(~4.2s tt) Average: 3895.438 ms

Initial testing for gpu(Intel(R) Arc(TM) Graphics):

local_range(2, 2):(~19.4s tt) Average: 19459.731 ms

local_range(2, 8):(~3.8s tt) Average: 3494.843 ms

local_range(4, 4):(~3.7s tt) Average: 3354.398 ms

local_range(2, 16):(~2.3s tt) Average: 1937.946 ms

local_range(8, 2):(~5.12s tt) Average: 4807.937 ms

local_range(4, 8):(~2.15s tt) Average: 1794.840 ms

local_range(8, 4):(~2.1s tt) Average: 1755.211 ms

local_range(8, 8):(~2.1s tt) Average: 1752.041 ms

local_range(16, 8):(~2.9s tt) Average: 2485.551 ms

local_range(16, 16):(~2.9s tt) Average: 2529.747 ms

Problem size: c(1000,4000) = a(1000,2000) * b(2000,4000)

Initial testing for cpu(Intel(R) Core(TM) Ultra 7 258V):

local_range(2, 2):(~1.77s tt) Average: 1544.134 ms

local_range(2, 8):(~0.47s tt) Average: 252.7264 ms

local_range(4, 4):(~1.72s tt) Average: 1502.442 ms

local_range(2, 16):(~0.47s tt) Average: 248.956 ms

local_range(8, 2):(~1.70s tt) Average: 1487.241 ms

local_range(4, 8):(~0.43s tt) Average: 221.0652 ms

local_range(8, 4):(~1.69s tt) Average: 1480.034 ms

local_range(8, 8):(~0.43s tt) Average: 210.1942 ms

Initial testing for gpu(Intel(R) Arc(TM) Graphics):

local_range(2, 2):(~2.23s tt) Average: 2059.227 ms

local_range(2, 8):(~0.58s tt) Average: 433.0812 ms

local_range(4, 4):(~0.57s tt) Average: 426.6053 ms

local_range(2, 16):(~0.37s tt) Average: 226.693 ms

local_range(8, 2):(~0.69s tt) Average: 534.3057 ms

local_range(4, 8):(~0.38s tt) Average: 223.6579 ms

local_range(8, 4):(~0.39s tt) Average: 236.7524 ms

local_range(8, 8):(~0.37s tt) Average: 221.6401 ms

test01o.cpp : naive matrix multiplication from oneAPI base toolkit

test01p.cpp : matrix multiplication with tiling and padding

test16p.cpp : added an option to choose subgroup size

The XMX engine can be utilized through the oneMKL library

oneAPI-samples -> Libraries -> oneMKL -> matrix_mul_mkl

To adjust the matrix sizes or floating-point precision, modify the GNUmakefile
