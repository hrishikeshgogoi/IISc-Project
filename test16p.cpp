#include <chrono>
#include <sycl/sycl.hpp>
#include <iostream>
//#include <ittnotify.h>

using namespace std;
using namespace sycl;
using namespace sycl::ext::oneapi;

constexpr int TILE_SIZE = 32; // Work-group size = TILE_SIZE * TILE_SIZE

constexpr int m_size = 4096;
constexpr int M = m_size;
constexpr int N = m_size;
constexpr int P = m_size;

// Chnage padded sizes to be multiples of TILE_SIZE
constexpr int M_pad = ((M + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
constexpr int N_pad = ((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
constexpr int P_pad = ((P + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

/*
// Test tile no.
constexpr int log_threads = M_pad * N_pad; // No. pf threads to log
constexpr int tile_count = (M_pad / TILE_SIZE) * (P_pad / TILE_SIZE); // Total number of tiles
std::vector<int> tile_log(log_threads * tile_count, -1); // One slot per thread per tile
buffer<int, 1> tile_buf(tile_log.data(), range<1>(log_threads * tile_count));
*/

int main() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    queue q(gpu_selector_v);
    cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;
    cout << "Problem size:" << ((M < M_pad || N < N_pad) ? "(padded) c(" : " c(") << M_pad << "," << P_pad << ") = a(" << M_pad << "," << N_pad
         << ") * b(" << N_pad << "," << P_pad << ")" << std::endl;

    float (*c_back)[P_pad] = new float[M_pad][P_pad]();
  
    try {
        buffer<float, 2> a_buf(range(M_pad, N_pad));
        buffer<float, 2> b_buf(range(N_pad, P_pad));
        buffer<float, 2> c_buf(reinterpret_cast<float *>(c_back), range(M_pad, P_pad));

        
        // Test sg 001
        size_t subgroup_result = 0;
        buffer<size_t, 1> sg_buf(&subgroup_result, range<1>(1));
        
        
        // Initialize Matrix A with padding
        q.submit([&](handler &h) {
            accessor a(a_buf, h, write_only, no_init);
            h.parallel_for(range(M_pad, N_pad), [=](id<2> index) {
                int i = index[0], j = index[1];
                a[i][j] = (i < M && j < N) ? (i * N + j + 1.0f) : 0.0f;  // Padding with 0
                //a[i][j] = (i == j) ? 1.0f : 0.0f;  // Sparse test
                //a[i][j] = (i == 0 && j ==0 ) ? 1.0f : 0.0f;  // Sparse test2
            });
        });
        
        /*
        //Ignore verification only
        q.wait();
        host_accessor a_host(a_buf, read_only);
        std::cout << "Matrix A:\n";
        for (int i = 0; i < M_pad; i++) {
            for (int j = 0; j < N_pad; j++) {
                std::cout << a_host[i][j] << " ";
            }
            std::cout << std::endl;
        }
        */
        
        // Initialize Matrix B with padding
        q.submit([&](handler &h) {
            accessor b(b_buf, h, write_only, no_init);
            h.parallel_for(range(N_pad, P_pad), [=](id<2> index) {
                int i = index[0], j = index[1];
                b[i][j] = (i < N && j < P) ? (i * P + j + 1.0f) : 0.0f;  // Padding with 0
                //b[i][j] = (i == j) ? 1.0f : 0.0f;
                //b[i][j] = (i == 0 && j == 0) ? 1.0f : 0.0f;
            });
        });
        
        /*
        //Ignore verification only
        q.wait();
        host_accessor b_host(b_buf, read_only);
        std::cout << "Matrix B:\n";
        for (int i = 0; i < N_pad; i++) {
            for (int j = 0; j < P_pad; j++) {
                std::cout << b_host[i][j] << " ";
            }
            std::cout << std::endl;
        }
        */
        
        auto max_wg_size = q.get_device().get_info<info::device::max_work_group_size>();
        cout << "Max work-group size: " << max_wg_size << std::endl << std::flush;
        auto max_c_units = q.get_device().get_info<info::device::max_compute_units>();
        cout << "Max compute units " << max_c_units << std::endl << std::flush;
        auto max_sgs = q.get_device().get_info<info::device::sub_group_sizes>();
        std::cout << "Max sub-group sizes: ";
        for (auto size : max_sgs) std::cout << size ;
        std::cout << std::endl;
        auto slm_size = q.get_device().get_info<sycl::info::device::local_mem_size>();
        std::cout << "Local Memory Size : " << slm_size << "\n" ;

        range<2> global_range(M_pad, P_pad);
        range<2> local_range(TILE_SIZE, TILE_SIZE);
        nd_range<2> exec_range(global_range, local_range);

        auto t1 = high_resolution_clock::now();

        // VTune start marker
        //__itt_domain* domain = __itt_domain_create("MatrixMultiply");
        //__itt_string_handle* handle = __itt_string_handle_create("SYCL Kernel Execution");
        //__itt_task_begin(domain, __itt_null, __itt_null, handle);

        q.submit([&](handler &h) {
            accessor a(a_buf, h, read_only);
            accessor b(b_buf, h, read_only);
            accessor c(c_buf, h, write_only, no_init);

            
            // Test sg 010
            accessor sg_acc(sg_buf, h, write_only, no_init);
            

            /*
            // Test tile no.
            accessor tile_acc(tile_buf, h, write_only);
            */

            local_accessor<float, 2> a_tile(range(TILE_SIZE, TILE_SIZE), h);
            local_accessor<float, 2> b_tile(range(TILE_SIZE, TILE_SIZE), h);

            int width_a = a_buf.get_range()[1];

            h.parallel_for<class MatMulKernel>(exec_range,[=](nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    int row = item.get_global_id(0);
                    int col = item.get_global_id(1);

                    
                    // Test sg 011(write subgroup size from one thread only)
                    if (row == 0 && col == 0) {
                        auto sg = item.get_sub_group();
                        sg_acc[0] = sg.get_local_range().size();
                    }
                    

                    int local_row = item.get_local_id(0);
                    int local_col = item.get_local_id(1);

                    float sum = 0.0f;

                    for (int tile = 0; tile < width_a / TILE_SIZE; ++tile) {

                        /*
                        // Test tile no.
                        // Log the tile index processed by this thread
                        int global_index = item.get_global_linear_id();
                        int tile_log_index = global_index * tile_count + tile; // Calculate index
                        if (global_index < log_threads) {
                            tile_acc[tile_log_index] = tile;  // Log the tile index for the current thread
                        }
                        */

                        a_tile[local_row][local_col] = a[row][tile * TILE_SIZE + local_col];
                        b_tile[local_row][local_col] = b[tile * TILE_SIZE + local_row][col];

                        item.barrier(access::fence_space::local_space);

                        for (int k = 0; k < TILE_SIZE; ++k) {
                            sum += a_tile[local_row][k] * b_tile[k][local_col];
                        }

                        item.barrier(access::fence_space::local_space);
                    }

                    // Store result only for valid space (ignores padding region)
                    if (row < M && col < P) {
                        c[row][col] = sum;
                    }
                }
            );

        });

        q.wait();

        auto t2 = high_resolution_clock::now();
        auto ms_int = duration_cast<milliseconds>(t2 - t1);
        duration<double, std::milli> ms_double = t2 - t1;
        std::cout << "Execution Time: " << ms_double.count() << " ms" << std::endl;
        
        
        // Test sg 100
        host_accessor sg_result(sg_buf, read_only);
        std::cout << "Sub-group size used: " << sg_result[0] << std::endl;
        
        
        /*
        // Test tile no.
        host_accessor tile_host(tile_buf, read_only);
        std::cout << "Tile indices logged per thread (first few threads):\n";
        for (int i = 0; i < log_threads; ++i) {
            std::cout << "Thread " << i << " processed tiles: ";
            for (int j = 0; j < tile_count; ++j) {
                std::cout << tile_host[i * tile_count + j] << " ";
            }
            std::cout << std::endl;
        }
        */

        // VTune end marker
        //__itt_task_end(domain);
        
        /*
        //Ignore verification only
        host_accessor c_host(c_buf, read_only);
        std::cout << "Matrix C:\n";
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < P; j++) {
                std::cout << c_host[i][j] << " ";
            }
            std::cout << std::endl;
        }
        */

    } catch (sycl::exception const &e) {
        cout << "An exception is caught while multiplying matrices." << std::endl;
        terminate();
    }

    delete[] c_back;
    return 0;
}