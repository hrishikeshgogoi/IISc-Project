#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

using namespace sycl;
using namespace std;

constexpr int TILE_SIZE = 32;
constexpr int m_size = 4096;
constexpr int M = m_size, N = m_size, P = m_size;

// Pad dimensions to multiples of TILE_SIZE
constexpr int M_pad = ((M + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
constexpr int N_pad = ((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
constexpr int P_pad = ((P + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

int main() {
    queue q(gpu_selector_v);
    cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
    cout << "Computing: (" << M << "x" << P << ") = (" << M << "x" << N << ") * (" << N << "x" << P << ")\n";

    float (*c_back)[P_pad] = new float[M_pad][P_pad]();

    try {
        buffer<float, 2> a_buf(range(M_pad, N_pad));
        buffer<float, 2> b_buf(range(N_pad, P_pad));
        buffer<float, 2> c_buf(reinterpret_cast<float*>(c_back), range(M_pad, P_pad));

        // Initialize A
        q.submit([&](handler &h) {
            accessor a(a_buf, h, write_only, no_init);
            h.parallel_for(range(M_pad, N_pad), [=](id<2> idx) {
                int i = idx[0], j = idx[1];
                a[i][j] = (i < M && j < N) ? (i * N + j + 1.0f) : 0.0f;
            });
        });

        // Initialize B
        q.submit([&](handler &h) {
            accessor b(b_buf, h, write_only, no_init);
            h.parallel_for(range(N_pad, P_pad), [=](id<2> idx) {
                int i = idx[0], j = idx[1];
                b[i][j] = (i < N && j < P) ? (i * P + j + 1.0f) : 0.0f;
            });
        });

        // Prepare exec range
        range<2> global_range(M_pad, P_pad / 4);
        range<2> local_range(TILE_SIZE, TILE_SIZE / 4);
        nd_range<2> exec_range(global_range, local_range);

        auto t1 = std::chrono::high_resolution_clock::now();

        q.submit([&](handler &h) {
            accessor a(a_buf, h, read_only);
            accessor b(b_buf, h, read_only);
            accessor c(c_buf, h, write_only, no_init);

            h.parallel_for(exec_range, [=](nd_item<2> item) {
                int row = item.get_global_id(0);
                int col_base = item.get_global_id(1) * 4;

                sycl::vec<float, 4> sum(0.0f);

                for (int k = 0; k < N_pad; ++k) {
                    float a_val = a[row][k];

                    sycl::vec<float, 4> b_vec(
                        b[k][col_base + 0],
                        b[k][col_base + 1],
                        b[k][col_base + 2],
                        b[k][col_base + 3]);

                    sum += a_val * b_vec;
                }

                if (row < M && col_base < P) {
                    if (col_base + 0 < P) c[row][col_base + 0] = sum.s0();
                    if (col_base + 1 < P) c[row][col_base + 1] = sum.s1();
                    if (col_base + 2 < P) c[row][col_base + 2] = sum.s2();
                    if (col_base + 3 < P) c[row][col_base + 3] = sum.s3();
                }
            });
        });

        q.wait();
        auto t2 = std::chrono::high_resolution_clock::now();

        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        cout << "Execution Time: " << elapsed_ms << " ms\n";

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
        cerr << "SYCL Exception: " << e.what() << "\n";
        return 1;
    }

    delete[] c_back;
    return 0;
}