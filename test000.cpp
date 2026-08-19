#include <chrono>
#include <sycl/sycl.hpp>
#include <iostream>
#include <limits>
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

constexpr int m_size = 2000 * 8;
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;

int main() {
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  queue q(gpu_selector_v);  
  cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
  cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
         << ") * b(" << N << "," << P << ")\n";

    float(*c_back)[P] = new float[M][P];

    for (int i = 0; i < M; i++)
      for (int j = 0; j < P; j++) c_back[i][j] = 0.0f;

    try {
      buffer<float, 2> a_buf(range(M, N));
      buffer<float, 2> b_buf(range(N, P));
      buffer c_buf(reinterpret_cast<float *>(c_back), range(M, P));

      q.submit([&](auto &h) {
        accessor a(a_buf, h, write_only);
        h.parallel_for(range(M, N), [=](auto index) {
          a[index] = 1.0f;
        });
      });

      q.submit([&](auto &h) {
        accessor b(b_buf, h, write_only);
        h.parallel_for(range(N, P), [=](auto index) {
          b[index] = index[0] + 1.0f;
        });
      });

      auto t1 = high_resolution_clock::now();

      q.submit([&](auto &h) {
        accessor a(a_buf, h, read_only);
        accessor b(b_buf, h, read_only);
        accessor c(c_buf, h, write_only);

        int width_a = a_buf.get_range()[1];
        
        h.parallel_for(range(M, P), [=](auto index) {
          int row = index[0];
          int col = index[1];

          float sum = 0.0f;

          for (int i = 0; i < width_a; i++) {
            sum += a[row][i] * b[i][col];
          }

          c[index] = sum;
        });
      });

      q.wait();
      auto t2 = high_resolution_clock::now();
      auto ms_int = duration_cast<milliseconds>(t2 - t1);
      duration<double, std::milli> ms_double = t2 - t1;
      std::cout << ms_int.count() << "ms\n";
      std::cout << ms_double.count() << "ms\n";

    } catch (sycl::exception const &e) {
      cout << "An exception is caught while multiplying matrices.\n";
      terminate();
    }
    
    delete[] c_back;

  return 0;
}