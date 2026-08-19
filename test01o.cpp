#include <chrono>
#include <sycl/sycl.hpp>
#include <iostream>
#include <limits>

#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

constexpr int m_size = 8000;
constexpr int M = 2000;
constexpr int N = m_size;
constexpr int P = 4000;

int main() {

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
  // Host memory buffer that device will write data back before destruction.
  float(*c_back)[P] = new float[M][P];

  // Intialize c_back
  for (int i = 0; i < M; i++)
    for (int j = 0; j < P; j++) c_back[i][j] = 0.0f;

  // Initialize the device queue with the default selector. The device queue is
  // used to enqueue kernels. It encapsulates all states needed for execution.
  try {
    queue q(cpu_selector_v);

    cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    // Create 2D buffers for matrices, buffer c is bound with host memory c_back

    buffer<float, 2> a_buf(range(M, N));
    buffer<float, 2> b_buf(range(N, P));
    buffer c_buf(reinterpret_cast<float *>(c_back), range(M, P));

    cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
         << ") * b(" << N << "," << P << ")\n";

    // Using three command groups to illustrate execution order. The use of
    // first two command groups for initializing matrices is not the most
    // efficient way. It just demonstrates the implicit multiple command group
    // execution ordering.

    // Submit command group to queue to initialize matrix a
    q.submit([&](auto &h) {
      // Get write only access to the buffer on a device.
      accessor a(a_buf, h, write_only);

      // Execute kernel.
      h.parallel_for(range(M, N), [=](auto index) {
        // Each element of matrix a is 1.
        a[index] = index[0] * N + index[1] + 1.0f;
      });
    });

    // Submit command group to queue to initialize matrix b
    q.submit([&](auto &h) {
      // Get write only access to the buffer on a device
      accessor b(b_buf, h, write_only);

      // Execute kernel.
      h.parallel_for(range(N, P), [=](auto index) {
        // Each column of b is the sequence 1,2,...,N
        b[index] = index[0] * P + index[1] + 1.0f;
      });
    });

    auto t1 = high_resolution_clock::now();
    
    // Submit command group to queue to multiply matrices: c = a * b
    q.submit([&](auto &h) {
      // Read from a and b, write to c
      accessor a(a_buf, h, read_only);
      accessor b(b_buf, h, read_only);
      accessor c(c_buf, h, write_only);

      int width_a = a_buf.get_range()[1];

      // Execute kernel.
      h.parallel_for(range(M, P), [=](auto index) {
        // Get global position in Y direction.
        int row = index[0];
        // Get global position in X direction.
        int col = index[1];

        float sum = 0.0f;

        // Compute the result of one element of c
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
    std::cout << "Execution Time: " << ms_double.count() << " ms" << std::endl;


  } catch (sycl::exception const &e) {
    cout << "An exception is caught while multiplying matrices.\n";
    terminate();
  }

  delete[] c_back;

  return 0;
}