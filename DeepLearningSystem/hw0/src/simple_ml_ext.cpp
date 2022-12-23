#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

unsigned char* one_hot_encoding(const unsigned char *y, int m, int k) {
    unsigned char* Y = new unsigned char[m * k];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            if (y[i] == j)
                Y[i*k + j] = 1;
            else
                Y[i*k + j] = 0;
        }
    }
    return Y;
}

float* matmul(float *x, float *y, int m, int d, int n) {
    float *z = new float[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int ij = i * n + j;
            z[ij] = 0;
            for (int k = 0; k < d; k++) {
                int ik = i * d + k;
                int kj = k * n + j;
                z[ij] += x[ik] * y[kj];
            }
        }
    }

    return z;
}

float* slice(const float *x, int start_row, int end_row, int d) {
    int n = (end_row - start_row) * d;
    int start = start_row * d;
    float* z = new float[n];
    for (int i = 0; i < n; i++) {
        z[i] = x[start + i];
    }

    return z;
}

unsigned char* slice(const unsigned char *x, int start_row, int end_row, int d) {
    int n = (end_row - start_row) * d;
    int start = start_row * d;
    unsigned char* z = new unsigned char[n];
    for (int i = 0; i < n; i++) {
        z[i] = x[start + i];
    }

    return z;
}

float* softmax(float *x, int m, int n) {
    float* res = new float[m * n];
    for (int i = 0; i < m; i++) {
        float s = 0;
        for (int j = 0; j < n; j++) {
            int index = i * n + j;
            res[index] = exp(x[index]);
            s += res[index];
        }
        for (int j = 0; j < n; j++) {
            int index = i * n + j;
            res[index] /= s;
        }
    }

    return res;
}

float* transpose(float *x, int m, int n) {
    float *y = new float[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            y[j * m + i] = x[i * n + j];
        }
    }

    return y;
}

void minus(float *x, float *y, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int index = i * n + j;
            x[index] -= y[index];
        }
    }
}

void mul(float *x, float a, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int index = i * n + j;
            x[index] *= a;
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int step = m / batch;
    // (gdb) x/10c 0x1a99a20
    unsigned char *Y = one_hot_encoding(y, m, k);
    for (int i=0; i<step; i++) {
        int start = i * batch;
        int end = std::min(start + batch, m);
        if (start == end) {
            break;
        }
        int b = end - start;
        float *x = slice(X, start, end, n);
        unsigned char *y = slice(Y, start, end, k);
        float *score = matmul(x, theta, b, n, k);
        float *z = softmax(score, b, k);
        //  loss = z - Iy
        for (int i=0; i<b; i++) {
            for (int j=0; j<(int)k; j++) {
                z[i*k + j] -= y[i*k + j];
            }
        }
        // grad = x.T @ (z - Iy)
        float *x_transpose = transpose(x, b, n);
        float *grad = matmul(x_transpose, z, n, b, k);
        // update
        mul(grad, 1.0 * lr / batch, n, k);
        // shape: n, k
        minus(theta, grad, n, k);
    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
