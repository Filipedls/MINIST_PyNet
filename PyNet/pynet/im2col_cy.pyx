# HOW TO COMPILE:
# python setup_cython.py build_ext --inplace

# FROM -> https://github.com/huyouare/CS231n/tree/master/assignment2/cs231n

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def im2col_cython(np.ndarray[DTYPE_t, ndim=4] x, int field_height,
                  int field_width, int stride, int yH, int yW):
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]

    cdef np.ndarray[DTYPE_t, ndim=2] cols = np.empty(
            ( N * yH * yW, C * field_height * field_width))

    # Moving the inner loop to a C function with no bounds checking works, but does
    # not seem to help performance in any measurable way.

    im2col_cython_inner(cols, x, N, C, H, W, yH, yW,
                        field_height, field_width, stride)
    # cdef int yy, xx, y0, y1, x0, x1, i
    # cdef int ii, jj, c, row, col

    # for c in range(C):
    #     for ii in range(field_height):
    #         for jj in range(field_width):
    #             row = c * field_width * field_height + ii * field_height + jj
    #             for yy in range(yH):
    #                 for xx in range(yW):
    #                     for i in range(N):
    #                         col = yy * yW * N + xx * N + i
    #                         cols[row, col] = x_padded[i, c, stride * yy + ii, stride * xx + jj]
    
    return cols


@cython.boundscheck(False)
cdef int im2col_cython_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=4] x,
                             int N, int C, int H, int W, int yH, int yW,
                             int field_height, int field_width, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    row = 0
    for i in range(N):
        for yy in range(0,yH*stride, stride):
            for xx in range(0,yW*stride, stride):
                col = 0
                for c in range(C):
                    for ii in range(yy,yy+field_height):
                        for jj in range(xx,xx+field_width):
                            #print i, c, ii, jj, row, col
                            cols[row, col] = x[i, c, ii, jj]
                            col += 1

                row += 1


def col2im_cython(np.ndarray[DTYPE_t, ndim=2] cols, int N, int C, int H, int W,
                  int field_height, int field_width, int stride, int yH, int yW):
    
    cdef np.ndarray x = np.zeros((N, C, H, W), dtype=DTYPE)

    # Moving the inner loop to a C-function with no bounds checking improves
    # performance quite a bit for col2im.
    col2im_cython_inner(cols, x, N, C, H, W, yH, yW, 
                        field_height, field_width, stride)

    return x


@cython.boundscheck(False)
cdef int col2im_cython_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=4] x,
                             int N, int C, int H, int W, int yH, int yW,
                             int field_height, int field_width, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col


    row = 0
    for i in range(N):
        for yy in range(0,yH*stride, stride):
            for xx in range(0,yW*stride, stride):
                col = 0
                for c in range(C):
                    for ii in range(yy,yy+field_height):
                        for jj in range(xx,xx+field_width):
                            #print i, c, ii, jj, row, col
                            x[i, c, ii, jj] += cols[row, col]
                            col += 1

                row += 1

