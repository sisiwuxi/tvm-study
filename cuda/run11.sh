"""
./build/11_simple_sum_matrix2D/simple_sum_matrix 64 1
./build/11_simple_sum_matrix2D/simple_sum_matrix 64 2
./build/11_simple_sum_matrix2D/simple_sum_matrix 64 4
./build/11_simple_sum_matrix2D/simple_sum_matrix 64 8
./build/11_simple_sum_matrix2D/simple_sum_matrix 64 16
./build/11_simple_sum_matrix2D/simple_sum_matrix 128 1
./build/11_simple_sum_matrix2D/simple_sum_matrix 128 2
./build/11_simple_sum_matrix2D/simple_sum_matrix 128 4
./build/11_simple_sum_matrix2D/simple_sum_matrix 128 8
./build/11_simple_sum_matrix2D/simple_sum_matrix 128 16
./build/11_simple_sum_matrix2D/simple_sum_matrix 256 1
./build/11_simple_sum_matrix2D/simple_sum_matrix 256 2
./build/11_simple_sum_matrix2D/simple_sum_matrix 256 4
./build/11_simple_sum_matrix2D/simple_sum_matrix 256 8
./build/11_simple_sum_matrix2D/simple_sum_matrix 256 16
"""

"""
nvprof --metrics achieved_occupancy,gld_throughput,gld_efficiency ./build/11_simple_sum_matrix2D/simple_sum_matrix 32 32
nvprof --metrics achieved_occupancy,gld_throughput,gld_efficiency ./build/11_simple_sum_matrix2D/simple_sum_matrix 32 16
nvprof --metrics achieved_occupancy,gld_throughput,gld_efficiency ./build/11_simple_sum_matrix2D/simple_sum_matrix 16 32
nvprof --metrics achieved_occupancy,gld_throughput,gld_efficiency ./build/11_simple_sum_matrix2D/simple_sum_matrix 16 16
nvprof --metrics achieved_occupancy,gld_throughput,gld_efficiency ./build/11_simple_sum_matrix2D/simple_sum_matrix 16 8
nvprof --metrics achieved_occupancy,gld_throughput,gld_efficiency ./build/11_simple_sum_matrix2D/simple_sum_matrix 8 16
"""
"""
./build/11_simple_sum_matrix2D/simple_sum_matrix 32 32
./build/11_simple_sum_matrix2D/simple_sum_matrix 32 32
./build/11_simple_sum_matrix2D/simple_sum_matrix 32 16
./build/11_simple_sum_matrix2D/simple_sum_matrix 16 32
./build/11_simple_sum_matrix2D/simple_sum_matrix 16 16
./build/11_simple_sum_matrix2D/simple_sum_matrix 16 8
./build/11_simple_sum_matrix2D/simple_sum_matrix 8 16
"""
