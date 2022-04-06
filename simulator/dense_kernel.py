from mem import *

class DenseKernel():

    def __init__(self, DEBUG=0):
        self.DEBUG = DEBUG
        return

    def PRINT(self, string):
        if self.DEBUG==1: print(string)
        # print(string)
    # =============================================================== #
    #                       0. SERIAL dense
    # =============================================================== #
    def ser_dense(self, param):
        #         NNNN
        #         NNNN
        #         NNNN
        #         NNNN

        # MMMM    OOOO
        # MMMM    OOOO
        # MMMM    OOOO
        # MMMM    OOOO
        # MMMM    OOOO
        # MMMM    OOOO
        # MMMM    OOOO
        # MMMM    OOOO
        print('code: ser_dense')
        # basic implement
        shape, lhs, rhs, res = param
        M, N, K = shape
        for m in range(M):
            for n in range(N):
                res[m, n] = 0
                for k in range(K):
                    res[m, n] += lhs[m, k] * rhs[n,k]
        return res


    # =============================================================== #
    #                       1. step0_dense
    # =============================================================== #
    def step0_dense(self, param):
        print('code: step0_dense')
        mem = MEM()
        shape, lhs, rhs, res = param
        
        M, N, K = shape
        A = lhs.flatten()
        B = rhs.flatten()
        T = res.flatten()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        shape = max(M*K, N*K, M*N)
        A_shared = mem.new((shape), "zero")
        A_shared_local = mem.new((M*K), "zero")
        B_shared_local = mem.new((N*K), "zero")
        T_shared_local = mem.new((M*N), "zero")
        
        for m in range(M):
            for k in range(K):
                A_shared[m*K + k] = A[m*K + k]
        for m in range(M):
            for k in range(K):
                A_shared_local[m*K + k] = A_shared[m*K + k]
        for n in range(N):
            for k in range(K):
                A_shared[n*K + k] = B[n*K + k]
        for n in range(N):
            for k in range(K):
                B_shared_local[n*K + k] = A_shared[n*K + k]
        
        for m in range(M):
            for n in range(N):
                T_shared_local[m*N + n] = 0
                for k in range(K):
                    T_shared_local[m*N + n] = T_shared_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
        for m in range(M):
            for n in range(N):
                A_shared[m*N + n] = T_shared_local[m*N + n]
        for m in range(M):
            for n in range(N):
                T[m*N + n] = A_shared[m*N + n]                                    
        # import pdb;pdb.set_trace()
        res = np.reshape(T, (M, N))
        return res

    # =============================================================== #
    #                       2. step1_dense
    # =============================================================== #
    def step1_dense_1(self, param):
        print('code: step1_dense_1')
        mem = MEM()
        shape, lhs, rhs, res, grid, block, wmma = param
        M_ori, N_ori, K_ori = shape
        BLOCK_X = grid[0]
        BLOCK_Y = grid[1]
        BLOCK_Z = grid[2]
        # calculate [16,16,2048] in 1 block
        M = M_ori//BLOCK_Y # 128/8 = 16
        N = N_ori//BLOCK_X # 1008/63 = 16
        K = K_ori//BLOCK_Z # 2048/1 = 2048
        shape = max(M*K, N*K, M*N)
        A = lhs.flatten()
        B = rhs.flatten()
        T = res.flatten()
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z):
            for blockIdx_y in range(BLOCK_Y): # M
                for blockIdx_x in range(BLOCK_X): # N
                    A_shared = mem.new((shape), "zero")
                    A_shared_local = mem.new((M*K), "zero")
                    B_shared_local = mem.new((N*K), "zero")
                    T_shared_local = mem.new((M*N), "zero")
                    T_local_shared = mem.new((M*N), "zero")

                    for m in range(M):
                        for k in range(K):
                            A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                    for m in range(M):
                        for k in range(K):
                            A_shared_local[m*K + k] = A_shared[m*K + k]
                    for n in range(N):
                        for k in range(K):
                            A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                    for n in range(N):
                        for k in range(K):
                            B_shared_local[n*K + k] = A_shared[n*K + k]
                    for m in range(M):
                        for n in range(N):
                            T_shared_local[m*N + n] = 0
                            for k in range(K):
                                T_shared_local[m*N + n] = T_shared_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                    for m in range(M):
                        for n in range(N):
                            T_local_shared[m*N + n] = T_shared_local[m*N + n]
                    for m in range(M):
                        for n in range(N):
                            T[blockIdx_y*M*N_ori + m*N_ori + blockIdx_x*N + n] = T_local_shared[m*N + n]
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       2. step1_dense
    # =============================================================== #
    def step1_dense(self, param):
        print('code: step1_dense')
        mem = MEM()
        shape, lhs, rhs, res, grid, block, wmma = param
        M_ori, N_ori, K_ori = shape
        wmma_m = wmma[0] # 16
        wmma_n = wmma[1] # 16
        wmma_k = wmma[2] # 16
        # calculate [64,16,2048] in 1 block
        M = grid[1]*block[1]*wmma_m # 1*4*16 = 64
        N = grid[2]*block[2]*wmma_n # 1*1*16 = 16
        K = K_ori
        BLOCK_X = N_ori//N # 1008/16 = 63
        BLOCK_Y = M_ori//M # 128/64 = 2
        BLOCK_Z = grid[2] # 1
        shape = max(M*K, N*K, M*N)
        A = lhs.flatten()
        B = rhs.flatten()
        T = res.flatten()
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z):
            for blockIdx_y in range(BLOCK_Y): # M
                for blockIdx_x in range(BLOCK_X): # N
                    A_shared = mem.new((shape), "zero")
                    A_shared_local = mem.new((M*K), "zero")
                    B_shared_local = mem.new((N*K), "zero")
                    T_shared_local = mem.new((M*N), "zero")
                    T_local_shared = mem.new((M*N), "zero")

                    for m in range(M):
                        for k in range(K):
                            A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                    for m in range(M):
                        for k in range(K):
                            A_shared_local[m*K + k] = A_shared[m*K + k]
                    for n in range(N):
                        for k in range(K):
                            A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                    for n in range(N):
                        for k in range(K):
                            B_shared_local[n*K + k] = A_shared[n*K + k]
                    for m in range(M):
                        for n in range(N):
                            T_shared_local[m*N + n] = 0
                            for k in range(K):
                                T_shared_local[m*N + n] = T_shared_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                    for m in range(M):
                        for n in range(N):
                            T_local_shared[m*N + n] = T_shared_local[m*N + n]
                    for m in range(M):
                        for n in range(N):
                            T[blockIdx_y*M*N_ori + m*N_ori + blockIdx_x*N + n] = T_local_shared[m*N + n]
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       3. step2_dense
    # =============================================================== #
    def step2_dense(self, param):
        print('code: step2_dense')
        mem = MEM()
        shape, lhs, rhs, res, grid, block = param
        
        M_ori, N_ori, K_ori = shape
        BLOCK_X = grid[0]
        BLOCK_Y = grid[1]
        BLOCK_Z = grid[2]
        THREAD_X = block[0]
        THREAD_Y = block[1]
        THREAD_Z = block[2]
        M = M_ori//BLOCK_Y
        N = N_ori//BLOCK_X
        K = K_ori//BLOCK_Z
        shape = max(M*K, N*K, M*N)
        A = lhs.flatten()
        B = rhs.flatten()
        T = res.flatten()
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z):
            for blockIdx_y in range(BLOCK_Y): # M
                for blockIdx_x in range(BLOCK_X): # N
                    A_shared = mem.new((shape), "zero")
                    A_shared_local = mem.new((M*K), "zero")
                    B_shared_local = mem.new((N*K), "zero")
                    T_shared_local = mem.new((M*N), "zero")
                    T_local_shared = mem.new((M*N), "zero")

                    for m in range(M):
                        for k in range(K):
                            A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                    for m in range(M):
                        for k in range(K):
                            A_shared_local[m*K + k] = A_shared[m*K + k]
                    for n in range(N):
                        for k in range(K):
                            A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                    for n in range(N):
                        for k in range(K):
                            B_shared_local[n*K + k] = A_shared[n*K + k]
                    for m in range(M):
                        for n in range(N):
                            T_shared_local[m*N + n] = 0
                            for k in range(K):
                                T_shared_local[m*N + n] = T_shared_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                    for m in range(M):
                        for n in range(N):
                            T_local_shared[m*N + n] = T_shared_local[m*N + n]
                    # import pdb;pdb.set_trace()
                    # for m in range(M):
                    #     for n in range(N):
                    #         T[blockIdx_y*M*N_ori + m*N_ori + blockIdx_x*N + n] = T_local_shared[m*N + n]                                
                    # fused
                    # m_n_fused = M*N
                    # m = m_n_fused//N
                    # n = m_n_fused%N
                    for m_n_fused in range(M*N):
                        T[blockIdx_y*M*N_ori + (m_n_fused//N)*N_ori + blockIdx_x*N + (m_n_fused%N)] = T_local_shared[(m_n_fused//N)*N + (m_n_fused%N)]
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       3. step3_dense
    # =============================================================== #
    def step3_dense(self, param):
        print('code: step3_dense')
        mem = MEM()
        shape, lhs, rhs, res, grid, block = param
        
        M_ori, N_ori, K_ori = shape
        BLOCK_X = grid[0]
        BLOCK_Y = grid[1]
        BLOCK_Z = grid[2]
        THREAD_X = block[0]
        THREAD_Y = block[1]
        THREAD_Z = block[2]
        M = M_ori//BLOCK_Y
        N = N_ori//BLOCK_X
        K = K_ori//BLOCK_Z
        shape = max(M*K, N*K, M*N)

        A = lhs.flatten()
        B = rhs.flatten()
        T = res.flatten()
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z):
            for blockIdx_y in range(BLOCK_Y): # M
                for blockIdx_x in range(BLOCK_X): # N
                    A_shared = mem.new((shape), "zero")
                    A_shared_local = mem.new((M*K), "zero")
                    B_shared_local = mem.new((N*K), "zero")
                    T_shared_local = mem.new((M*N), "zero")
                    T_local_shared = mem.new((M*N), "zero")

                    for m in range(M):
                        for k in range(K):
                            A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                    for m in range(M):
                        for k in range(K):
                            A_shared_local[m*K + k] = A_shared[m*K + k]
                    for n in range(N):
                        for k in range(K):
                            A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                    for n in range(N):
                        for k in range(K):
                            B_shared_local[n*K + k] = A_shared[n*K + k]
                    for m in range(M):
                        for n in range(N):
                            T_shared_local[m*N + n] = 0
                            for k in range(K):
                                T_shared_local[m*N + n] = T_shared_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                    for m in range(M):
                        for n in range(N):
                            T_local_shared[m*N + n] = T_shared_local[m*N + n]
                    # # fused
                    # # m_n_fused = M*N
                    # # m = m_n_fused//N
                    # # n = m_n_fused%N
                    # for m_n_fused in range(M*N):
                    #     T[blockIdx_y*M*N_ori + (m_n_fused//N)*N_ori + blockIdx_x*N + (m_n_fused%N)] = T_local_shared[(m_n_fused//N)*N + (m_n_fused%N)]

                    # C_o_o_o = (M*N)//THREAD_X
                    # for m_n_fused_o in range(C_o_o_o):
                    #     for m_n_fused_i in range(THREAD_X):
                    #         # T[blockIdx_y*M*N_ori + ((m_n_fused_o*THREAD_X + m_n_fused_i)//N)*N_ori + blockIdx_x*N + ((m_n_fused_o*THREAD_X + m_n_fused_i)%N)] = T_local_shared[((m_n_fused_o*THREAD_X + m_n_fused_i)//N)*N + ((m_n_fused_o*THREAD_X + m_n_fused_i)%N)]
                    #         T[blockIdx_y*M*N_ori + (m_n_fused_o*THREAD_X//N)*N_ori + (m_n_fused_i//N)*N_ori + blockIdx_x*N + m_n_fused_o*THREAD_X%N + m_n_fused_i%N] = T_local_shared[((m_n_fused_o*THREAD_X + m_n_fused_i)//N)*N + ((m_n_fused_o*THREAD_X + m_n_fused_i)%N)]

                    # C_o_o_o = (M*N)//THREAD_X//THREAD_Y
                    # for m_n_fused_o_o in range(C_o_o_o):
                    #     for m_n_fused_o_i in range(THREAD_Y):
                    #         for m_n_fused_i in range(THREAD_X):
                    #             T[blockIdx_y*M*N_ori + ((m_n_fused_o_o*THREAD_Y+m_n_fused_o_i)*THREAD_X//N)*N_ori + (m_n_fused_i//N)*N_ori + blockIdx_x*N + (m_n_fused_o_o*THREAD_Y+m_n_fused_o_i)*THREAD_X%N + m_n_fused_i%N] = T_local_shared[(((m_n_fused_o_o*THREAD_Y+m_n_fused_o_i)*THREAD_X + m_n_fused_i)//N)*N + (((m_n_fused_o_o*THREAD_Y+m_n_fused_o_i)*THREAD_X + m_n_fused_i)%N)]

                    C_o_o_o = (M*N)//THREAD_X//THREAD_Y//THREAD_Z
                    for m_n_fused_o_o_o in range(C_o_o_o):
                        for m_n_fused_o_o_i in range(THREAD_Z):
                            for m_n_fused_o_i in range(THREAD_Y):
                                for m_n_fused_i in range(THREAD_X):
                                    T[blockIdx_y*M*N_ori + (((m_n_fused_o_o_o*THREAD_Z + m_n_fused_o_o_i)*THREAD_Y+m_n_fused_o_i)*THREAD_X//N)*N_ori + (m_n_fused_i//N)*N_ori + blockIdx_x*N + ((m_n_fused_o_o_o*THREAD_Z + m_n_fused_o_o_i)*THREAD_Y+m_n_fused_o_i)*THREAD_X%N + m_n_fused_i%N] = T_local_shared[((((m_n_fused_o_o_o*THREAD_Z + m_n_fused_o_o_i)*THREAD_Y+m_n_fused_o_i)*THREAD_X + m_n_fused_i)//N)*N + ((((m_n_fused_o_o_o*THREAD_Z + m_n_fused_o_o_i)*THREAD_Y+m_n_fused_o_i)*THREAD_X + m_n_fused_i)%N)]
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       4. step4_dense
    # =============================================================== #
    def step4_dense(self, param):
        print('code: step4_dense')
        mem = MEM()
        shape, lhs, rhs, res, grid, block = param
        
        M_ori, N_ori, K_ori = shape
        BLOCK_X = grid[0]
        BLOCK_Y = grid[1]
        BLOCK_Z = grid[2]
        THREAD_X = block[0]
        THREAD_Y = block[1]
        THREAD_Z = block[2]
        M = M_ori//BLOCK_Y
        N = N_ori//BLOCK_X
        K = K_ori//BLOCK_Z
        shape = max(M*K, N*K, M*N)

        A = lhs.flatten()
        B = rhs.flatten()
        T = res.flatten()
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z):
            for blockIdx_y in range(BLOCK_Y): # M
                for blockIdx_x in range(BLOCK_X): # N
                    for threadIdx_x in range(THREAD_X):
                        for threadIdx_y in range(THREAD_Y):
                            for threadIdx_z in range(THREAD_Z):
                                A_shared = mem.new((shape), "zero")
                                A_shared_local = mem.new((M*K), "zero")
                                B_shared_local = mem.new((N*K), "zero")
                                T_shared_local = mem.new((M*N), "zero")
                                T_local_shared = mem.new((M*N), "zero")

                                for m in range(M):
                                    for k in range(K):
                                        A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                                for m in range(M):
                                    for k in range(K):
                                        A_shared_local[m*K + k] = A_shared[m*K + k]
                                for n in range(N):
                                    for k in range(K):
                                        A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                                for n in range(N):
                                    for k in range(K):
                                        B_shared_local[n*K + k] = A_shared[n*K + k]
                                for m in range(M):
                                    for n in range(N):
                                        T_shared_local[m*N + n] = 0
                                        for k in range(K):
                                            T_shared_local[m*N + n] = T_shared_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                                for m in range(M):
                                    for n in range(N):
                                        T_local_shared[m*N + n] = T_shared_local[m*N + n]

                                C_o_o_o = (M*N)//THREAD_X//THREAD_Y//THREAD_Z
                                for m_n_fused_o_o_o in range(C_o_o_o):
                                    T[blockIdx_y*M*N_ori + (((m_n_fused_o_o_o*THREAD_Z + threadIdx_z)*THREAD_Y+threadIdx_y)*THREAD_X//N)*N_ori + (threadIdx_x//N)*N_ori + blockIdx_x*N + ((m_n_fused_o_o_o*THREAD_Z + threadIdx_z)*THREAD_Y+threadIdx_y)*THREAD_X%N + threadIdx_x%N] = T_local_shared[((((m_n_fused_o_o_o*THREAD_Z + threadIdx_z)*THREAD_Y+threadIdx_y)*THREAD_X + threadIdx_x)//N)*N + ((((m_n_fused_o_o_o*THREAD_Z + threadIdx_z)*THREAD_Y+threadIdx_y)*THREAD_X + threadIdx_x)%N)]
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       5. step5_dense
    # =============================================================== #
    def step5_dense(self, param):
        print('code: step5_dense')
        mem = MEM()
        shape, lhs, rhs, res, grid, block, wmma = param
        
        M_ori, N_ori, K_ori = shape
        BLOCK_X = grid[0]
        BLOCK_Y = grid[1]
        BLOCK_Z = grid[2]
        THREAD_X = block[0]
        THREAD_Y = block[1]
        THREAD_Z = block[2]
        wmma_m = wmma[0]
        wmma_n = wmma[1]
        wmma_k = wmma[2]
        M = M_ori//BLOCK_Y
        N = N_ori//BLOCK_X
        K = K_ori//BLOCK_Z
        shape = max(M*K, N*K, M*N)

        A = lhs.flatten()
        B = rhs.flatten()
        T = res.flatten()
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z):
            for blockIdx_y in range(BLOCK_Y): # M
                for blockIdx_x in range(BLOCK_X): # N
                    for threadIdx_x in range(THREAD_X):
                        for threadIdx_y in range(THREAD_Y):
                            for threadIdx_z in range(THREAD_Z):
                                A_shared = mem.new((shape), "zero")
                                A_shared_local = mem.new((M*K), "zero")
                                B_shared_local = mem.new((N*K), "zero")
                                T_shared_local = mem.new((M*N), "zero")
                                T_local_shared = mem.new((M*N), "zero")
                                # lhs HBM to L2
                                for m in range(M):
                                    for k in range(K):
                                        A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                                # lhs L2 to L1
                                for m in range(M):
                                    for k in range(K):
                                        A_shared_local[m*K + k] = A_shared[m*K + k]
                                # rhs HBM to L2
                                for n in range(N):
                                    for k in range(K):
                                        A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                                # rhs L2 to L1
                                for n in range(N):
                                    for k in range(K):
                                        B_shared_local[n*K + k] = A_shared[n*K + k]
                                # output calculate
                                for m in range(M):
                                    for n in range(N):
                                        T_shared_local[m*N + n] = 0
                                        for k in range(K):
                                            T_shared_local[m*N + n] = T_shared_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                                # # output L1 to L2
                                # for m_o in range(M//wmma_m):
                                #     for m_i in range(wmma_m):
                                #         for n_o in range(N//wmma_n):
                                #             for n_i in range(wmma_n):
                                #                 T_local_shared[(m_o*wmma_m + m_i)*N + (n_o*wmma_n + n_i)] = T_shared_local[(m_o*wmma_m + m_i)*N + (n_o*wmma_n + n_i)]
                                # output L1 to L2
                                for m_o_o in range(M//wmma_m//BLOCK_Y):
                                    for m_o_i in range(BLOCK_Y):
                                        for m_i in range(wmma_m):
                                            for n_o_o in range(N//wmma_n//BLOCK_Z):
                                                for n_o_i in range(BLOCK_Z):
                                                    for n_i in range(wmma_n):
                                                        T_local_shared[((m_o_o*BLOCK_Y + m_o_i)*wmma_m + m_i)*N + ((n_o_o*BLOCK_Z + n_o_i)*wmma_n + n_i)] = T_shared_local[((m_o_o*BLOCK_Y + m_o_i)*wmma_m + m_i)*N + ((n_o_o*BLOCK_Z + n_o_i)*wmma_n + n_i)]
                                # output L2 to HBM
                                C_o_o_o = (M*N)//THREAD_X//THREAD_Y//THREAD_Z
                                for m_n_fused_o_o_o in range(C_o_o_o):
                                    T[blockIdx_y*M*N_ori + (((m_n_fused_o_o_o*THREAD_Z + threadIdx_z)*THREAD_Y+threadIdx_y)*THREAD_X//N)*N_ori + (threadIdx_x//N)*N_ori + blockIdx_x*N + ((m_n_fused_o_o_o*THREAD_Z + threadIdx_z)*THREAD_Y+threadIdx_y)*THREAD_X%N + threadIdx_x%N] = T_local_shared[((((m_n_fused_o_o_o*THREAD_Z + threadIdx_z)*THREAD_Y+threadIdx_y)*THREAD_X + threadIdx_x)//N)*N + ((((m_n_fused_o_o_o*THREAD_Z + threadIdx_z)*THREAD_Y+threadIdx_y)*THREAD_X + threadIdx_x)%N)]
        res = np.reshape(T, (M_ori, N_ori))
        return res

