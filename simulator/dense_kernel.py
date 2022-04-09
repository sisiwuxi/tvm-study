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
    #                       0. ori_dense
    # =============================================================== #
    def ori_dense(self, param):
        print('code: ori_dense')
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
        T_local = mem.new((M*N), "zero")
        
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
                T_local[m*N + n] = 0
                for k in range(K):
                    T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
        for m in range(M):
            for n in range(N):
                A_shared[m*N + n] = T_local[m*N + n]
        for m in range(M):
            for n in range(N):
                T[m*N + n] = A_shared[m*N + n]                                    
        # import pdb;pdb.set_trace()
        res = np.reshape(T, (M, N))
        return res

    # =============================================================== #
    #                       0. step0_dense_1
    # =============================================================== #
    def step0_dense_1(self, param):
        print('code: step0_dense_1')
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
                    T_local = mem.new((M*N), "zero")
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
                            T_local[m*N + n] = 0
                            for k in range(K):
                                T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                    for m in range(M):
                        for n in range(N):
                            T_local_shared[m*N + n] = T_local[m*N + n]
                    for m in range(M):
                        for n in range(N):
                            T[blockIdx_y*M*N_ori + m*N_ori + blockIdx_x*N + n] = T_local_shared[m*N + n]
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       0. step0_dense
    # =============================================================== #
    def step0_dense(self, param):
        print('code: step0_dense')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma = param
        M_ori, N_ori, K_ori = shape
        wmma_m = wmma[0] # 16
        wmma_n = wmma[1] # 16
        wmma_k = wmma[2] # 16
        # block=(64,4,7)
        # calculate [64,112,2048] in 1 thread
        M = block_loop[1]*block[1]*wmma_m # 1*4*16 = 64
        N = block_loop[2]*block[2]*wmma_n # 1*7*16 = 112
        K = K_ori
        # grid=(BLOCK_X,BLOCK_X,BLOCK_Z)=(9,2,1)
        BLOCK_X = N_ori//N # 1008/112 = 9
        BLOCK_Y = M_ori//M # 128/64 = 2
        BLOCK_Z = block_loop[2] # 1
        print("grid(%d,%d,%d), block_loop(%d,%d,%d), block(%d,%d,%d)"%(BLOCK_X,BLOCK_Y,BLOCK_Z,block_loop[0],block_loop[1],block_loop[2],block[0],block[1],block[2]))
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
                    T_local = mem.new((M*N), "zero")
                    T_local_shared = mem.new((M*N), "zero")
                    # lhs HBM -> L2
                    for m in range(M):
                        for k in range(K):
                            A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                    # lhs L2 -> L1
                    for m in range(M):
                        for k in range(K):
                            A_shared_local[m*K + k] = A_shared[m*K + k]
                    # rhs HBM -> L2
                    for n in range(N):
                        for k in range(K):
                            A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                    # rhs L2 -> L1
                    for n in range(N):
                        for k in range(K):
                            B_shared_local[n*K + k] = A_shared[n*K + k]
                    # calculate
                    for m in range(M):
                        for n in range(N):
                            T_local[m*N + n] = 0
                            for k in range(K):
                                T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                    # out L1 -> L2
                    for m in range(M):
                        for n in range(N):
                            T_local_shared[m*N + n] = T_local[m*N + n]
                    # out L2 -> HBM
                    for m in range(M):
                        for n in range(N):
                            T[blockIdx_y*M*N_ori + m*N_ori + blockIdx_x*N + n] = T_local_shared[m*N + n]
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       1. step1_0_fuse_dense
    # =============================================================== #
    def step1_0_fuse_dense(self, param):
        print('code: step1_0_fuse_dense')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma = param
        M_ori, N_ori, K_ori = shape
        wmma_m = wmma[0] # 16
        wmma_n = wmma[1] # 16
        wmma_k = wmma[2] # 16
        # block=(64,4,7)
        # calculate [64,112,2048] in 1 thread
        M = block_loop[1]*block[1]*wmma_m # 1*4*16 = 64
        N = block_loop[2]*block[2]*wmma_n # 1*7*16 = 112
        K = K_ori
        # grid=(BLOCK_X,BLOCK_X,BLOCK_Z)=(9,2,1)
        BLOCK_X = N_ori//N # 1008/112 = 9
        BLOCK_Y = M_ori//M # 128/64 = 2
        BLOCK_Z = block_loop[2] # 1
        print("grid(%d,%d,%d), block_loop(%d,%d,%d), block(%d,%d,%d)"%(BLOCK_X,BLOCK_Y,BLOCK_Z,block_loop[0],block_loop[1],block_loop[2],block[0],block[1],block[2]))
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
                    T_local = mem.new((M*N), "zero")
                    T_local_shared = mem.new((M*N), "zero")
                    # lhs HBM -> L2
                    for m in range(M):
                        for k in range(K):
                            A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                    # lhs L2 -> L1
                    for m in range(M):
                        for k in range(K):
                            A_shared_local[m*K + k] = A_shared[m*K + k]
                    # rhs HBM -> L2
                    for n in range(N):
                        for k in range(K):
                            A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                    # rhs L2 -> L1
                    for n in range(N):
                        for k in range(K):
                            B_shared_local[n*K + k] = A_shared[n*K + k]
                    # calculate
                    for m in range(M):
                        for n in range(N):
                            T_local[m*N + n] = 0
                            for k in range(K):
                                T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                    # out L1 -> L2
                    for m in range(M):
                        for n in range(N):
                            T_local_shared[m*N + n] = T_local[m*N + n]
                    # out L2 -> HBM                               
                    # fused
                    # m_n_fused = M*N
                    # m = m_n_fused//N
                    # n = m_n_fused%N
                    for m_n_fused in range(M*N):
                        left = blockIdx_y*M*N_ori + (m_n_fused//N)*N_ori + blockIdx_x*N + (m_n_fused%N)
                        right = (m_n_fused//N)*N + (m_n_fused%N)
                        T[left] = T_local_shared[right]
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       1. step1_0
    # =============================================================== #
    def step1_0(self, param):
        print('code: step1_0')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec = param
        M_ori, N_ori, K_ori = shape
        wmma_m = wmma[0] # 16
        wmma_n = wmma[1] # 16
        wmma_k = wmma[2] # 16
        # block=(64,4,7)
        # calculate [64,112,2048] in 1 thread
        M = block_loop[1]*block[1]*wmma_m # 1*4*16 = 64
        N = block_loop[2]*block[2]*wmma_n # 1*7*16 = 112
        K = K_ori
        # grid=(BLOCK_X,BLOCK_X,BLOCK_Z)=(9,2,1)
        BLOCK_X = N_ori//N # 1008/112 = 9
        BLOCK_Y = M_ori//M # 128/64 = 2
        BLOCK_Z = block_loop[2] # 1
        print("grid(%d,%d,%d), block_loop(%d,%d,%d), block(%d,%d,%d)"%(BLOCK_X,BLOCK_Y,BLOCK_Z,block_loop[0],block_loop[1],block_loop[2],block[0],block[1],block[2]))
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
                    T_local = mem.new((M*N), "zero")
                    T_local_shared = mem.new((M*N), "zero")
                    # lhs HBM -> L2
                    for m in range(M):
                        for k in range(K):
                            A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                    # lhs L2 -> L1
                    for m in range(M):
                        for k in range(K):
                            A_shared_local[m*K + k] = A_shared[m*K + k]
                    # rhs HBM -> L2
                    for n in range(N):
                        for k in range(K):
                            A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                    # rhs L2 -> L1
                    for n in range(N):
                        for k in range(K):
                            B_shared_local[n*K + k] = A_shared[n*K + k]
                    # calculate
                    for m in range(M):
                        for n in range(N):
                            T_local[m*N + n] = 0
                            for k in range(K):
                                T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                    # out L1 -> L2
                    for m in range(M):
                        for n in range(N):
                            T_local_shared[m*N + n] = T_local[m*N + n]
                    # out L2 -> HBM
                    # # fused
                    # # m_n_fused = M*N
                    # # m = m_n_fused//N
                    # # n = m_n_fused%N
                    # for m_n_fused in range(M*N):
                    #     left = blockIdx_y*M*N_ori + (m_n_fused//N)*N_ori + blockIdx_x*N + (m_n_fused%N)
                    #     right = (m_n_fused//N)*N + (m_n_fused%N)
                    #     T[left] = T_local_shared[right]

                    # C_o_o_o = (M*N)//block[0]
                    # for m_n_fused_o in range(C_o_o_o):
                    #     for m_n_fused_i in range(block[0]):
                    #         left = blockIdx_y*M*N_ori + ((m_n_fused_o*block[0] + m_n_fused_i)//N)*N_ori + blockIdx_x*N + ((m_n_fused_o*block[0] + m_n_fused_i)%N)
                    #         right = ((m_n_fused_o*block[0] + m_n_fused_i)//N)*N + ((m_n_fused_o*block[0] + m_n_fused_i)%N)
                    #         T[left] = T_local_shared[right]

                    # C_o_o_o = (M*N)//block[0]//block[1]
                    # for m_n_fused_o_o in range(C_o_o_o):
                    #     for m_n_fused_o_i in range(block[1]):
                    #         for m_n_fused_i in range(block[0]):
                    #             left = blockIdx_y*M*N_ori + (((m_n_fused_o_o*block[1] + m_n_fused_o_i)*block[0] + m_n_fused_i)//N)*N_ori + blockIdx_x*N + (((m_n_fused_o_o*block[1] + m_n_fused_o_i)*block[0] + m_n_fused_i)%N)
                    #             right = (((m_n_fused_o_o*block[1] + m_n_fused_o_i)*block[0] + m_n_fused_i)//N)*N + (((m_n_fused_o_o*block[1] + m_n_fused_o_i)*block[0] + m_n_fused_i)%N)
                    #             T[left] = T_local_shared[right]

                    C_o_o_o = (M*N)//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o in range(C_o_o_o):
                        for m_n_fused_o_o_i in range(block[2]):
                            for m_n_fused_o_i in range(block[1]):
                                for m_n_fused_i in range(block[0]):
                                    left = blockIdx_y*M*N_ori + ((((m_n_fused_o_o_o*block[2] + m_n_fused_o_o_i)*block[1] + m_n_fused_o_i)*block[0] + m_n_fused_i)//N)*N_ori + blockIdx_x*N + ((((m_n_fused_o_o_o*block[2] + m_n_fused_o_o_i)*block[1] + m_n_fused_o_i)*block[0] + m_n_fused_i)%N)
                                    right = ((((m_n_fused_o_o_o*block[2] + m_n_fused_o_o_i)*block[1] + m_n_fused_o_i)*block[0] + m_n_fused_i)//N)*N + ((((m_n_fused_o_o_o*block[2] + m_n_fused_o_o_i)*block[1] + m_n_fused_o_i)*block[0] + m_n_fused_i)%N)
                                    T[left] = T_local_shared[right]
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       1. step1_1
    # =============================================================== #
    def step1_1(self, param):
        print('code: step1_1')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma = param
        M_ori, N_ori, K_ori = shape
        wmma_m = wmma[0] # 16
        wmma_n = wmma[1] # 16
        wmma_k = wmma[2] # 16
        # block=(64,4,7)
        # calculate [64,112,2048] in 1 thread
        M = block_loop[1]*block[1]*wmma_m # 1*4*16 = 64
        N = block_loop[2]*block[2]*wmma_n # 1*7*16 = 112
        K = K_ori
        # grid=(BLOCK_X,BLOCK_X,BLOCK_Z)=(9,2,1)
        BLOCK_X = N_ori//N # 1008/112 = 9
        BLOCK_Y = M_ori//M # 128/64 = 2
        BLOCK_Z = block_loop[2] # 1
        print("grid(%d,%d,%d), block_loop(%d,%d,%d), block(%d,%d,%d)"%(BLOCK_X,BLOCK_Y,BLOCK_Z,block_loop[0],block_loop[1],block_loop[2],block[0],block[1],block[2]))
        shape = max(M*K, N*K, M*N)

        A = lhs.flatten()
        B = rhs.flatten()
        T = res.flatten()
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                for blockIdx_x in range(BLOCK_X): # 9
                    A_shared = mem.new((shape), "zero")
                    A_shared_local = mem.new((M*K), "zero")
                    B_shared_local = mem.new((N*K), "zero")
                    T_local = mem.new((M*N), "zero")
                    T_local_shared = mem.new((M*N), "zero")
                    # lhs HBM -> L2
                    for m in range(M):
                        for k in range(K):
                            A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                    # lhs L2 -> L1
                    for m in range(M):
                        for k in range(K):
                            A_shared_local[m*K + k] = A_shared[m*K + k]
                    # rhs HBM -> L2
                    for n in range(N):
                        for k in range(K):
                            A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                    # rhs L2 -> L1
                    for n in range(N):
                        for k in range(K):
                            B_shared_local[n*K + k] = A_shared[n*K + k]
                    # calculate
                    for m in range(M):
                        for n in range(N):
                            T_local[m*N + n] = 0
                            for k in range(K):
                                T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                    # out L1 -> L2
                    for m in range(M):
                        for n in range(N):
                            T_local_shared[m*N + n] = T_local[m*N + n]
                    # out L2 -> HBM
                    for threadIdx_z in range(block[2]): # 7
                        for threadIdx_y in range(block[1]): # 4
                            for threadIdx_x in range(block[0]): # 64
                                C_o_o_o = (M*N)//block[0]//block[1]//block[2]
                                for m_n_fused_o_o_o in range(C_o_o_o):
                                    left = blockIdx_y*M*N_ori + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)//N)*N_ori + blockIdx_x*N + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)%N)
                                    right = ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)//N)*N + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)%N)
                                    T[left] = T_local_shared[right]                                
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       1. step1_2
    # =============================================================== #
    def step1_2(self, param):
        print('code: step1_1')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma = param
        M_ori, N_ori, K_ori = shape
        wmma_m = wmma[0] # 16
        wmma_n = wmma[1] # 16
        wmma_k = wmma[2] # 16
        # block=(64,4,7)
        # calculate [64,112,2048] in 1 thread
        M = block_loop[1]*block[1]*wmma_m # 1*4*16 = 64
        N = block_loop[2]*block[2]*wmma_n # 1*7*16 = 112
        K = K_ori
        # grid=(BLOCK_X,BLOCK_X,BLOCK_Z)=(9,2,1)
        BLOCK_X = N_ori//N # 1008/112 = 9
        BLOCK_Y = M_ori//M # 128/64 = 2
        BLOCK_Z = block_loop[2] # 1
        print("grid(%d,%d,%d), block_loop(%d,%d,%d), block(%d,%d,%d)"%(BLOCK_X,BLOCK_Y,BLOCK_Z,block_loop[0],block_loop[1],block_loop[2],block[0],block[1],block[2]))
        shape = max(M*K, N*K, M*N)

        A = lhs.flatten()
        B = rhs.flatten()
        T = res.flatten()
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                for blockIdx_x in range(BLOCK_X): # 9
                    A_shared = mem.new((shape), "zero")
                    A_shared_local = mem.new((M*K), "zero")
                    B_shared_local = mem.new((N*K), "zero")
                    T_local = mem.new((M*N), "zero")
                    T_local_shared = mem.new((M*N), "zero")
                    # lhs HBM -> L2
                    for m in range(M):
                        for k in range(K):
                            A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                    # lhs L2 -> L1
                    for m in range(M):
                        for k in range(K):
                            A_shared_local[m*K + k] = A_shared[m*K + k]
                    # rhs HBM -> L2
                    for n in range(N):
                        for k in range(K):
                            A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                    # rhs L2 -> L1
                    for n in range(N):
                        for k in range(K):
                            B_shared_local[n*K + k] = A_shared[n*K + k]
                    # calculate
                    for m in range(M):
                        for n in range(N):
                            T_local[m*N + n] = 0
                            for k in range(K):
                                T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                    # out L1 -> L2
                    for m in range(M):
                        for n in range(N):
                            T_local_shared[m*N + n] = T_local[m*N + n]
                    # out L2 -> HBM
                    for threadIdx_z in range(block[2]): # 7
                        for threadIdx_y in range(block[1]): # 4
                            for threadIdx_x in range(block[0]): # 64
                                C_o_o_o = (M*N)//block[0]//block[1]//block[2]
                                for m_n_fused_o_o_o in range(C_o_o_o):
                                    left = blockIdx_y*M*N_ori + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)//N)*N_ori + blockIdx_x*N + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)%N)
                                    right = ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)//N)*N + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)%N)
                                    T[left] = T_local_shared[right]                                
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       1. step1
    # =============================================================== #
    def step1(self, param):
        print('code: step1_1')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma = param
        M_ori, N_ori, K_ori = shape
        wmma_m = wmma[0] # 16
        wmma_n = wmma[1] # 16
        wmma_k = wmma[2] # 16
        # block=(64,4,7)
        # calculate [64,112,2048] in 1 thread
        M = block_loop[1]*block[1]*wmma_m # 1*4*16 = 64
        N = block_loop[2]*block[2]*wmma_n # 1*7*16 = 112
        K = K_ori
        # grid=(BLOCK_X,BLOCK_X,BLOCK_Z)=(9,2,1)
        BLOCK_X = N_ori//N # 1008/112 = 9
        BLOCK_Y = M_ori//M # 128/64 = 2
        BLOCK_Z = block_loop[2] # 1
        print("grid(%d,%d,%d), block_loop(%d,%d,%d), block(%d,%d,%d)"%(BLOCK_X,BLOCK_Y,BLOCK_Z,block_loop[0],block_loop[1],block_loop[2],block[0],block[1],block[2]))
        shape = max(M*K, N*K, M*N)

        A = lhs.flatten()
        B = rhs.flatten()
        T = res.flatten()
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                for blockIdx_x in range(BLOCK_X): # 9
                    A_shared = mem.new((shape), "zero")
                    A_shared_local = mem.new((M*K), "zero")
                    B_shared_local = mem.new((N*K), "zero")
                    T_local = mem.new((M*N), "zero")
                    T_local_shared = mem.new((M*N), "zero")
                    # lhs HBM -> L2
                    for m in range(M):
                        for k in range(K):
                            A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                    # lhs L2 -> L1
                    for m in range(M):
                        for k in range(K):
                            A_shared_local[m*K + k] = A_shared[m*K + k]
                    # rhs HBM -> L2
                    for n in range(N):
                        for k in range(K):
                            A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                    # rhs L2 -> L1
                    for n in range(N):
                        for k in range(K):
                            B_shared_local[n*K + k] = A_shared[n*K + k]
                    # calculate
                    for m in range(M):
                        for n in range(N):
                            T_local[m*N + n] = 0
                            for k in range(K):
                                T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                    # out L1 -> L2
                    for m in range(M):
                        for n in range(N):
                            T_local_shared[m*N + n] = T_local[m*N + n]
                    # out L2 -> HBM
                    for threadIdx_z in range(block[2]): # 7
                        for threadIdx_y in range(block[1]): # 4
                            for threadIdx_x in range(block[0]): # 64
                                C_o_o_o = (M*N)//block[0]//block[1]//block[2]
                                for m_n_fused_o_o_o in range(C_o_o_o):
                                    left = blockIdx_y*M*N_ori + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)//N)*N_ori + blockIdx_x*N + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)%N)
                                    right = ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)//N)*N + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)%N)
                                    T[left] = T_local_shared[right]                                
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       5. step5_dense_1
    # =============================================================== #
    def step5_dense_1(self, param):
        print('code: step5_dense_1')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma = param
        M_ori, N_ori, K_ori = shape
        wmma_m = wmma[0] # 16
        wmma_n = wmma[1] # 16
        wmma_k = wmma[2] # 16
        # block=(64,4,7)
        # calculate [64,112,2048] in 1 thread
        M = block_loop[1]*block[1]*wmma_m # 1*4*16 = 64
        N = block_loop[2]*block[2]*wmma_n # 1*7*16 = 112
        K = K_ori
        # grid=(BLOCK_X,BLOCK_X,BLOCK_Z)=(9,2,1)
        BLOCK_X = N_ori//N # 1008/112 = 9
        BLOCK_Y = M_ori//M # 128/64 = 2
        BLOCK_Z = block_loop[2] # 1
        print("grid(%d,%d,%d), block_loop(%d,%d,%d), block(%d,%d,%d)"%(BLOCK_X,BLOCK_Y,BLOCK_Z,block_loop[0],block_loop[1],block_loop[2],block[0],block[1],block[2]))
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
                    T_local = mem.new((M*N), "zero")
                    T_local_shared = mem.new((M*N), "zero")
                    # lhs HBM -> L2
                    for m in range(M):
                        for k in range(K):
                            A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                    # lhs L2 -> L1
                    for m in range(M):
                        for k in range(K):
                            A_shared_local[m*K + k] = A_shared[m*K + k]
                    # rhs HBM -> L2
                    for n in range(N):
                        for k in range(K):
                            A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                    # rhs L2 -> L1
                    for n in range(N):
                        for k in range(K):
                            B_shared_local[n*K + k] = A_shared[n*K + k]
                    # calculate
                    for m in range(M):
                        for n in range(N):
                            T_local[m*N + n] = 0
                            for k in range(K):
                                T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                    # # -------------- step 1.0
                    # # out L1 -> L2
                    # for m_o_o in range(M//wmma_m//block[1]):
                    #     for n_o_o in range(N//wmma_n//block[2]):
                    #         for m_o_i in range(block[1]):
                    #             for n_o_i in range(block[2]):
                    #                 for m_i in range(wmma_m):
                    #                     for n_i in range(wmma_n):
                    #                         left = ((m_o_o*block[1] + m_o_i)*wmma_m + m_i)*N + ((n_o_o*block[2] + n_o_i)*wmma_n + n_i)
                    #                         T_local_shared[left] = T_local[left]
                    # # out L2 -> HBM
                    # for threadIdx_z in range(block[2]): # 7
                    #     for threadIdx_y in range(block[1]): # 4
                    #         for threadIdx_x in range(block[0]): # 64
                    #             C_o_o_o = (M*N)//block[0]//block[1]//block[2]
                    #             for m_n_fused_o_o_o in range(C_o_o_o):
                    #                 left = blockIdx_y*M*N_ori + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)//N)*N_ori + blockIdx_x*N + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)%N)
                    #                 right = ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)//N)*N + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)%N)
                    #                 T[left] = T_local_shared[right]

                    # -------------- step 2.0
                    # out L1 -> L2
                    for threadIdx_z in range(block[2]):
                        for threadIdx_y in range(block[1]):
                            for m_o_o in range(M//wmma_m//block[1]):
                                for n_o_o in range(N//wmma_n//block[2]):
                                    for m_i in range(wmma_m):
                                        for n_i in range(wmma_n):
                                            left = ((m_o_o*block[1] + threadIdx_y)*wmma_m + m_i)*N + ((n_o_o*block[2] + threadIdx_z)*wmma_n + n_i)
                                            T_local_shared[left] = T_local[left]
                    # out L2 -> HBM
                    for threadIdx_z in range(block[2]): # 7
                        for threadIdx_y in range(block[1]): # 4
                            for threadIdx_x in range(block[0]): # 64
                                C_o_o_o = (M*N)//block[0]//block[1]//block[2]
                                for m_n_fused_o_o_o in range(C_o_o_o):
                                    left = blockIdx_y*M*N_ori + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)//N)*N_ori + blockIdx_x*N + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)%N)
                                    right = ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)//N)*N + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)%N)
                                    T[left] = T_local_shared[right]

        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       5. step5_dense
    # =============================================================== #
    def step5_dense(self, param):
        print('code: step5_dense')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma = param
        M_ori, N_ori, K_ori = shape
        wmma_m = wmma[0] # 16
        wmma_n = wmma[1] # 16
        wmma_k = wmma[2] # 16
        # block=(64,4,7)
        # calculate [64,112,2048] in 1 thread
        M = block_loop[1]*block[1]*wmma_m # 1*4*16 = 64
        N = block_loop[2]*block[2]*wmma_n # 1*7*16 = 112
        K = K_ori
        # grid=(BLOCK_X,BLOCK_X,BLOCK_Z)=(9,2,1)
        BLOCK_X = N_ori//N # 1008/112 = 9
        BLOCK_Y = M_ori//M # 128/64 = 2
        BLOCK_Z = block_loop[2] # 1
        print("grid(%d,%d,%d), block_loop(%d,%d,%d), block(%d,%d,%d)"%(BLOCK_X,BLOCK_Y,BLOCK_Z,block_loop[0],block_loop[1],block_loop[2],block[0],block[1],block[2]))

        A = lhs.flatten()
        B = rhs.flatten()
        T = res.flatten()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z):
            for blockIdx_y in range(BLOCK_Y): # M
                for blockIdx_x in range(BLOCK_X): # N
                    for threadIdx_z in range(block[2]): # 7
                        N = N//block[2]
                        for threadIdx_y in range(block[1]): # 4
                            M = M//block[1]
                            # import pdb;pdb.set_trace()
                            shape = max(M*K, N*K, M*N)
                            A_shared = mem.new((shape), "zero")
                            A_shared_local = mem.new((M*K), "zero")
                            B_shared_local = mem.new((N*K), "zero")
                            T_local = mem.new((M*N), "zero")
                            T_local_shared = mem.new((M*N), "zero")
                            # lhs HBM -> L2
                            for m in range(M):
                                for k in range(K):
                                    A_shared[m*K + k] = A[blockIdx_y*M*K + m*K + k]
                            # lhs L2 -> L1
                            for m in range(M):
                                for k in range(K):
                                    A_shared_local[m*K + k] = A_shared[m*K + k]
                            # rhs HBM -> L2
                            for n in range(N):
                                for k in range(K):
                                    A_shared[n*K + k] = B[blockIdx_x*N*K + n*K + k]
                            # rhs L2 -> L1
                            for n in range(N):
                                for k in range(K):
                                    B_shared_local[n*K + k] = A_shared[n*K + k]
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0
                                    for k in range(K):
                                        T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                            # out L1 -> L2
                            for m_o_o in range(M//wmma_m):
                                for n_o_o in range(N//wmma_n):
                                    for m_i in range(wmma_m):
                                        for n_i in range(wmma_n):
                                            left = ((m_o_o*block[1] + threadIdx_y)*wmma_m + m_i)*N + ((n_o_o*block[2] + threadIdx_z)*wmma_n + n_i)
                                            T_local_shared[left] = T_local[left]
                            import pdb;pdb.set_trace()
                            # out L2 -> HBM
                            for threadIdx_x in range(block[0]): # 64
                                C_o_o_o = (M*N)//block[0]
                                for m_n_fused_o_o_o in range(C_o_o_o):
                                    left = blockIdx_y*M*N_ori + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)//N)*N_ori + blockIdx_x*N + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)%N)
                                    right = ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)//N)*N + ((((m_n_fused_o_o_o*block[2] + threadIdx_z)*block[1] + threadIdx_y)*block[0] + threadIdx_x)%N)
                                    T[left] = T_local_shared[right]
        res = np.reshape(T, (M_ori, N_ori))
        return res
