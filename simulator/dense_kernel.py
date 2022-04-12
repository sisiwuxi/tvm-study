from numpy import broadcast
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
    #                       0. step0
    # =============================================================== #
    def step0(self, param):
        print('code: step0')
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
    #                       1. step1_0_1
    # =============================================================== #
    def step1_0_1(self, param):
        print('code: step1_0_1')
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

                    # C_o_o_o = (M*N)//vec
                    # for m_n_fused_o in range(C_o_o_o):
                    #     for m_n_fused_i in range(vec):
                    #         left = blockIdx_y*M*N_ori + ((m_n_fused_o*vec + m_n_fused_i)//N)*N_ori + blockIdx_x*N + ((m_n_fused_o*vec + m_n_fused_i)%N)
                    #         right = ((m_n_fused_o*bvec + m_n_fused_i)//N)*N + ((m_n_fused_o*vec + m_n_fused_i)%N)
                    #         T[left] = T_local_shared[right]

                    # C_o_o_o = (M*N)//vec//block[0]
                    # for m_n_fused_o_o in range(C_o_o_o):
                    #     for m_n_fused_o_i in range(block[0]):
                    #         for m_n_fused_i in range(vec):
                    #             left = blockIdx_y*M*N_ori + (((m_n_fused_o_o*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)//N)*N_ori + blockIdx_x*N + (((m_n_fused_o_o*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)%N)
                    #             right = (((m_n_fused_o_o*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)//N)*N + (((m_n_fused_o_o*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)%N)
                    #             T[left] = T_local_shared[right]

                    # C_o_o_o = (M*N)//vec//block[0]//block[1]
                    # for m_n_fused_o_o_o in range(C_o_o_o):
                    #     for m_n_fused_o_o_i in range(block[1]):
                    #         for m_n_fused_o_i in range(block[0]):
                    #             for m_n_fused_i in range(vec):
                    #                 left = blockIdx_y*M*N_ori + ((((m_n_fused_o_o_o*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)//N)*N_ori + blockIdx_x*N + ((((m_n_fused_o_o_o*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)%N)
                    #                 right = ((((m_n_fused_o_o_o*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)//N)*N + ((((m_n_fused_o_o_o*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)%N)
                    #                 T[left] = T_local_shared[right]

                    C_o_o_o = (M*N)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for m_n_fused_o_o_o_i in range(block[2]):
                            for m_n_fused_o_o_i in range(block[1]):
                                for m_n_fused_o_i in range(block[0]):
                                    for m_n_fused_i in range(vec):
                                        left = blockIdx_y*M*N_ori + (((((m_n_fused_o_o_o_o*block[2] + m_n_fused_o_o_o_i)*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)//N)*N_ori + blockIdx_x*N + (((((m_n_fused_o_o_o_o*block[2] + m_n_fused_o_o_o_i)*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)%N)
                                        right = (((((m_n_fused_o_o_o_o*block[2] + m_n_fused_o_o_o_i)*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)//N)*N + (((((m_n_fused_o_o_o_o*block[2] + m_n_fused_o_o_o_i)*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)%N)
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
                    # C_o_o_o = (M*N)//vec
                    # for m_n_fused_o in range(C_o_o_o):
                    #     for m_n_fused_i in range(vec):
                    #         m_n_fused = m_n_fused_o*vec + m_n_fused_i
                    #         left = blockIdx_y*M*N_ori + (m_n_fused//N)*N_ori + blockIdx_x*N + (m_n_fused%N)
                    #         right = (m_n_fused//N)*N + (m_n_fused%N)
                    #         T[left] = T_local_shared[right]

                    # C_o_o_o = (M*N)//vec//block[0]
                    # for m_n_fused_o_o in range(C_o_o_o):
                    #     for m_n_fused_o_i in range(block[0]):
                    #         for m_n_fused_i in range(vec):
                    #             m_n_fused_o = m_n_fused_o_o*block[0] + m_n_fused_o_i
                    #             m_n_fused = m_n_fused_o*vec + m_n_fused_i
                    #             left = blockIdx_y*M*N_ori + (m_n_fused//N)*N_ori + blockIdx_x*N + (m_n_fused%N)
                    #             right = (m_n_fused//N)*N + (m_n_fused%N)
                    #             T[left] = T_local_shared[right]

                    # C_o_o_o = (M*N)//vec//block[0]//block[1]
                    # for m_n_fused_o_o_o in range(C_o_o_o):
                    #     for m_n_fused_o_o_i in range(block[1]):
                    #         for m_n_fused_o_i in range(block[0]):
                    #             for m_n_fused_i in range(vec):
                    #                 m_n_fused_o_o = m_n_fused_o_o_o*block[1] + m_n_fused_o_o_i
                    #                 m_n_fused_o = m_n_fused_o_o*block[0] + m_n_fused_o_i
                    #                 m_n_fused = m_n_fused_o*vec + m_n_fused_i
                    #                 left = blockIdx_y*M*N_ori + (m_n_fused//N)*N_ori + blockIdx_x*N + (m_n_fused%N)
                    #                 right = (m_n_fused//N)*N + (m_n_fused%N)
                    #                 T[left] = T_local_shared[right]
                    C_o_o_o = (M*N)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for m_n_fused_o_o_o_i in range(block[2]):
                            for m_n_fused_o_o_i in range(block[1]):
                                for m_n_fused_o_i in range(block[0]):
                                    for m_n_fused_i in range(vec):
                                        m_n_fused_o_o_o = m_n_fused_o_o_o_o*block[2] + m_n_fused_o_o_o_i
                                        m_n_fused_o_o = m_n_fused_o_o_o*block[1] + m_n_fused_o_o_i
                                        m_n_fused_o = m_n_fused_o_o*block[0] + m_n_fused_o_i
                                        m_n_fused = m_n_fused_o*vec + m_n_fused_i
                                        left = blockIdx_y*M*N_ori + (m_n_fused//N)*N_ori + blockIdx_x*N + (m_n_fused%N)
                                        right = (m_n_fused//N)*N + (m_n_fused%N)
                                        T[left] = T_local_shared[right]
        res = np.reshape(T, (M_ori, N_ori))
        return res

    def ramp(self, T, T_local_shared, vec, param):
        m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M,N_ori,N,blockIdx_x,N_alignment = param
        for m_n_fused_i in range(vec):
            m_n_fused_o_o_o = m_n_fused_o_o_o_o*block[2] + threadIdx_z
            m_n_fused_o_o = m_n_fused_o_o_o*block[1] + threadIdx_y
            m_n_fused_o = m_n_fused_o_o*block[0] + threadIdx_x
            m_n_fused = m_n_fused_o*vec + m_n_fused_i
            m = m_n_fused//N
            n = m_n_fused%N
            # left = blockIdx_y*M*N_ori + (m)*N_ori + blockIdx_x*N + (n)
            # right = (m)*N + (n)
            left = blockIdx_y*M*N_ori + (m)*N_ori + blockIdx_x*N + (n)
            right = m*N_alignment + n
            T[left] = T_local_shared[right]
        return


    # =============================================================== #
    #                       1. step1
    # =============================================================== #
    def step1(self, param):
        print('code: step1')
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
                    C_o_o_o = (M*N)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for threadIdx_z in range(block[2]):
                            for threadIdx_y in range(block[1]):
                                for threadIdx_x in range(block[0]):
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M,N_ori,N,blockIdx_x,N
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       2.0 step2_0_full
    # =============================================================== #
    def step2_0_full(self, param):
        print('code: step2_0_full')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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
                    T_local = mem.new((M*N), "zero") # [64,112]
                    N_alignment = ((N + CS_align-1)//CS_align)*CS_align
                    T_local_shared = mem.new(M*N_alignment, "zero") # [64,128]
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
                    # # out L1 -> L2

                    # for m in range(M):
                    #     for n in range(N):
                    #         left = m*N_alignment + n
                    #         right = m*N + n
                    #         T_local_shared[left] = T_local[right]

                    for m_o_o in range(M//wmma_m//block[1]): # 1
                        for n_o_o in range(N//wmma_n//block[2]): # 1
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_z in range(block[2]): # 7
                                    for m_i in range(wmma_m): # 16
                                        for n_i in range(wmma_n): # 16
                                            m = ((m_o_o*block[1] + threadIdx_y)*wmma_m + m_i)
                                            n = ((n_o_o*block[2] + threadIdx_z)*wmma_n + n_i)
                                            left = m*N_alignment + n
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]

                    # # out L2 -> HBM

                    # for m_n_fused in range((M*N)):
                    #     m = m_n_fused // N
                    #     n = m_n_fused % N
                    #     left = blockIdx_y*M*N_ori + m*N_ori + blockIdx_x*N + n
                    #     right = m*N_alignment + n
                    #     T[left] = T_local_shared[right]

                    # for m_n_fused_o in range((M*N)//vec):
                    #     for m_n_fused_i in range(vec):
                    #         m_n_fused = m_n_fused_o*vec + m_n_fused_i
                    #         m = m_n_fused // N
                    #         n = m_n_fused % N
                    #         left = blockIdx_y*M*N_ori + m*N_ori + blockIdx_x*N + n
                    #         right = m*N_alignment + n
                    #         T[left] = T_local_shared[right]

                    # C_o_o_o = (M*N)//vec//block[0]//block[1]//block[2]
                    # for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                    #     for threadIdx_z in range(block[2]): # 7
                    #         for threadIdx_y in range(block[1]): # 4
                    #             for threadIdx_x in range(block[0]): # 64
                    #                 for m_n_fused_i in range(vec): # 2
                    #                     m_n_fused_o_o_o = m_n_fused_o_o_o_o*block[2] + threadIdx_z
                    #                     m_n_fused_o_o = m_n_fused_o_o_o*block[1] + threadIdx_y
                    #                     m_n_fused_o = m_n_fused_o_o*block[0] + threadIdx_x
                    #                     m_n_fused = m_n_fused_o*vec + m_n_fused_i
                    #                     m = m_n_fused // N
                    #                     n = m_n_fused % N
                    #                     left = blockIdx_y*M*N_ori + m*N_ori + blockIdx_x*N + n
                    #                     right = m*N_alignment + n
                    #                     T[left] = T_local_shared[right]

                    C_o_o_o = (M*N)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M,N_ori,N,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)

        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       2.0 step2_0
    # =============================================================== #
    def step2_0(self, param):
        print('code: step2_0')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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
                    T_local = mem.new((M*N), "zero") # [64,112]
                    N_alignment = ((N + CS_align-1)//CS_align)*CS_align
                    T_local_shared = mem.new(M*N_alignment, "zero") # [64,128]
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
                    # # out L1 -> L2
                    for m_o_o in range(M//wmma_m//block[1]): # 1
                        for n_o_o in range(N//wmma_n//block[2]): # 1
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_z in range(block[2]): # 7
                                    for m_i in range(wmma_m): # 16
                                        for n_i in range(wmma_n): # 16
                                            m = ((m_o_o*block[1] + threadIdx_y)*wmma_m + m_i)
                                            n = ((n_o_o*block[2] + threadIdx_z)*wmma_n + n_i)
                                            left = m*N_alignment + n
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]

                    # # out L2 -> HBM
                    C_o_o_o = (M*N)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M,N_ori,N,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)

        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       2.1 step2_1
    # =============================================================== #
    def step2_1(self, param):
        print('code: step2_1')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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
        M_shared = M
        M = M//block[1]
        
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                for blockIdx_x in range(BLOCK_X): # 9
                    N_alignment = ((N + CS_align-1)//CS_align)*CS_align
                    T_local_shared = mem.new((M_shared*N_alignment), "zero")
                    shape = max(M*K, N*K, M*N)
                    A_shared = mem.new((shape), "zero")
                    
                    for threadIdx_y in range(block[1]): # 4
                        A_shared_local = mem.new((M*K), "zero")
                        B_shared_local = mem.new((N*K), "zero")
                        T_local = mem.new((M*N), "zero")
                        # lhs HBM -> L2
                        for m in range(M):
                            for k in range(K):
                                left = m*K + k
                                right = blockIdx_y*M_shared*K + threadIdx_y*M*K + m*K + k
                                A_shared[left] = A[right]
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
                        
                        # # out L1 -> L2

                        # for m in range(M):
                        #     for n in range(N):
                        #         left = threadIdx_y*M*N_alignment + m*N_alignment + n
                        #         right = m*N + n
                        #         T_local_shared[left] = T_local[right]

                        for m_o in range(M//wmma_m):
                            for m_i in range(wmma_m): # 16
                                for n_o_o in range(N//wmma_n//block[2]): # 1
                                    for n_o_i in range(block[2]):
                                        for n_i in range(wmma_n): # 16
                                            n_o = n_o_o*block[2] + n_o_i
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            left = threadIdx_y*M*N_alignment + m*N_alignment + n
                                            right = m*N + n
                                            if left == 8192:
                                                import pdb;pdb.set_trace()
                                            T_local_shared[left] = T_local[right]

                    # import pdb;pdb.set_trace()
                    # # out L2 -> HBM
                    # for m_n_fused in range((M_shared*N)):
                    #     m = m_n_fused // N
                    #     n = m_n_fused % N
                    #     left = blockIdx_y*M_shared*N_ori + m*N_ori + blockIdx_x*N + n
                    #     right = m*N_alignment + n
                    #     T[left] = T_local_shared[right]

                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)

        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       2 step2
    # =============================================================== #
    def step2(self, param):
        print('code: step2')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M
        M = M//block[1]
        N_shared = N
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                for blockIdx_x in range(BLOCK_X): # 9
                    # how to reuse A_shared without compute_at
                    T_local_shared = mem.new((M_shared*N_alignment), "zero")
                    shape = max(M*K, N*K, M*N)
                    A_shared = mem.new((shape), "zero")
                    # import pdb;pdb.set_trace()
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            A_shared_local = mem.new((M*K), "zero")
                            B_shared_local = mem.new((N*K), "zero")
                            T_local = mem.new((M*N), "zero")
                            # lhs HBM -> L2
                            for m in range(M):
                                for k in range(K):
                                    left = m*K + k
                                    right = blockIdx_y*M_shared*K + threadIdx_y*M*K + m*K + k
                                    A_shared[left] = A[right]
                            # lhs L2 -> L1
                            for m in range(M):
                                for k in range(K):
                                    A_shared_local[m*K + k] = A_shared[m*K + k]
                            # rhs HBM -> L2
                            for n in range(N):
                                for k in range(K):
                                    left = n*K + k
                                    right = blockIdx_x*N_shared*K + threadIdx_z*N*K + n*K + k
                                    A_shared[left] = B[right]
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
                            
                            # # out L1 -> L2
                            # import pdb;pdb.set_trace()
                            # for m in range(M):
                            #     for n in range(N):
                            #         right = m*N + n
                            #         A_shared[right] = T_local[right]
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]

                            # import pdb;pdb.set_trace()
                    # # out L2 -> HBM

                    # import pdb;pdb.set_trace()
                    # for m in range(M_shared):
                    #     for n in range(N_shared):
                    #         left = blockIdx_y*M_shared*N_ori + m*N_ori + blockIdx_x*N_shared + n
                    #         right = m*N_alignment + n
                    #         T[left] = T_local_shared[right]

                    # C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    # for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                    #     for threadIdx_z in range(block[2]): # 7
                    #         for threadIdx_y in range(block[1]): # 4
                    #             for threadIdx_x in range(block[0]): # 64
                    #                 for m_n_fused_i in range(vec): # 2
                    #                     m_n_fused_o_o_o = m_n_fused_o_o_o_o*block[2] + threadIdx_z
                    #                     m_n_fused_o_o = m_n_fused_o_o_o*block[1] + threadIdx_y
                    #                     m_n_fused_o = m_n_fused_o_o*block[0] + threadIdx_x
                    #                     m_n_fused = m_n_fused_o*vec + m_n_fused_i
                    #                     m = m_n_fused//N_shared
                    #                     n = m_n_fused%N_shared
                    #                     left = blockIdx_y*M_shared*N_ori + m*N_ori + blockIdx_x*N_shared + n
                    #                     right = m*N_alignment + n
                    #                     T[left] = T_local_shared[right]

                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       3 step3
    # =============================================================== #
    def step3(self, param):
        print('code: step3')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M
        M = M//block[1]
        N_shared = N
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                for blockIdx_x in range(BLOCK_X): # 9
                    # how to reuse A_shared without compute_at
                    T_local_shared = mem.new((M_shared*N_alignment), "zero")
                    shape = max(M*K, N*K, M*N)
                    A_shared = mem.new((shape), "zero")
                    # import pdb;pdb.set_trace()
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            A_shared_local = mem.new((M*K), "zero")
                            B_shared_local = mem.new((N*K), "zero")
                            T_local = mem.new((M*N), "zero")
                            # lhs HBM -> L2
                            for m in range(M):
                                for k in range(K):
                                    left = m*K + k
                                    right = blockIdx_y*M_shared*K + threadIdx_y*M*K + m*K + k
                                    A_shared[left] = A[right]
                            # lhs L2 -> L1
                            for m in range(M):
                                for k in range(K):
                                    A_shared_local[m*K + k] = A_shared[m*K + k]
                            # rhs HBM -> L2
                            for n in range(N):
                                for k in range(K):
                                    left = n*K + k
                                    right = blockIdx_x*N_shared*K + threadIdx_z*N*K + n*K + k
                                    A_shared[left] = B[right]
                            # rhs L2 -> L1
                            for n in range(N):
                                for k in range(K):
                                    B_shared_local[n*K + k] = A_shared[n*K + k]
                            # calculate
                            # for m in range(M):
                            #     for n in range(N):
                            #         T_local[m*N + n] = 0
                            #         for k in range(K):
                            #             T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            # for m_o in range(M//wmma_m): # 1
                            #     for m_i in range(wmma_m): # 16
                            #         for n_o in range(N//wmma_n): # 1
                            #             for n_i in range(wmma_n): # 16
                            #                 m = m_o*wmma_m + m_i
                            #                 n = n_o*wmma_n + n_i
                            #                 T_local[m*N + n] = 0
                            #                 for k_o_o in range(K//wmma_k//k_factor): # 16
                            #                     for k_o_i in range(k_factor): # 8
                            #                         for k_i in range(wmma_k):
                            #                             k_o = k_o_o*k_factor + k_o_i
                            #                             k = k_o*wmma_k + k_i
                            #                             T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m):
                                        for n_o in range(N//wmma_n):
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k):
                                                        k_o = k_o_o*k_factor + k_o_i
                                                        k = k_o*wmma_k + k_i
                                                        T_local[m*N + n] += A_shared_local[m*K + k] * B_shared_local[n*K + k]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       4.0 step4_0
    # =============================================================== #
    def step4_0(self, param):
        print('code: step4_0')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M
        M = M//block[1]
        N_shared = N
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        shape = max(M*K, N*K, M*N)
        A_shared = mem.new((shape), "zero")        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero")
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            A_shared_local = mem.new((M*K), "zero")
                            B_shared_local = mem.new((N*K), "zero")
                            # lhs HBM -> L2
                            for m in range(M):
                                for k in range(K):
                                    left = m*K + k
                                    right = blockIdx_y*M_shared*K + threadIdx_y*M*K + m*K + k
                                    A_shared[left] = A[right]
                            # lhs L2 -> L1
                            for m in range(M):
                                for k in range(K):
                                    A_shared_local[m*K + k] = A_shared[m*K + k]
                            # rhs HBM -> L2
                            for n in range(N):
                                for k in range(K):
                                    left = n*K + k
                                    right = blockIdx_x*N_shared*K + threadIdx_z*N*K + n*K + k
                                    A_shared[left] = B[right]
                            # rhs L2 -> L1
                            for n in range(N):
                                for k in range(K):
                                    B_shared_local[n*K + k] = A_shared[n*K + k]
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0
                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m):
                                        for n_o in range(N//wmma_n):
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k):
                                                        k_o = k_o_o*k_factor + k_o_i
                                                        k = k_o*wmma_k + k_i
                                                        T_local[m*N + n] += A_shared_local[m*K + k] * B_shared_local[n*K + k]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # import pdb;pdb.set_trace()
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       4.1 step4_1
    # =============================================================== #
    def step4_1(self, param):
        print('code: step4_1')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M
        M = M//block[1]
        N_shared = N
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        B_shared = mem.new((N*K), "zero")
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero")
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            B_shared_local = mem.new((N*K), "zero")
                            # rhs HBM -> L2
                            for n in range(N):
                                for k in range(K):
                                    left = n*K + k
                                    right = blockIdx_x*N_shared*K + threadIdx_z*N*K + n*K + k
                                    B_shared[left] = B[right]
                            # rhs L2 -> L1
                            for n in range(N):
                                for k in range(K):
                                    B_shared_local[n*K + k] = B_shared[n*K + k]
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            A_shared = mem.new(M_shared*K_ko, "zero") # 8192
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                for m in range(M_shared): # 64
                                    for k in range(K_ko): # 128
                                        left = m*K_ko + k
                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                        # A_shared[64,128]
                                        A_shared[left] = A[right]
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for m in range(M): # 16
                                            for k in range(K_m): # 16
                                                left = m*K_m + k
                                                right = threadIdx_y*M*K_ko + m*K_ko + k_o_i*K_m + k
                                                A_shared_local[left] = A_shared[right]
                                        for m_i in range(wmma_m): # 16
                                            for n_o in range(N//wmma_n): # 1
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i

                                                    for k_i in range(wmma_k): # 16
                                                        k_o = k_o_o*k_factor + k_o_i
                                                        k = k_o*wmma_k + k_i
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K + k]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       4 step4
    # =============================================================== #
    def step4(self, param):
        print('code: step4')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            A_shared = mem.new(M_shared*K_ko, "zero") # 64*128 = 8192
                            B_shared = mem.new((N_shared*K_ko), "zero") # 112*128 = 14336
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                for m in range(M_shared): # 64
                                    for k in range(K_ko): # 128
                                        left = m*K_ko + k
                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                        # A_shared[64,128]
                                        A_shared[left] = A[right]
                                # rhs HBM -> L2
                                for n in range(N_shared): # 112
                                    for k in range(K_ko): # 128
                                        left = n*K_ko + k
                                        right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                        B_shared[left] = B[right]
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for m in range(M): # 16
                                            for k in range(K_m): # 16
                                                left = m*K_m + k
                                                right = threadIdx_y*M*K_ko + m*K_ko + k_o_i*K_m + k
                                                A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for n in range(N): # 16
                                            for k in range(K_m): # 16
                                                left = n*K_m + k
                                                right = threadIdx_z*N*K_ko + n*K_ko + k_o_i*K_m + k
                                                B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       5 step5
    # =============================================================== #
    def step5(self, param):
        print('code: step5')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            A_shared = mem.new(M_shared*K_ko, "zero") # 64*128 = 8192
                            B_shared = mem.new((N_shared*K_ko), "zero") # 112*128 = 14336
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                for m in range(M_shared): # 64
                                    for k in range(K_ko): # 128
                                        left = m*K_ko + k
                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                        # A_shared[64,128]
                                        A_shared[left] = A[right]
                                # rhs HBM -> L2
                                for n in range(N_shared): # 112
                                    for k in range(K_ko): # 128
                                        left = n*K_ko + k
                                        right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                        B_shared[left] = B[right]
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for as_m_o in range(M//wmma_m): # 16
                                            for as_k_o in range(K_m//wmma_k): # 16
                                                for as_m_i in range(wmma_m): # 16
                                                    for as_k_i in range(wmma_k): # 16
                                                        m = as_m_o*wmma_m + as_m_i
                                                        k = as_k_o*wmma_k + as_k_i
                                                        left = m*K_m + k
                                                        right = threadIdx_y*M*K_ko + m*K_ko + k_o_i*K_m + k
                                                        A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for n in range(N): # 16
                                            for k in range(K_m): # 16
                                                left = n*K_m + k
                                                right = threadIdx_z*N*K_ko + n*K_ko + k_o_i*K_m + k
                                                B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       6 step6
    # =============================================================== #
    def step6(self, param):
        print('code: step6')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            A_shared = mem.new(M_shared*K_ko, "zero") # 64*128 = 8192
                            B_shared = mem.new((N_shared*K_ko), "zero") # 112*128 = 14336
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                for m in range(M_shared): # 64
                                    for k in range(K_ko): # 128
                                        left = m*K_ko + k
                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                        # A_shared[64,128]
                                        A_shared[left] = A[right]
                                # rhs HBM -> L2
                                for n in range(N_shared): # 112
                                    for k in range(K_ko): # 128
                                        left = n*K_ko + k
                                        right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                        B_shared[left] = B[right]
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for as_m_o in range(M//wmma_m): # 16
                                            for as_k_o in range(K_m//wmma_k): # 16
                                                for as_m_i in range(wmma_m): # 16
                                                    for as_k_i in range(wmma_k): # 16
                                                        m = as_m_o*wmma_m + as_m_i
                                                        k = as_k_o*wmma_k + as_k_i
                                                        left = m*K_m + k
                                                        right = threadIdx_y*M*K_ko + m*K_ko + k_o_i*K_m + k
                                                        A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for bs_n_o in range(N//wmma_n): # 16
                                            for bs_k_o in range(K_m//wmma_k): # 16
                                                for bs_n_i in range(wmma_n): # 16
                                                    for bs_k_i in range(wmma_k): # 16
                                                        n = bs_n_o*wmma_m + bs_n_i
                                                        k = bs_k_o*wmma_k + bs_k_i
                                                        left = n*K_m + k
                                                        right = threadIdx_z*N*K_ko + n*K_ko + k_o_i*K_m + k
                                                        B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       7.0 step7_0
    # =============================================================== #
    def step7_0(self, param):
        print('code: step7_0')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 136
                            A_shared = mem.new(M_shared*K_ko_alignment, "zero") # 64*136 = 8704
                            B_shared = mem.new((N_shared*K_ko), "zero") # 112*128 = 14336
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2

                                # for m in range(M_shared): # 64
                                #     for k in range(K_ko): # 128
                                #         left = m*K_ko_alignment + k
                                #         right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                #         # A_shared[64,136]
                                #         A_shared[left] = A[right]

                                # for m_k_fuse in range(M_shared*K_ko): # 64*128 = 8192
                                #     m = m_k_fuse//K_ko
                                #     k = m_k_fuse%K_ko
                                #     left = m*K_ko_alignment + k
                                #     right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                #     A_shared[left] = A[right]

                                # mk_o_o_o = (M_shared*K_ko)//vec
                                # for m_k_fuse_o in range(mk_o_o_o): # 4096
                                #     for m_k_fuse_i in range(vec): # 2
                                #         m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                #         m = m_k_fuse//K_ko
                                #         k = m_k_fuse%K_ko
                                #         left = m*K_ko_alignment + k
                                #         right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                #         A_shared[left] = A[right]

                                # mk_o_o_o = (M_shared*K_ko)//vec//block[0]
                                # for m_k_fuse_o_o in range(mk_o_o_o): # 64
                                #     for m_k_fuse_o_i in range(block[0]): # 64
                                #         for m_k_fuse_i in range(vec): # 2
                                #             m_k_fuse_o = m_k_fuse_o_o*block[0] + m_k_fuse_o_i
                                #             m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                #             m = m_k_fuse//K_ko
                                #             k = m_k_fuse%K_ko
                                #             left = m*K_ko_alignment + k
                                #             right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                #             A_shared[left] = A[right]

                                # mk_o_o_o = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                # for m_k_fuse_o_o_o in range(mk_o_o_o): # 16
                                #     for m_k_fuse_o_o_i in range(block[1]): # 4
                                #         for m_k_fuse_o_i in range(block[0]): # 64
                                #             for m_k_fuse_i in range(vec): # 2
                                #                 m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + m_k_fuse_o_o_i
                                #                 m_k_fuse_o = m_k_fuse_o_o*block[0] + m_k_fuse_o_i
                                #                 m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                #                 m = m_k_fuse//K_ko
                                #                 k = m_k_fuse%K_ko
                                #                 left = m*K_ko_alignment + k
                                #                 right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                #                 A_shared[left] = A[right]
                                

                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for m_k_fuse_o_o_o_i in range(block[2]): # 7
                                        m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + m_k_fuse_o_o_o_i
                                        if m_k_fuse_o_o_o < upper_bound:
                                            for m_k_fuse_o_o_i in range(block[1]): # 4
                                                m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + m_k_fuse_o_o_i
                                                for m_k_fuse_o_i in range(block[0]): # 64
                                                    for m_k_fuse_i in range(vec): # 2
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + m_k_fuse_o_i
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        left = m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                for n in range(N_shared): # 112
                                    for k in range(K_ko): # 128
                                        left = n*K_ko + k
                                        right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                        B_shared[left] = B[right]
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for as_m_o in range(M//wmma_m): # 16
                                            for as_k_o in range(K_m//wmma_k): # 16
                                                for as_m_i in range(wmma_m): # 16
                                                    for as_k_i in range(wmma_k): # 16
                                                        m = as_m_o*wmma_m + as_m_i
                                                        k = as_k_o*wmma_k + as_k_i
                                                        left = m*K_m + k
                                                        right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                                        A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for bs_n_o in range(N//wmma_n): # 16
                                            for bs_k_o in range(K_m//wmma_k): # 16
                                                for bs_n_i in range(wmma_n): # 16
                                                    for bs_k_i in range(wmma_k): # 16
                                                        n = bs_n_o*wmma_m + bs_n_i
                                                        k = bs_k_o*wmma_k + bs_k_i
                                                        left = n*K_m + k
                                                        right = threadIdx_z*N*K_ko + n*K_ko + k_o_i*K_m + k
                                                        B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       7 step7
    # =============================================================== #
    def step7(self, param):
        print('code: step7')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 136
                            A_shared = mem.new(M_shared*K_ko_alignment, "zero") # 64*136 = 8704
                            B_shared = mem.new((N_shared*K_ko), "zero") # 112*128 = 14336
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                for m_k_fuse_i in range(vec): # 2
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        left = m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                for n in range(N_shared): # 112
                                    for k in range(K_ko): # 128
                                        left = n*K_ko + k
                                        right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                        B_shared[left] = B[right]
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for as_m_o in range(M//wmma_m): # 16
                                            for as_k_o in range(K_m//wmma_k): # 16
                                                for as_m_i in range(wmma_m): # 16
                                                    for as_k_i in range(wmma_k): # 16
                                                        m = as_m_o*wmma_m + as_m_i
                                                        k = as_k_o*wmma_k + as_k_i
                                                        left = m*K_m + k
                                                        right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                                        A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for bs_n_o in range(N//wmma_n): # 16
                                            for bs_k_o in range(K_m//wmma_k): # 16
                                                for bs_n_i in range(wmma_n): # 16
                                                    for bs_k_i in range(wmma_k): # 16
                                                        n = bs_n_o*wmma_m + bs_n_i
                                                        k = bs_k_o*wmma_k + bs_k_i
                                                        left = n*K_m + k
                                                        right = threadIdx_z*N*K_ko + n*K_ko + k_o_i*K_m + k
                                                        B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       8 step8_0
    # =============================================================== #
    def step8_0(self, param):
        print('code: step8_0')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 136
                            A_shared = mem.new(M_shared*K_ko_alignment, "zero") # 64*136 = 8704
                            B_shared = mem.new((N_shared*K_ko_alignment), "zero") # 112*136 = 15232
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                for m_k_fuse_i in range(vec): # 2
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        left = m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                # for n in range(N_shared): # 112
                                #     for k in range(K_ko): # 128
                                #         left = n*K_ko_alignment + k
                                #         right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                #         B_shared[left] = B[right]

                                nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                                for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                    for n_k_fuse_o_o_o_i in range(block[2]): # 7
                                        n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + n_k_fuse_o_o_o_i
                                        for n_k_fuse_o_o_i in range(block[1]): # 4
                                            n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + n_k_fuse_o_o_i
                                            for n_k_fuse_o_i in range(block[0]): # 64
                                                n_k_fuse_o = n_k_fuse_o_o*block[0] + n_k_fuse_o_i
                                                for n_k_fuse_i in range(vec): # 2
                                                    n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                    n = n_k_fuse//K_ko
                                                    k = n_k_fuse%K_ko
                                                    left = n*K_ko_alignment + k
                                                    right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                    B_shared[left] = B[right]

                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for as_m_o in range(M//wmma_m): # 16
                                            for as_k_o in range(K_m//wmma_k): # 16
                                                for as_m_i in range(wmma_m): # 16
                                                    for as_k_i in range(wmma_k): # 16
                                                        m = as_m_o*wmma_m + as_m_i
                                                        k = as_k_o*wmma_k + as_k_i
                                                        left = m*K_m + k
                                                        right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                                        A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for bs_n_o in range(N//wmma_n): # 16
                                            for bs_k_o in range(K_m//wmma_k): # 16
                                                for bs_n_i in range(wmma_n): # 16
                                                    for bs_k_i in range(wmma_k): # 16
                                                        n = bs_n_o*wmma_m + bs_n_i
                                                        k = bs_k_o*wmma_k + bs_k_i
                                                        left = n*K_m + k
                                                        right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                                        B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       8 step8
    # =============================================================== #
    def step8(self, param):
        print('code: step8')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 136
                            A_shared = mem.new(M_shared*K_ko_alignment, "zero") # 64*136 = 8704
                            B_shared = mem.new((N_shared*K_ko_alignment), "zero") # 112*136 = 15232
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                for m_k_fuse_i in range(vec): # 2
                                                    m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        left = m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                                for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                    for bs_threadIdx_z in range(block[2]): # 7
                                        for bs_threadIdx_y in range(block[1]): # 4
                                            for bs_threadIdx_x in range(block[0]): # 64
                                                for n_k_fuse_i in range(vec): # 2
                                                    n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + bs_threadIdx_z
                                                    n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + bs_threadIdx_y
                                                    n_k_fuse_o = n_k_fuse_o_o*block[0] + bs_threadIdx_x
                                                    n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                    n = n_k_fuse//K_ko
                                                    k = n_k_fuse%K_ko
                                                    left = n*K_ko_alignment + k
                                                    right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                    B_shared[left] = B[right]

                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for as_m_o in range(M//wmma_m): # 16
                                            for as_k_o in range(K_m//wmma_k): # 16
                                                for as_m_i in range(wmma_m): # 16
                                                    for as_k_i in range(wmma_k): # 16
                                                        m = as_m_o*wmma_m + as_m_i
                                                        k = as_k_o*wmma_k + as_k_i
                                                        left = m*K_m + k
                                                        right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                                        A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for bs_n_o in range(N//wmma_n): # 16
                                            for bs_k_o in range(K_m//wmma_k): # 16
                                                for bs_n_i in range(wmma_n): # 16
                                                    for bs_k_i in range(wmma_k): # 16
                                                        n = bs_n_o*wmma_m + bs_n_i
                                                        k = bs_k_o*wmma_k + bs_k_i
                                                        left = n*K_m + k
                                                        right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                                        B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res

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
    #                       0. step0
    # =============================================================== #
    def step0(self, param):
        print('code: step0')
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
    #                       1. step1_0_1
    # =============================================================== #
    def step1_0_1(self, param):
        print('code: step1_0_1')
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

                    # C_o_o_o = (M*N)//vec
                    # for m_n_fused_o in range(C_o_o_o):
                    #     for m_n_fused_i in range(vec):
                    #         left = blockIdx_y*M*N_ori + ((m_n_fused_o*vec + m_n_fused_i)//N)*N_ori + blockIdx_x*N + ((m_n_fused_o*vec + m_n_fused_i)%N)
                    #         right = ((m_n_fused_o*bvec + m_n_fused_i)//N)*N + ((m_n_fused_o*vec + m_n_fused_i)%N)
                    #         T[left] = T_local_shared[right]

                    # C_o_o_o = (M*N)//vec//block[0]
                    # for m_n_fused_o_o in range(C_o_o_o):
                    #     for m_n_fused_o_i in range(block[0]):
                    #         for m_n_fused_i in range(vec):
                    #             left = blockIdx_y*M*N_ori + (((m_n_fused_o_o*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)//N)*N_ori + blockIdx_x*N + (((m_n_fused_o_o*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)%N)
                    #             right = (((m_n_fused_o_o*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)//N)*N + (((m_n_fused_o_o*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)%N)
                    #             T[left] = T_local_shared[right]

                    # C_o_o_o = (M*N)//vec//block[0]//block[1]
                    # for m_n_fused_o_o_o in range(C_o_o_o):
                    #     for m_n_fused_o_o_i in range(block[1]):
                    #         for m_n_fused_o_i in range(block[0]):
                    #             for m_n_fused_i in range(vec):
                    #                 left = blockIdx_y*M*N_ori + ((((m_n_fused_o_o_o*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)//N)*N_ori + blockIdx_x*N + ((((m_n_fused_o_o_o*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)%N)
                    #                 right = ((((m_n_fused_o_o_o*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)//N)*N + ((((m_n_fused_o_o_o*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)%N)
                    #                 T[left] = T_local_shared[right]

                    C_o_o_o = (M*N)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for m_n_fused_o_o_o_i in range(block[2]):
                            for m_n_fused_o_o_i in range(block[1]):
                                for m_n_fused_o_i in range(block[0]):
                                    for m_n_fused_i in range(vec):
                                        left = blockIdx_y*M*N_ori + (((((m_n_fused_o_o_o_o*block[2] + m_n_fused_o_o_o_i)*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)//N)*N_ori + blockIdx_x*N + (((((m_n_fused_o_o_o_o*block[2] + m_n_fused_o_o_o_i)*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)%N)
                                        right = (((((m_n_fused_o_o_o_o*block[2] + m_n_fused_o_o_o_i)*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)//N)*N + (((((m_n_fused_o_o_o_o*block[2] + m_n_fused_o_o_o_i)*block[1] + m_n_fused_o_o_i)*block[0] + m_n_fused_o_i)*vec + m_n_fused_i)%N)
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
                    # C_o_o_o = (M*N)//vec
                    # for m_n_fused_o in range(C_o_o_o):
                    #     for m_n_fused_i in range(vec):
                    #         m_n_fused = m_n_fused_o*vec + m_n_fused_i
                    #         left = blockIdx_y*M*N_ori + (m_n_fused//N)*N_ori + blockIdx_x*N + (m_n_fused%N)
                    #         right = (m_n_fused//N)*N + (m_n_fused%N)
                    #         T[left] = T_local_shared[right]

                    # C_o_o_o = (M*N)//vec//block[0]
                    # for m_n_fused_o_o in range(C_o_o_o):
                    #     for m_n_fused_o_i in range(block[0]):
                    #         for m_n_fused_i in range(vec):
                    #             m_n_fused_o = m_n_fused_o_o*block[0] + m_n_fused_o_i
                    #             m_n_fused = m_n_fused_o*vec + m_n_fused_i
                    #             left = blockIdx_y*M*N_ori + (m_n_fused//N)*N_ori + blockIdx_x*N + (m_n_fused%N)
                    #             right = (m_n_fused//N)*N + (m_n_fused%N)
                    #             T[left] = T_local_shared[right]

                    # C_o_o_o = (M*N)//vec//block[0]//block[1]
                    # for m_n_fused_o_o_o in range(C_o_o_o):
                    #     for m_n_fused_o_o_i in range(block[1]):
                    #         for m_n_fused_o_i in range(block[0]):
                    #             for m_n_fused_i in range(vec):
                    #                 m_n_fused_o_o = m_n_fused_o_o_o*block[1] + m_n_fused_o_o_i
                    #                 m_n_fused_o = m_n_fused_o_o*block[0] + m_n_fused_o_i
                    #                 m_n_fused = m_n_fused_o*vec + m_n_fused_i
                    #                 left = blockIdx_y*M*N_ori + (m_n_fused//N)*N_ori + blockIdx_x*N + (m_n_fused%N)
                    #                 right = (m_n_fused//N)*N + (m_n_fused%N)
                    #                 T[left] = T_local_shared[right]
                    C_o_o_o = (M*N)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for m_n_fused_o_o_o_i in range(block[2]):
                            for m_n_fused_o_o_i in range(block[1]):
                                for m_n_fused_o_i in range(block[0]):
                                    for m_n_fused_i in range(vec):
                                        m_n_fused_o_o_o = m_n_fused_o_o_o_o*block[2] + m_n_fused_o_o_o_i
                                        m_n_fused_o_o = m_n_fused_o_o_o*block[1] + m_n_fused_o_o_i
                                        m_n_fused_o = m_n_fused_o_o*block[0] + m_n_fused_o_i
                                        m_n_fused = m_n_fused_o*vec + m_n_fused_i
                                        left = blockIdx_y*M*N_ori + (m_n_fused//N)*N_ori + blockIdx_x*N + (m_n_fused%N)
                                        right = (m_n_fused//N)*N + (m_n_fused%N)
                                        T[left] = T_local_shared[right]
        res = np.reshape(T, (M_ori, N_ori))
        return res

    def ramp(self, T, T_local_shared, vec, param):
        m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M,N_ori,N,blockIdx_x,N_alignment = param
        for m_n_fused_i in range(vec):
            m_n_fused_o_o_o = m_n_fused_o_o_o_o*block[2] + threadIdx_z
            m_n_fused_o_o = m_n_fused_o_o_o*block[1] + threadIdx_y
            m_n_fused_o = m_n_fused_o_o*block[0] + threadIdx_x
            m_n_fused = m_n_fused_o*vec + m_n_fused_i
            m = m_n_fused//N
            n = m_n_fused%N
            # left = blockIdx_y*M*N_ori + (m)*N_ori + blockIdx_x*N + (n)
            # right = (m)*N + (n)
            left = blockIdx_y*M*N_ori + (m)*N_ori + blockIdx_x*N + (n)
            right = m*N_alignment + n
            T[left] = T_local_shared[right]
        return


    # =============================================================== #
    #                       1. step1
    # =============================================================== #
    def step1(self, param):
        print('code: step1')
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
                    C_o_o_o = (M*N)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for threadIdx_z in range(block[2]):
                            for threadIdx_y in range(block[1]):
                                for threadIdx_x in range(block[0]):
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M,N_ori,N,blockIdx_x,N
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       2.0 step2_0_full
    # =============================================================== #
    def step2_0_full(self, param):
        print('code: step2_0_full')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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
                    T_local = mem.new((M*N), "zero") # [64,112]
                    N_alignment = ((N + CS_align-1)//CS_align)*CS_align
                    T_local_shared = mem.new(M*N_alignment, "zero") # [64,128]
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
                    # # out L1 -> L2

                    # for m in range(M):
                    #     for n in range(N):
                    #         left = m*N_alignment + n
                    #         right = m*N + n
                    #         T_local_shared[left] = T_local[right]

                    for m_o_o in range(M//wmma_m//block[1]): # 1
                        for n_o_o in range(N//wmma_n//block[2]): # 1
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_z in range(block[2]): # 7
                                    for m_i in range(wmma_m): # 16
                                        for n_i in range(wmma_n): # 16
                                            m = ((m_o_o*block[1] + threadIdx_y)*wmma_m + m_i)
                                            n = ((n_o_o*block[2] + threadIdx_z)*wmma_n + n_i)
                                            left = m*N_alignment + n
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]

                    # # out L2 -> HBM

                    # for m_n_fused in range((M*N)):
                    #     m = m_n_fused // N
                    #     n = m_n_fused % N
                    #     left = blockIdx_y*M*N_ori + m*N_ori + blockIdx_x*N + n
                    #     right = m*N_alignment + n
                    #     T[left] = T_local_shared[right]

                    # for m_n_fused_o in range((M*N)//vec):
                    #     for m_n_fused_i in range(vec):
                    #         m_n_fused = m_n_fused_o*vec + m_n_fused_i
                    #         m = m_n_fused // N
                    #         n = m_n_fused % N
                    #         left = blockIdx_y*M*N_ori + m*N_ori + blockIdx_x*N + n
                    #         right = m*N_alignment + n
                    #         T[left] = T_local_shared[right]

                    # C_o_o_o = (M*N)//vec//block[0]//block[1]//block[2]
                    # for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                    #     for threadIdx_z in range(block[2]): # 7
                    #         for threadIdx_y in range(block[1]): # 4
                    #             for threadIdx_x in range(block[0]): # 64
                    #                 for m_n_fused_i in range(vec): # 2
                    #                     m_n_fused_o_o_o = m_n_fused_o_o_o_o*block[2] + threadIdx_z
                    #                     m_n_fused_o_o = m_n_fused_o_o_o*block[1] + threadIdx_y
                    #                     m_n_fused_o = m_n_fused_o_o*block[0] + threadIdx_x
                    #                     m_n_fused = m_n_fused_o*vec + m_n_fused_i
                    #                     m = m_n_fused // N
                    #                     n = m_n_fused % N
                    #                     left = blockIdx_y*M*N_ori + m*N_ori + blockIdx_x*N + n
                    #                     right = m*N_alignment + n
                    #                     T[left] = T_local_shared[right]

                    C_o_o_o = (M*N)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M,N_ori,N,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)

        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       2.0 step2_0
    # =============================================================== #
    def step2_0(self, param):
        print('code: step2_0')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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
                    T_local = mem.new((M*N), "zero") # [64,112]
                    N_alignment = ((N + CS_align-1)//CS_align)*CS_align
                    T_local_shared = mem.new(M*N_alignment, "zero") # [64,128]
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
                    # # out L1 -> L2
                    for m_o_o in range(M//wmma_m//block[1]): # 1
                        for n_o_o in range(N//wmma_n//block[2]): # 1
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_z in range(block[2]): # 7
                                    for m_i in range(wmma_m): # 16
                                        for n_i in range(wmma_n): # 16
                                            m = ((m_o_o*block[1] + threadIdx_y)*wmma_m + m_i)
                                            n = ((n_o_o*block[2] + threadIdx_z)*wmma_n + n_i)
                                            left = m*N_alignment + n
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]

                    # # out L2 -> HBM
                    C_o_o_o = (M*N)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M,N_ori,N,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)

        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       2.1 step2_1
    # =============================================================== #
    def step2_1(self, param):
        print('code: step2_1')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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
        M_shared = M
        M = M//block[1]
        
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                for blockIdx_x in range(BLOCK_X): # 9
                    N_alignment = ((N + CS_align-1)//CS_align)*CS_align
                    T_local_shared = mem.new((M_shared*N_alignment), "zero")
                    shape = max(M*K, N*K, M*N)
                    A_shared = mem.new((shape), "zero")
                    
                    for threadIdx_y in range(block[1]): # 4
                        A_shared_local = mem.new((M*K), "zero")
                        B_shared_local = mem.new((N*K), "zero")
                        T_local = mem.new((M*N), "zero")
                        # lhs HBM -> L2
                        for m in range(M):
                            for k in range(K):
                                left = m*K + k
                                right = blockIdx_y*M_shared*K + threadIdx_y*M*K + m*K + k
                                A_shared[left] = A[right]
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
                        
                        # # out L1 -> L2

                        # for m in range(M):
                        #     for n in range(N):
                        #         left = threadIdx_y*M*N_alignment + m*N_alignment + n
                        #         right = m*N + n
                        #         T_local_shared[left] = T_local[right]

                        for m_o in range(M//wmma_m):
                            for m_i in range(wmma_m): # 16
                                for n_o_o in range(N//wmma_n//block[2]): # 1
                                    for n_o_i in range(block[2]):
                                        for n_i in range(wmma_n): # 16
                                            n_o = n_o_o*block[2] + n_o_i
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            left = threadIdx_y*M*N_alignment + m*N_alignment + n
                                            right = m*N + n
                                            if left == 8192:
                                                import pdb;pdb.set_trace()
                                            T_local_shared[left] = T_local[right]

                    # import pdb;pdb.set_trace()
                    # # out L2 -> HBM
                    # for m_n_fused in range((M_shared*N)):
                    #     m = m_n_fused // N
                    #     n = m_n_fused % N
                    #     left = blockIdx_y*M_shared*N_ori + m*N_ori + blockIdx_x*N + n
                    #     right = m*N_alignment + n
                    #     T[left] = T_local_shared[right]

                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)

        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       2 step2
    # =============================================================== #
    def step2(self, param):
        print('code: step2')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M
        M = M//block[1]
        N_shared = N
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                for blockIdx_x in range(BLOCK_X): # 9
                    # how to reuse A_shared without compute_at
                    T_local_shared = mem.new((M_shared*N_alignment), "zero")
                    shape = max(M*K, N*K, M*N)
                    A_shared = mem.new((shape), "zero")
                    # import pdb;pdb.set_trace()
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            A_shared_local = mem.new((M*K), "zero")
                            B_shared_local = mem.new((N*K), "zero")
                            T_local = mem.new((M*N), "zero")
                            # lhs HBM -> L2
                            for m in range(M):
                                for k in range(K):
                                    left = m*K + k
                                    right = blockIdx_y*M_shared*K + threadIdx_y*M*K + m*K + k
                                    A_shared[left] = A[right]
                            # lhs L2 -> L1
                            for m in range(M):
                                for k in range(K):
                                    A_shared_local[m*K + k] = A_shared[m*K + k]
                            # rhs HBM -> L2
                            for n in range(N):
                                for k in range(K):
                                    left = n*K + k
                                    right = blockIdx_x*N_shared*K + threadIdx_z*N*K + n*K + k
                                    A_shared[left] = B[right]
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
                            
                            # # out L1 -> L2
                            # import pdb;pdb.set_trace()
                            # for m in range(M):
                            #     for n in range(N):
                            #         right = m*N + n
                            #         A_shared[right] = T_local[right]
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]

                            # import pdb;pdb.set_trace()
                    # # out L2 -> HBM

                    # import pdb;pdb.set_trace()
                    # for m in range(M_shared):
                    #     for n in range(N_shared):
                    #         left = blockIdx_y*M_shared*N_ori + m*N_ori + blockIdx_x*N_shared + n
                    #         right = m*N_alignment + n
                    #         T[left] = T_local_shared[right]

                    # C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    # for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                    #     for threadIdx_z in range(block[2]): # 7
                    #         for threadIdx_y in range(block[1]): # 4
                    #             for threadIdx_x in range(block[0]): # 64
                    #                 for m_n_fused_i in range(vec): # 2
                    #                     m_n_fused_o_o_o = m_n_fused_o_o_o_o*block[2] + threadIdx_z
                    #                     m_n_fused_o_o = m_n_fused_o_o_o*block[1] + threadIdx_y
                    #                     m_n_fused_o = m_n_fused_o_o*block[0] + threadIdx_x
                    #                     m_n_fused = m_n_fused_o*vec + m_n_fused_i
                    #                     m = m_n_fused//N_shared
                    #                     n = m_n_fused%N_shared
                    #                     left = blockIdx_y*M_shared*N_ori + m*N_ori + blockIdx_x*N_shared + n
                    #                     right = m*N_alignment + n
                    #                     T[left] = T_local_shared[right]

                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       3 step3
    # =============================================================== #
    def step3(self, param):
        print('code: step3')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M
        M = M//block[1]
        N_shared = N
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                for blockIdx_x in range(BLOCK_X): # 9
                    # how to reuse A_shared without compute_at
                    T_local_shared = mem.new((M_shared*N_alignment), "zero")
                    shape = max(M*K, N*K, M*N)
                    A_shared = mem.new((shape), "zero")
                    # import pdb;pdb.set_trace()
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            A_shared_local = mem.new((M*K), "zero")
                            B_shared_local = mem.new((N*K), "zero")
                            T_local = mem.new((M*N), "zero")
                            # lhs HBM -> L2
                            for m in range(M):
                                for k in range(K):
                                    left = m*K + k
                                    right = blockIdx_y*M_shared*K + threadIdx_y*M*K + m*K + k
                                    A_shared[left] = A[right]
                            # lhs L2 -> L1
                            for m in range(M):
                                for k in range(K):
                                    A_shared_local[m*K + k] = A_shared[m*K + k]
                            # rhs HBM -> L2
                            for n in range(N):
                                for k in range(K):
                                    left = n*K + k
                                    right = blockIdx_x*N_shared*K + threadIdx_z*N*K + n*K + k
                                    A_shared[left] = B[right]
                            # rhs L2 -> L1
                            for n in range(N):
                                for k in range(K):
                                    B_shared_local[n*K + k] = A_shared[n*K + k]
                            # calculate
                            # for m in range(M):
                            #     for n in range(N):
                            #         T_local[m*N + n] = 0
                            #         for k in range(K):
                            #             T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            # for m_o in range(M//wmma_m): # 1
                            #     for m_i in range(wmma_m): # 16
                            #         for n_o in range(N//wmma_n): # 1
                            #             for n_i in range(wmma_n): # 16
                            #                 m = m_o*wmma_m + m_i
                            #                 n = n_o*wmma_n + n_i
                            #                 T_local[m*N + n] = 0
                            #                 for k_o_o in range(K//wmma_k//k_factor): # 16
                            #                     for k_o_i in range(k_factor): # 8
                            #                         for k_i in range(wmma_k):
                            #                             k_o = k_o_o*k_factor + k_o_i
                            #                             k = k_o*wmma_k + k_i
                            #                             T_local[m*N + n] = T_local[m*N + n] + A_shared_local[m*K + k] * B_shared_local[n*K + k]

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m):
                                        for n_o in range(N//wmma_n):
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k):
                                                        k_o = k_o_o*k_factor + k_o_i
                                                        k = k_o*wmma_k + k_i
                                                        T_local[m*N + n] += A_shared_local[m*K + k] * B_shared_local[n*K + k]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o):
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       4.0 step4_0
    # =============================================================== #
    def step4_0(self, param):
        print('code: step4_0')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M
        M = M//block[1]
        N_shared = N
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        shape = max(M*K, N*K, M*N)
        A_shared = mem.new((shape), "zero")        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero")
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            A_shared_local = mem.new((M*K), "zero")
                            B_shared_local = mem.new((N*K), "zero")
                            # lhs HBM -> L2
                            for m in range(M):
                                for k in range(K):
                                    left = m*K + k
                                    right = blockIdx_y*M_shared*K + threadIdx_y*M*K + m*K + k
                                    A_shared[left] = A[right]
                            # lhs L2 -> L1
                            for m in range(M):
                                for k in range(K):
                                    A_shared_local[m*K + k] = A_shared[m*K + k]
                            # rhs HBM -> L2
                            for n in range(N):
                                for k in range(K):
                                    left = n*K + k
                                    right = blockIdx_x*N_shared*K + threadIdx_z*N*K + n*K + k
                                    A_shared[left] = B[right]
                            # rhs L2 -> L1
                            for n in range(N):
                                for k in range(K):
                                    B_shared_local[n*K + k] = A_shared[n*K + k]
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0
                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m):
                                        for n_o in range(N//wmma_n):
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k):
                                                        k_o = k_o_o*k_factor + k_o_i
                                                        k = k_o*wmma_k + k_i
                                                        T_local[m*N + n] += A_shared_local[m*K + k] * B_shared_local[n*K + k]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # import pdb;pdb.set_trace()
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       4.1 step4_1
    # =============================================================== #
    def step4_1(self, param):
        print('code: step4_1')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M
        M = M//block[1]
        N_shared = N
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        B_shared = mem.new((N*K), "zero")
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero")
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            B_shared_local = mem.new((N*K), "zero")
                            # rhs HBM -> L2
                            for n in range(N):
                                for k in range(K):
                                    left = n*K + k
                                    right = blockIdx_x*N_shared*K + threadIdx_z*N*K + n*K + k
                                    B_shared[left] = B[right]
                            # rhs L2 -> L1
                            for n in range(N):
                                for k in range(K):
                                    B_shared_local[n*K + k] = B_shared[n*K + k]
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            A_shared = mem.new(M_shared*K_ko, "zero") # 8192
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                for m in range(M_shared): # 64
                                    for k in range(K_ko): # 128
                                        left = m*K_ko + k
                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                        # A_shared[64,128]
                                        A_shared[left] = A[right]
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for m in range(M): # 16
                                            for k in range(K_m): # 16
                                                left = m*K_m + k
                                                right = threadIdx_y*M*K_ko + m*K_ko + k_o_i*K_m + k
                                                A_shared_local[left] = A_shared[right]
                                        for m_i in range(wmma_m): # 16
                                            for n_o in range(N//wmma_n): # 1
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i

                                                    for k_i in range(wmma_k): # 16
                                                        k_o = k_o_o*k_factor + k_o_i
                                                        k = k_o*wmma_k + k_i
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K + k]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       4 step4
    # =============================================================== #
    def step4(self, param):
        print('code: step4')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            A_shared = mem.new(M_shared*K_ko, "zero") # 64*128 = 8192
                            B_shared = mem.new((N_shared*K_ko), "zero") # 112*128 = 14336
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                for m in range(M_shared): # 64
                                    for k in range(K_ko): # 128
                                        left = m*K_ko + k
                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                        # A_shared[64,128]
                                        A_shared[left] = A[right]
                                # rhs HBM -> L2
                                for n in range(N_shared): # 112
                                    for k in range(K_ko): # 128
                                        left = n*K_ko + k
                                        right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                        B_shared[left] = B[right]
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for m in range(M): # 16
                                            for k in range(K_m): # 16
                                                left = m*K_m + k
                                                right = threadIdx_y*M*K_ko + m*K_ko + k_o_i*K_m + k
                                                A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for n in range(N): # 16
                                            for k in range(K_m): # 16
                                                left = n*K_m + k
                                                right = threadIdx_z*N*K_ko + n*K_ko + k_o_i*K_m + k
                                                B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       5 step5
    # =============================================================== #
    def step5(self, param):
        print('code: step5')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            A_shared = mem.new(M_shared*K_ko, "zero") # 64*128 = 8192
                            B_shared = mem.new((N_shared*K_ko), "zero") # 112*128 = 14336
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                for m in range(M_shared): # 64
                                    for k in range(K_ko): # 128
                                        left = m*K_ko + k
                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                        # A_shared[64,128]
                                        A_shared[left] = A[right]
                                # rhs HBM -> L2
                                for n in range(N_shared): # 112
                                    for k in range(K_ko): # 128
                                        left = n*K_ko + k
                                        right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                        B_shared[left] = B[right]
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for as_m_o in range(M//wmma_m): # 16
                                            for as_k_o in range(K_m//wmma_k): # 16
                                                for as_m_i in range(wmma_m): # 16
                                                    for as_k_i in range(wmma_k): # 16
                                                        m = as_m_o*wmma_m + as_m_i
                                                        k = as_k_o*wmma_k + as_k_i
                                                        left = m*K_m + k
                                                        right = threadIdx_y*M*K_ko + m*K_ko + k_o_i*K_m + k
                                                        A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for n in range(N): # 16
                                            for k in range(K_m): # 16
                                                left = n*K_m + k
                                                right = threadIdx_z*N*K_ko + n*K_ko + k_o_i*K_m + k
                                                B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       6 step6
    # =============================================================== #
    def step6(self, param):
        print('code: step6')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor = param
        M_ori, N_ori, K_ori = shape
        CS_align = 128 # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            A_shared = mem.new(M_shared*K_ko, "zero") # 64*128 = 8192
                            B_shared = mem.new((N_shared*K_ko), "zero") # 112*128 = 14336
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                for m in range(M_shared): # 64
                                    for k in range(K_ko): # 128
                                        left = m*K_ko + k
                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                        # A_shared[64,128]
                                        A_shared[left] = A[right]
                                # rhs HBM -> L2
                                for n in range(N_shared): # 112
                                    for k in range(K_ko): # 128
                                        left = n*K_ko + k
                                        right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                        B_shared[left] = B[right]
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for as_m_o in range(M//wmma_m): # 16
                                            for as_k_o in range(K_m//wmma_k): # 16
                                                for as_m_i in range(wmma_m): # 16
                                                    for as_k_i in range(wmma_k): # 16
                                                        m = as_m_o*wmma_m + as_m_i
                                                        k = as_k_o*wmma_k + as_k_i
                                                        left = m*K_m + k
                                                        right = threadIdx_y*M*K_ko + m*K_ko + k_o_i*K_m + k
                                                        A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for bs_n_o in range(N//wmma_n): # 16
                                            for bs_k_o in range(K_m//wmma_k): # 16
                                                for bs_n_i in range(wmma_n): # 16
                                                    for bs_k_i in range(wmma_k): # 16
                                                        n = bs_n_o*wmma_m + bs_n_i
                                                        k = bs_k_o*wmma_k + bs_k_i
                                                        left = n*K_m + k
                                                        right = threadIdx_z*N*K_ko + n*K_ko + k_o_i*K_m + k
                                                        B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       7.0 step7_0
    # =============================================================== #
    def step7_0(self, param):
        print('code: step7_0')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 136
                            A_shared = mem.new(M_shared*K_ko_alignment, "zero") # 64*136 = 8704
                            B_shared = mem.new((N_shared*K_ko), "zero") # 112*128 = 14336
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2

                                # for m in range(M_shared): # 64
                                #     for k in range(K_ko): # 128
                                #         left = m*K_ko_alignment + k
                                #         right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                #         # A_shared[64,136]
                                #         A_shared[left] = A[right]

                                # for m_k_fuse in range(M_shared*K_ko): # 64*128 = 8192
                                #     m = m_k_fuse//K_ko
                                #     k = m_k_fuse%K_ko
                                #     left = m*K_ko_alignment + k
                                #     right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                #     A_shared[left] = A[right]

                                # mk_o_o_o = (M_shared*K_ko)//vec
                                # for m_k_fuse_o in range(mk_o_o_o): # 4096
                                #     for m_k_fuse_i in range(vec): # 2
                                #         m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                #         m = m_k_fuse//K_ko
                                #         k = m_k_fuse%K_ko
                                #         left = m*K_ko_alignment + k
                                #         right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                #         A_shared[left] = A[right]

                                # mk_o_o_o = (M_shared*K_ko)//vec//block[0]
                                # for m_k_fuse_o_o in range(mk_o_o_o): # 64
                                #     for m_k_fuse_o_i in range(block[0]): # 64
                                #         for m_k_fuse_i in range(vec): # 2
                                #             m_k_fuse_o = m_k_fuse_o_o*block[0] + m_k_fuse_o_i
                                #             m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                #             m = m_k_fuse//K_ko
                                #             k = m_k_fuse%K_ko
                                #             left = m*K_ko_alignment + k
                                #             right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                #             A_shared[left] = A[right]

                                # mk_o_o_o = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                # for m_k_fuse_o_o_o in range(mk_o_o_o): # 16
                                #     for m_k_fuse_o_o_i in range(block[1]): # 4
                                #         for m_k_fuse_o_i in range(block[0]): # 64
                                #             for m_k_fuse_i in range(vec): # 2
                                #                 m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + m_k_fuse_o_o_i
                                #                 m_k_fuse_o = m_k_fuse_o_o*block[0] + m_k_fuse_o_i
                                #                 m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                #                 m = m_k_fuse//K_ko
                                #                 k = m_k_fuse%K_ko
                                #                 left = m*K_ko_alignment + k
                                #                 right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                #                 A_shared[left] = A[right]
                                

                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for m_k_fuse_o_o_o_i in range(block[2]): # 7
                                        m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + m_k_fuse_o_o_o_i
                                        if m_k_fuse_o_o_o < upper_bound:
                                            for m_k_fuse_o_o_i in range(block[1]): # 4
                                                m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + m_k_fuse_o_o_i
                                                for m_k_fuse_o_i in range(block[0]): # 64
                                                    for m_k_fuse_i in range(vec): # 2
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + m_k_fuse_o_i
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        left = m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                for n in range(N_shared): # 112
                                    for k in range(K_ko): # 128
                                        left = n*K_ko + k
                                        right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                        B_shared[left] = B[right]
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for as_m_o in range(M//wmma_m): # 16
                                            for as_k_o in range(K_m//wmma_k): # 16
                                                for as_m_i in range(wmma_m): # 16
                                                    for as_k_i in range(wmma_k): # 16
                                                        m = as_m_o*wmma_m + as_m_i
                                                        k = as_k_o*wmma_k + as_k_i
                                                        left = m*K_m + k
                                                        right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                                        A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for bs_n_o in range(N//wmma_n): # 16
                                            for bs_k_o in range(K_m//wmma_k): # 16
                                                for bs_n_i in range(wmma_n): # 16
                                                    for bs_k_i in range(wmma_k): # 16
                                                        n = bs_n_o*wmma_m + bs_n_i
                                                        k = bs_k_o*wmma_k + bs_k_i
                                                        left = n*K_m + k
                                                        right = threadIdx_z*N*K_ko + n*K_ko + k_o_i*K_m + k
                                                        B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       7 step7
    # =============================================================== #
    def step7(self, param):
        print('code: step7')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 136
                            A_shared = mem.new(M_shared*K_ko_alignment, "zero") # 64*136 = 8704
                            B_shared = mem.new((N_shared*K_ko), "zero") # 112*128 = 14336
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                for m_k_fuse_i in range(vec): # 2
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        left = m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                for n in range(N_shared): # 112
                                    for k in range(K_ko): # 128
                                        left = n*K_ko + k
                                        right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                        B_shared[left] = B[right]
                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for as_m_o in range(M//wmma_m): # 16
                                            for as_k_o in range(K_m//wmma_k): # 16
                                                for as_m_i in range(wmma_m): # 16
                                                    for as_k_i in range(wmma_k): # 16
                                                        m = as_m_o*wmma_m + as_m_i
                                                        k = as_k_o*wmma_k + as_k_i
                                                        left = m*K_m + k
                                                        right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                                        A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for bs_n_o in range(N//wmma_n): # 16
                                            for bs_k_o in range(K_m//wmma_k): # 16
                                                for bs_n_i in range(wmma_n): # 16
                                                    for bs_k_i in range(wmma_k): # 16
                                                        n = bs_n_o*wmma_m + bs_n_i
                                                        k = bs_k_o*wmma_k + bs_k_i
                                                        left = n*K_m + k
                                                        right = threadIdx_z*N*K_ko + n*K_ko + k_o_i*K_m + k
                                                        B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       8 step8_0
    # =============================================================== #
    def step8_0(self, param):
        print('code: step8_0')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 136
                            A_shared = mem.new(M_shared*K_ko_alignment, "zero") # 64*136 = 8704
                            B_shared = mem.new((N_shared*K_ko_alignment), "zero") # 112*136 = 15232
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                for m_k_fuse_i in range(vec): # 2
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        left = m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                # for n in range(N_shared): # 112
                                #     for k in range(K_ko): # 128
                                #         left = n*K_ko_alignment + k
                                #         right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                #         B_shared[left] = B[right]

                                nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                                for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                    for n_k_fuse_o_o_o_i in range(block[2]): # 7
                                        n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + n_k_fuse_o_o_o_i
                                        for n_k_fuse_o_o_i in range(block[1]): # 4
                                            n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + n_k_fuse_o_o_i
                                            for n_k_fuse_o_i in range(block[0]): # 64
                                                n_k_fuse_o = n_k_fuse_o_o*block[0] + n_k_fuse_o_i
                                                for n_k_fuse_i in range(vec): # 2
                                                    n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                    n = n_k_fuse//K_ko
                                                    k = n_k_fuse%K_ko
                                                    left = n*K_ko_alignment + k
                                                    right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                    B_shared[left] = B[right]

                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for as_m_o in range(M//wmma_m): # 16
                                            for as_k_o in range(K_m//wmma_k): # 16
                                                for as_m_i in range(wmma_m): # 16
                                                    for as_k_i in range(wmma_k): # 16
                                                        m = as_m_o*wmma_m + as_m_i
                                                        k = as_k_o*wmma_k + as_k_i
                                                        left = m*K_m + k
                                                        right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                                        A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for bs_n_o in range(N//wmma_n): # 16
                                            for bs_k_o in range(K_m//wmma_k): # 16
                                                for bs_n_i in range(wmma_n): # 16
                                                    for bs_k_i in range(wmma_k): # 16
                                                        n = bs_n_o*wmma_m + bs_n_i
                                                        k = bs_k_o*wmma_k + bs_k_i
                                                        left = n*K_m + k
                                                        right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                                        B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       8 step8
    # =============================================================== #
    def step8(self, param):
        print('code: step8')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 136
                            A_shared = mem.new(M_shared*K_ko_alignment, "zero") # 64*136 = 8704
                            B_shared = mem.new((N_shared*K_ko_alignment), "zero") # 112*136 = 15232
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                for m_k_fuse_i in range(vec): # 2
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        left = m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                                for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                    for bs_threadIdx_z in range(block[2]): # 7
                                        for bs_threadIdx_y in range(block[1]): # 4
                                            for bs_threadIdx_x in range(block[0]): # 64
                                                for n_k_fuse_i in range(vec): # 2
                                                    n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + bs_threadIdx_z
                                                    n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + bs_threadIdx_y
                                                    n_k_fuse_o = n_k_fuse_o_o*block[0] + bs_threadIdx_x
                                                    n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                    n = n_k_fuse//K_ko
                                                    k = n_k_fuse%K_ko
                                                    left = n*K_ko_alignment + k
                                                    right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                    B_shared[left] = B[right]

                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for as_m_o in range(M//wmma_m): # 16
                                            for as_k_o in range(K_m//wmma_k): # 16
                                                for as_m_i in range(wmma_m): # 16
                                                    for as_k_i in range(wmma_k): # 16
                                                        m = as_m_o*wmma_m + as_m_i
                                                        k = as_k_o*wmma_k + as_k_i
                                                        left = m*K_m + k
                                                        right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                                        A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for bs_n_o in range(N//wmma_n): # 16
                                            for bs_k_o in range(K_m//wmma_k): # 16
                                                for bs_n_i in range(wmma_n): # 16
                                                    for bs_k_i in range(wmma_k): # 16
                                                        n = bs_n_o*wmma_m + bs_n_i
                                                        k = bs_k_o*wmma_k + bs_k_i
                                                        left = n*K_m + k
                                                        right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                                        B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       9 step9
    # =============================================================== #
    def step9(self, param):
        print('code: step9')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 136
                            A_shared = mem.new(M_shared*K_ko_alignment, "zero") # 64*136 = 8704
                            B_shared = mem.new((N_shared*K_ko_alignment), "zero") # 112*136 = 15232
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                for m_k_fuse_i in range(vec): # 2
                                                    m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        left = m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                                for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                    for bs_threadIdx_z in range(block[2]): # 7
                                        for bs_threadIdx_y in range(block[1]): # 4
                                            for bs_threadIdx_x in range(block[0]): # 64
                                                for n_k_fuse_i in range(vec): # 2
                                                    n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + bs_threadIdx_z
                                                    n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + bs_threadIdx_y
                                                    n_k_fuse_o = n_k_fuse_o_o*block[0] + bs_threadIdx_x
                                                    n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                    n = n_k_fuse//K_ko
                                                    k = n_k_fuse%K_ko
                                                    left = n*K_ko_alignment + k
                                                    right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                    B_shared[left] = B[right]

                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        # for as_m_o in range(M//wmma_m): # 1
                                        #     for as_k_o in range(K_m//wmma_k): # 1
                                        #         for as_m_i in range(wmma_m): # 16
                                        #             for as_k_i in range(wmma_k): # 16
                                        #                 m = as_m_o*wmma_m + as_m_i
                                        #                 k = as_k_o*wmma_k + as_k_i
                                        #                 left = m*K_m + k
                                        #                 right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                        #                 A_shared_local[left] = A_shared[right]


                                        # for m in range(wmma_m): # 16
                                        #     for k in range(wmma_k): # 16
                                        #         left = m*K_m + k
                                        #         right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                        #         A_shared_local[left] = A_shared[right]

                                        # for mk in range(wmma_m*wmma_k): # 256
                                        #     m = mk//wmma_k
                                        #     k = mk%wmma_k
                                        #     left = m*K_m + k
                                        #     right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                        #     A_shared_local[left] = A_shared[right]

                                        # for mk_o in range(wmma_m*wmma_k//4): # 64
                                        #     for mk_i in range(4): # 4
                                        #         mk = mk_o*4 + mk_i
                                        #         m = mk//wmma_k
                                        #         k = mk%wmma_k
                                        #         left = m*K_m + k
                                        #         right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                        #         A_shared_local[left] = A_shared[right]

                                        for mk_o in range(wmma_m*wmma_k//4): # 64
                                            m = (mk_o*4 + 0)//wmma_k
                                            k = (mk_o*4 + 0)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 1)//wmma_k
                                            k = (mk_o*4 + 1)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 2)//wmma_k
                                            k = (mk_o*4 + 2)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 3)//wmma_k
                                            k = (mk_o*4 + 3)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for bs_n_o in range(N//wmma_n): # 16
                                            for bs_k_o in range(K_m//wmma_k): # 16
                                                for bs_n_i in range(wmma_n): # 16
                                                    for bs_k_i in range(wmma_k): # 16
                                                        n = bs_n_o*wmma_m + bs_n_i
                                                        k = bs_k_o*wmma_k + bs_k_i
                                                        left = n*K_m + k
                                                        right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                                        B_shared_local[left] = B_shared[right]
                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       10 step10
    # =============================================================== #
    def step10(self, param):
        print('code: step10')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 136
                            A_shared = mem.new(M_shared*K_ko_alignment, "zero") # 64*136 = 8704
                            B_shared = mem.new((N_shared*K_ko_alignment), "zero") # 112*136 = 15232
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # calculate
                            for m in range(M):
                                for n in range(N):
                                    T_local[m*N + n] = 0

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                for m_k_fuse_i in range(vec): # 2
                                                    m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        left = m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                                for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                    for bs_threadIdx_z in range(block[2]): # 7
                                        for bs_threadIdx_y in range(block[1]): # 4
                                            for bs_threadIdx_x in range(block[0]): # 64
                                                for n_k_fuse_i in range(vec): # 2
                                                    n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + bs_threadIdx_z
                                                    n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + bs_threadIdx_y
                                                    n_k_fuse_o = n_k_fuse_o_o*block[0] + bs_threadIdx_x
                                                    n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                    n = n_k_fuse//K_ko
                                                    k = n_k_fuse%K_ko
                                                    left = n*K_ko_alignment + k
                                                    right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                    B_shared[left] = B[right]

                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for mk_o in range(wmma_m*wmma_k//4): # 64
                                            m = (mk_o*4 + 0)//wmma_k
                                            k = (mk_o*4 + 0)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 1)//wmma_k
                                            k = (mk_o*4 + 1)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 2)//wmma_k
                                            k = (mk_o*4 + 2)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 3)//wmma_k
                                            k = (mk_o*4 + 3)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        # for bs_n_o in range(N//wmma_n): # 1
                                        #     for bs_k_o in range(K_m//wmma_k): # 1
                                        #         for bs_n_i in range(wmma_n): # 16
                                        #             for bs_k_i in range(wmma_k): # 16
                                        #                 n = bs_n_o*wmma_m + bs_n_i
                                        #                 k = bs_k_o*wmma_k + bs_k_i
                                        #                 left = n*K_m + k
                                        #                 right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                        #                 B_shared_local[left] = B_shared[right]

                                        for nk_o in range(wmma_n*wmma_k//4): # 64
                                            n = (nk_o*4 + 0)//wmma_k
                                            k = (nk_o*4 + 0)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]
                                            n = (nk_o*4 + 1)//wmma_k
                                            k = (nk_o*4 + 1)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]
                                            n = (nk_o*4 + 2)//wmma_k
                                            k = (nk_o*4 + 2)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]
                                            n = (nk_o*4 + 3)//wmma_k
                                            k = (nk_o*4 + 3)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]

                                        for n_o in range(N//wmma_n): # 1
                                            for m_i in range(wmma_m): # 16
                                                for n_i in range(wmma_n): # 16
                                                    m = m_o*wmma_m + m_i
                                                    n = n_o*wmma_n + n_i
                                                    for k_i in range(wmma_k): # 16
                                                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       11 step11
    # =============================================================== #
    def dense_broadcast(self):
        mem = MEM()
        return mem.new((16*16), "zero")

    def llvm_matrix_mad_f32x4_f32x4(self,A_shared_local,B_shared_local,T_local,param):
        N,wmma_n,wmma_m,m_o,wmma_k,K_m = param
        for n_o in range(N//wmma_n): # 1
            for m_i in range(wmma_m): # 16
                for n_i in range(wmma_n): # 16
                    m = m_o*wmma_m + m_i
                    n = n_o*wmma_n + n_i
                    for k_i in range(wmma_k): # 16
                        T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
        return T_local

    def step11(self, param):
        print('code: step11')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 136
                            A_shared = mem.new(M_shared*K_ko_alignment, "zero") # 64*136 = 8704
                            B_shared = mem.new((N_shared*K_ko_alignment), "zero") # 112*136 = 15232
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # memset
                            # for m in range(M):
                            #     for n in range(N):
                            #         T_local[m*N + n] = 0
                            # T_local[ramp(0,1,4)] = broadcast(0f32,4)
                            T_local = self.dense_broadcast()

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                for m_k_fuse_i in range(vec): # 2
                                                    m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        left = m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                                for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                    for bs_threadIdx_z in range(block[2]): # 7
                                        for bs_threadIdx_y in range(block[1]): # 4
                                            for bs_threadIdx_x in range(block[0]): # 64
                                                for n_k_fuse_i in range(vec): # 2
                                                    n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + bs_threadIdx_z
                                                    n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + bs_threadIdx_y
                                                    n_k_fuse_o = n_k_fuse_o_o*block[0] + bs_threadIdx_x
                                                    n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                    n = n_k_fuse//K_ko
                                                    k = n_k_fuse%K_ko
                                                    left = n*K_ko_alignment + k
                                                    right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                    B_shared[left] = B[right]

                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for mk_o in range(wmma_m*wmma_k//4): # 64
                                            m = (mk_o*4 + 0)//wmma_k
                                            k = (mk_o*4 + 0)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 1)//wmma_k
                                            k = (mk_o*4 + 1)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 2)//wmma_k
                                            k = (mk_o*4 + 2)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 3)//wmma_k
                                            k = (mk_o*4 + 3)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for nk_o in range(wmma_n*wmma_k//4): # 64
                                            n = (nk_o*4 + 0)//wmma_k
                                            k = (nk_o*4 + 0)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]
                                            n = (nk_o*4 + 1)//wmma_k
                                            k = (nk_o*4 + 1)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]
                                            n = (nk_o*4 + 2)//wmma_k
                                            k = (nk_o*4 + 2)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]
                                            n = (nk_o*4 + 3)//wmma_k
                                            k = (nk_o*4 + 3)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]

                                        # for n_o in range(N//wmma_n): # 1
                                        #     for m_i in range(wmma_m): # 16
                                        #         for n_i in range(wmma_n): # 16
                                        #             m = m_o*wmma_m + m_i
                                        #             n = n_o*wmma_n + n_i
                                        #             for k_i in range(wmma_k): # 16
                                        #                 T_local[m*N + n] += A_shared_local[m*K_m + k_i] * B_shared_local[n*K_m + k_i]
                                        param = N,wmma_n,wmma_m,m_o,wmma_k,K_m
                                        T_local = self.llvm_matrix_mad_f32x4_f32x4(A_shared_local,B_shared_local,T_local,param)
                            # # out L1 -> L2
                            for m_o in range(M//wmma_m):
                                for m_i in range(wmma_m): # 16
                                    for n_o in range(N//wmma_n):
                                        for n_i in range(wmma_n): # 16
                                            m = m_o*wmma_m + m_i
                                            n = n_o*wmma_n + n_i
                                            # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                                            # T_local[16,16]
                                            left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                                            right = m*N + n
                                            T_local_shared[left] = T_local[right]
                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       12 step12
    # =============================================================== #
    def step12(self, param):
        print('code: step12')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        # import pdb;pdb.set_trace()
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 2
                T_local = mem.new((M*N), "zero")
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 64,128
                for blockIdx_x in range(BLOCK_X): # 9
                    for threadIdx_y in range(block[1]): # 4
                        for threadIdx_z in range(block[2]): # 7
                            cf_ko_factor = K//wmma_k//k_factor
                            K_ko = K//cf_ko_factor # 128
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 136
                            A_shared = mem.new(M_shared*K_ko_alignment, "zero") # 64*136 = 8704
                            B_shared = mem.new((N_shared*K_ko_alignment), "zero") # 112*136 = 15232
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # memset
                            T_local = self.dense_broadcast()

                            for k_o_o in range(K//wmma_k//k_factor): # 16
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                for m_k_fuse_i in range(vec): # 2
                                                    m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        left = m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                                for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                    for bs_threadIdx_z in range(block[2]): # 7
                                        for bs_threadIdx_y in range(block[1]): # 4
                                            for bs_threadIdx_x in range(block[0]): # 64
                                                for n_k_fuse_i in range(vec): # 2
                                                    n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + bs_threadIdx_z
                                                    n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + bs_threadIdx_y
                                                    n_k_fuse_o = n_k_fuse_o_o*block[0] + bs_threadIdx_x
                                                    n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                    n = n_k_fuse//K_ko
                                                    k = n_k_fuse%K_ko
                                                    left = n*K_ko_alignment + k
                                                    right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                    B_shared[left] = B[right]

                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for mk_o in range(wmma_m*wmma_k//4): # 64
                                            m = (mk_o*4 + 0)//wmma_k
                                            k = (mk_o*4 + 0)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 1)//wmma_k
                                            k = (mk_o*4 + 1)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 2)//wmma_k
                                            k = (mk_o*4 + 2)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 3)//wmma_k
                                            k = (mk_o*4 + 3)%wmma_k
                                            left = m*K_m + k
                                            right = threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for nk_o in range(wmma_n*wmma_k//4): # 64
                                            n = (nk_o*4 + 0)//wmma_k
                                            k = (nk_o*4 + 0)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]
                                            n = (nk_o*4 + 1)//wmma_k
                                            k = (nk_o*4 + 1)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]
                                            n = (nk_o*4 + 2)//wmma_k
                                            k = (nk_o*4 + 2)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]
                                            n = (nk_o*4 + 3)//wmma_k
                                            k = (nk_o*4 + 3)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]

                                        param = N,wmma_n,wmma_m,m_o,wmma_k,K_m
                                        T_local = self.llvm_matrix_mad_f32x4_f32x4(A_shared_local,B_shared_local,T_local,param)
                            # # out L1 -> L2
                            # for m_o in range(M//wmma_m): # 1
                            #     for n_o in range(N//wmma_n): # 1
                            #         for m_i in range(wmma_m): # 16                                
                            #             for n_i in range(wmma_n): # 16
                            #                 m = m_o*wmma_m + m_i
                            #                 n = n_o*wmma_n + n_i
                            #                 # T_local_shared[64,128] aligned & reuse A_shared[16,2048]
                            #                 # T_local[16,16]
                            #                 left = threadIdx_y*M*N_alignment + m_i*N_alignment + threadIdx_z*N + n_i
                            #                 right = m*N + n
                            #                 T_local_shared[left] = T_local[right]

                            for mn_o in range(wmma_m*wmma_n//4): # 64
                                m = (mn_o*4 + 0)//wmma_n
                                n = (mn_o*4 + 0)%wmma_n
                                left = threadIdx_y*M*N_alignment + m*N_alignment + threadIdx_z*N + n
                                right = m*N + n
                                T_local_shared[left] = T_local[right]
                                m = (mn_o*4 + 1)//wmma_n
                                n = (mn_o*4 + 1)%wmma_n
                                left = threadIdx_y*M*N_alignment + m*N_alignment + threadIdx_z*N + n
                                right = m*N + n
                                T_local_shared[left] = T_local[right]
                                m = (mn_o*4 + 2)//wmma_n
                                n = (mn_o*4 + 2)%wmma_n
                                left = threadIdx_y*M*N_alignment + m*N_alignment + threadIdx_z*N + n
                                right = m*N + n
                                T_local_shared[left] = T_local[right]
                                m = (mn_o*4 + 3)//wmma_n
                                n = (mn_o*4 + 3)%wmma_n
                                left = threadIdx_y*M*N_alignment + m*N_alignment + threadIdx_z*N + n
                                right = m*N + n
                                T_local_shared[left] = T_local[right]

                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res

    # =============================================================== #
    #                       13.0 step13_0
    # =============================================================== #
    def step13_0(self, param):
        print('code: step13_0')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 4
                T_local = mem.new((M*N), "zero") # 16*16 = 256
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 32*256 = 8192
                for blockIdx_x in range(BLOCK_X): # 7
                    for threadIdx_y in range(block[1]): # 2
                        for threadIdx_z in range(block[2]): # 9
                            cf_ko_factor = K//wmma_k//k_factor # 2048//16//2 = 64
                            K_ko = K//cf_ko_factor # 2048//64 = 32
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 32
                            A_shared = mem.new(2*M_shared*K_ko_alignment, "zero") # 2*32*32 = 2048
                            B_shared = mem.new((N_shared*K_ko_alignment), "zero") # 144*32 = 4608
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            A_shared_local_1 = mem.new((M*K_m), "zero") # 256
                            B_shared_local_1 = mem.new((N*K_m), "zero") # 256                            
                            # memset
                            T_local = self.dense_broadcast()

                            # head: lhs HBM -> L2 iter0 for double buffer
                            k_o_o = 0
                            upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                            mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                            for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                for as_threadIdx_z in range(block[2]): # 7
                                    for as_threadIdx_y in range(block[1]): # 4
                                        for as_threadIdx_x in range(block[0]): # 64
                                            for m_k_fuse_i in range(vec): # 1
                                                m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                if m_k_fuse_o_o_o < upper_bound:
                                                    m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                    m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                    m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                    m = m_k_fuse//K_ko
                                                    k = m_k_fuse%K_ko
                                                    left = m*K_ko_alignment + k
                                                    right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                    A_shared[left] = A[right]
                            # import pdb;pdb.set_trace()
                            # body: use A_shared double buffer
                            k_o_o_iters = K//wmma_k//k_factor - 1 # 2048//16//2 - 1 = 64 - 1 = 63
                            for k_o_o in range(k_o_o_iters): # 63.
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                for m_k_fuse_i in range(vec): # 1
                                                    m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        # select ping-pong buffer
                                                        left = ((k_o_o+1)%2)*M_shared*K_ko_alignment + m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                                for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                    for bs_threadIdx_z in range(block[2]): # 7
                                        for bs_threadIdx_y in range(block[1]): # 4
                                            for bs_threadIdx_x in range(block[0]): # 64
                                                for n_k_fuse_i in range(vec): # 2
                                                    n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + bs_threadIdx_z
                                                    n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + bs_threadIdx_y
                                                    n_k_fuse_o = n_k_fuse_o_o*block[0] + bs_threadIdx_x
                                                    n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                    n = n_k_fuse//K_ko
                                                    k = n_k_fuse%K_ko
                                                    left = n*K_ko_alignment + k
                                                    right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                    B_shared[left] = B[right]

                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for mk_o in range(wmma_m*wmma_k//4): # 64
                                            m = (mk_o*4 + 0)//wmma_k
                                            k = (mk_o*4 + 0)%wmma_k
                                            left = m*K_m + k
                                            right = (k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            # import pdb;pdb.set_trace()
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 1)//wmma_k
                                            k = (mk_o*4 + 1)%wmma_k
                                            left = m*K_m + k
                                            right = (k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 2)//wmma_k
                                            k = (mk_o*4 + 2)%wmma_k
                                            left = m*K_m + k
                                            right = (k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                            m = (mk_o*4 + 3)//wmma_k
                                            k = (mk_o*4 + 3)%wmma_k
                                            left = m*K_m + k
                                            right = (k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                            A_shared_local[left] = A_shared[right]
                                        # rhs L2 -> L1
                                        for nk_o in range(wmma_n*wmma_k//4): # 64
                                            n = (nk_o*4 + 0)//wmma_k
                                            k = (nk_o*4 + 0)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]
                                            n = (nk_o*4 + 1)//wmma_k
                                            k = (nk_o*4 + 1)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]
                                            n = (nk_o*4 + 2)//wmma_k
                                            k = (nk_o*4 + 2)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]
                                            n = (nk_o*4 + 3)//wmma_k
                                            k = (nk_o*4 + 3)%wmma_k
                                            left = n*K_m + k
                                            right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                            B_shared_local[left] = B_shared[right]

                                        param = N,wmma_n,wmma_m,m_o,wmma_k,K_m
                                        T_local = self.llvm_matrix_mad_f32x4_f32x4(A_shared_local,B_shared_local,T_local,param)
                            # tail: calculate the last iteration without A_shared load
                            k_o_o = k_o_o_iters
                            # rhs HBM -> L2
                            nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                            for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                for bs_threadIdx_z in range(block[2]): # 7
                                    for bs_threadIdx_y in range(block[1]): # 4
                                        for bs_threadIdx_x in range(block[0]): # 64
                                            for n_k_fuse_i in range(vec): # 2
                                                n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + bs_threadIdx_z
                                                n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + bs_threadIdx_y
                                                n_k_fuse_o = n_k_fuse_o_o*block[0] + bs_threadIdx_x
                                                n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                n = n_k_fuse//K_ko
                                                k = n_k_fuse%K_ko
                                                left = n*K_ko_alignment + k
                                                right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                B_shared[left] = B[right]

                            for k_o_i in range(k_factor): # 8
                                for m_o in range(M//wmma_m): # 1
                                    # lhs L2 -> L1
                                    for mk_o in range(wmma_m*wmma_k//4): # 64
                                        m = (mk_o*4 + 0)//wmma_k
                                        k = (mk_o*4 + 0)%wmma_k
                                        left = m*K_m + k
                                        right = (k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                        A_shared_local_1[left] = A_shared[right]
                                        m = (mk_o*4 + 1)//wmma_k
                                        k = (mk_o*4 + 1)%wmma_k
                                        left = m*K_m + k
                                        right = (k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                        A_shared_local_1[left] = A_shared[right]
                                        m = (mk_o*4 + 2)//wmma_k
                                        k = (mk_o*4 + 2)%wmma_k
                                        left = m*K_m + k
                                        right = (k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                        A_shared_local_1[left] = A_shared[right]
                                        m = (mk_o*4 + 3)//wmma_k
                                        k = (mk_o*4 + 3)%wmma_k
                                        left = m*K_m + k
                                        right = (k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + m*K_ko_alignment + k_o_i*K_m + k
                                        A_shared_local_1[left] = A_shared[right]
                                    # rhs L2 -> L1
                                    for nk_o in range(wmma_n*wmma_k//4): # 64
                                        n = (nk_o*4 + 0)//wmma_k
                                        k = (nk_o*4 + 0)%wmma_k
                                        left = n*K_m + k
                                        right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                        B_shared_local_1[left] = B_shared[right]
                                        n = (nk_o*4 + 1)//wmma_k
                                        k = (nk_o*4 + 1)%wmma_k
                                        left = n*K_m + k
                                        right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                        B_shared_local_1[left] = B_shared[right]
                                        n = (nk_o*4 + 2)//wmma_k
                                        k = (nk_o*4 + 2)%wmma_k
                                        left = n*K_m + k
                                        right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                        B_shared_local_1[left] = B_shared[right]
                                        n = (nk_o*4 + 3)//wmma_k
                                        k = (nk_o*4 + 3)%wmma_k
                                        left = n*K_m + k
                                        right = threadIdx_z*N*K_ko_alignment + n*K_ko_alignment + k_o_i*K_m + k
                                        B_shared_local_1[left] = B_shared[right]

                                    param = N,wmma_n,wmma_m,m_o,wmma_k,K_m
                                    T_local = self.llvm_matrix_mad_f32x4_f32x4(A_shared_local_1,B_shared_local_1,T_local,param)

                            # # out L1 -> L2
                            for mn_o in range(wmma_m*wmma_n//4): # 64
                                m = (mn_o*4 + 0)//wmma_n
                                n = (mn_o*4 + 0)%wmma_n
                                left = threadIdx_y*M*N_alignment + m*N_alignment + threadIdx_z*N + n
                                right = m*N + n
                                T_local_shared[left] = T_local[right]
                                m = (mn_o*4 + 1)//wmma_n
                                n = (mn_o*4 + 1)%wmma_n
                                left = threadIdx_y*M*N_alignment + m*N_alignment + threadIdx_z*N + n
                                right = m*N + n
                                T_local_shared[left] = T_local[right]
                                m = (mn_o*4 + 2)//wmma_n
                                n = (mn_o*4 + 2)%wmma_n
                                left = threadIdx_y*M*N_alignment + m*N_alignment + threadIdx_z*N + n
                                right = m*N + n
                                T_local_shared[left] = T_local[right]
                                m = (mn_o*4 + 3)//wmma_n
                                n = (mn_o*4 + 3)%wmma_n
                                left = threadIdx_y*M*N_alignment + m*N_alignment + threadIdx_z*N + n
                                right = m*N + n
                                T_local_shared[left] = T_local[right]

                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       13 step13
    # =============================================================== #
    def step13(self, param):
        print('code: step13')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 4
                T_local = mem.new((M*N), "zero") # 16*16 = 256
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 32*256 = 8192
                for blockIdx_x in range(BLOCK_X): # 7
                    for threadIdx_y in range(block[1]): # 2
                        for threadIdx_z in range(block[2]): # 9
                            cf_ko_factor = K//wmma_k//k_factor # 2048//16//2 = 64
                            K_ko = K//cf_ko_factor # 2048//64 = 32
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 32
                            A_shared = mem.new(2*M_shared*K_ko_alignment, "zero") # 2*32*32 = 2048
                            B_shared = mem.new((N_shared*K_ko_alignment), "zero") # 144*32 = 4608
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((M*K_m), "zero") # 256
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            A_shared_local_1 = mem.new((M*K_m), "zero") # 256
                            B_shared_local_1 = mem.new((N*K_m), "zero") # 256                            
                            # memset
                            T_local = self.dense_broadcast()

                            # head: lhs HBM -> L2 iter0 for double buffer
                            k_o_o = 0
                            upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                            mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                            for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                for as_threadIdx_z in range(block[2]): # 7
                                    for as_threadIdx_y in range(block[1]): # 4
                                        for as_threadIdx_x in range(block[0]): # 64
                                            for m_k_fuse_i in range(vec): # 1
                                                m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                if m_k_fuse_o_o_o < upper_bound:
                                                    m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                    m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                    m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                    m = m_k_fuse//K_ko
                                                    k = m_k_fuse%K_ko
                                                    left = m*K_ko_alignment + k
                                                    right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                    A_shared[left] = A[right]
                            # import pdb;pdb.set_trace()
                            # body: use A_shared double buffer
                            k_o_o_iters = K//wmma_k//k_factor - 1 # 2048//16//2 - 1 = 64 - 1 = 63
                            for k_o_o in range(k_o_o_iters): # 63.
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                for m_k_fuse_i in range(vec): # 1
                                                    m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        # select ping-pong buffer
                                                        left = ((k_o_o+1)%2)*M_shared*K_ko_alignment + m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + K_ko + k
                                                        A_shared[left] = A[right]
                                # import pdb;pdb.set_trace()
                                # rhs HBM -> L2
                                nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                                for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                    for bs_threadIdx_z in range(block[2]): # 7
                                        for bs_threadIdx_y in range(block[1]): # 4
                                            for bs_threadIdx_x in range(block[0]): # 64
                                                for n_k_fuse_i in range(vec): # 2
                                                    n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + bs_threadIdx_z
                                                    n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + bs_threadIdx_y
                                                    n_k_fuse_o = n_k_fuse_o_o*block[0] + bs_threadIdx_x
                                                    n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                    n = n_k_fuse//K_ko
                                                    k = n_k_fuse%K_ko
                                                    left = n*K_ko_alignment + k
                                                    right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                    B_shared[left] = B[right]

                                for k_o_i in range(k_factor): # 8
                                    for m_o in range(M//wmma_m): # 1
                                        # lhs L2 -> L1
                                        for mk_o in range(wmma_m*wmma_k//4): # 64
                                            A_shared_local[((mk_o*4 + 0)//wmma_k)*K_m + ((mk_o*4 + 0)%wmma_k)] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 0)//wmma_k)*K_ko_alignment + k_o_i*K_m + ((mk_o*4 + 0)%wmma_k)]
                                            A_shared_local[((mk_o*4 + 1)//wmma_k)*K_m + ((mk_o*4 + 1)%wmma_k)] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 1)//wmma_k)*K_ko_alignment + k_o_i*K_m + ((mk_o*4 + 1)%wmma_k)]
                                            A_shared_local[((mk_o*4 + 2)//wmma_k)*K_m + ((mk_o*4 + 2)%wmma_k)] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 2)//wmma_k)*K_ko_alignment + k_o_i*K_m + ((mk_o*4 + 2)%wmma_k)]
                                            A_shared_local[((mk_o*4 + 3)//wmma_k)*K_m + ((mk_o*4 + 3)%wmma_k)] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 3)//wmma_k)*K_ko_alignment + k_o_i*K_m + ((mk_o*4 + 3)%wmma_k)]

                                        # rhs L2 -> L1
                                        for nk_o in range(wmma_n*wmma_k//4): # 64
                                            B_shared_local[((nk_o*4 + 0)//wmma_k)*K_m + (nk_o*4 + 0)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 0)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 0)%wmma_k]
                                            B_shared_local[((nk_o*4 + 1)//wmma_k)*K_m + (nk_o*4 + 1)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 1)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 1)%wmma_k]
                                            B_shared_local[((nk_o*4 + 2)//wmma_k)*K_m + (nk_o*4 + 2)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 2)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 2)%wmma_k]
                                            B_shared_local[((nk_o*4 + 3)//wmma_k)*K_m + (nk_o*4 + 3)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 3)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 3)%wmma_k]


                                        param = N,wmma_n,wmma_m,m_o,wmma_k,K_m
                                        T_local = self.llvm_matrix_mad_f32x4_f32x4(A_shared_local,B_shared_local,T_local,param)
                            # tail: calculate the last iteration without A_shared load
                            k_o_o = k_o_o_iters
                            # rhs HBM -> L2
                            nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                            for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                for bs_threadIdx_z in range(block[2]): # 7
                                    for bs_threadIdx_y in range(block[1]): # 4
                                        for bs_threadIdx_x in range(block[0]): # 64
                                            for n_k_fuse_i in range(vec): # 2
                                                n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + bs_threadIdx_z
                                                n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + bs_threadIdx_y
                                                n_k_fuse_o = n_k_fuse_o_o*block[0] + bs_threadIdx_x
                                                n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                n = n_k_fuse//K_ko
                                                k = n_k_fuse%K_ko
                                                left = n*K_ko_alignment + k
                                                right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                B_shared[left] = B[right]

                            for k_o_i in range(k_factor): # 8
                                for m_o in range(M//wmma_m): # 1
                                    # lhs L2 -> L1
                                    for mk_o in range(wmma_m*wmma_k//4): # 64
                                        A_shared_local_1[((mk_o*4 + 0)//wmma_k)*K_m + (mk_o*4 + 0)%wmma_k] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 0)//wmma_k)*K_ko_alignment + k_o_i*K_m + (mk_o*4 + 0)%wmma_k]
                                        A_shared_local_1[((mk_o*4 + 1)//wmma_k)*K_m + (mk_o*4 + 1)%wmma_k] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 1)//wmma_k)*K_ko_alignment + k_o_i*K_m + (mk_o*4 + 1)%wmma_k]
                                        A_shared_local_1[((mk_o*4 + 2)//wmma_k)*K_m + (mk_o*4 + 2)%wmma_k] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 2)//wmma_k)*K_ko_alignment + k_o_i*K_m + (mk_o*4 + 2)%wmma_k]
                                        A_shared_local_1[((mk_o*4 + 3)//wmma_k)*K_m + (mk_o*4 + 3)%wmma_k] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 3)//wmma_k)*K_ko_alignment + k_o_i*K_m + (mk_o*4 + 3)%wmma_k]
                                    # rhs L2 -> L1
                                    for nk_o in range(wmma_n*wmma_k//4): # 64
                                        B_shared_local_1[((nk_o*4 + 0)//wmma_k)*K_m + (nk_o*4 + 0)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 0)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 0)%wmma_k]
                                        B_shared_local_1[((nk_o*4 + 1)//wmma_k)*K_m + (nk_o*4 + 1)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 1)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 1)%wmma_k]
                                        B_shared_local_1[((nk_o*4 + 2)//wmma_k)*K_m + (nk_o*4 + 2)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 2)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 2)%wmma_k]
                                        B_shared_local_1[((nk_o*4 + 3)//wmma_k)*K_m + (nk_o*4 + 3)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 3)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 3)%wmma_k]
                                    param = N,wmma_n,wmma_m,m_o,wmma_k,K_m
                                    T_local = self.llvm_matrix_mad_f32x4_f32x4(A_shared_local_1,B_shared_local_1,T_local,param)

                            # # out L1 -> L2
                            for mn_o in range(wmma_m*wmma_n//4): # 64
                                T_local_shared[threadIdx_y*M*N_alignment + ((mn_o*4 + 0)//wmma_n)*N_alignment + threadIdx_z*N + (mn_o*4 + 0)%wmma_n] = T_local[((mn_o*4 + 0)//wmma_n)*N + (mn_o*4 + 0)%wmma_n]
                                T_local_shared[threadIdx_y*M*N_alignment + ((mn_o*4 + 1)//wmma_n)*N_alignment + threadIdx_z*N + (mn_o*4 + 1)%wmma_n] = T_local[((mn_o*4 + 1)//wmma_n)*N + (mn_o*4 + 1)%wmma_n]
                                T_local_shared[threadIdx_y*M*N_alignment + ((mn_o*4 + 2)//wmma_n)*N_alignment + threadIdx_z*N + (mn_o*4 + 2)%wmma_n] = T_local[((mn_o*4 + 2)//wmma_n)*N + (mn_o*4 + 2)%wmma_n]
                                T_local_shared[threadIdx_y*M*N_alignment + ((mn_o*4 + 3)//wmma_n)*N_alignment + threadIdx_z*N + (mn_o*4 + 3)%wmma_n] = T_local[((mn_o*4 + 3)//wmma_n)*N + (mn_o*4 + 3)%wmma_n]

                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res


    # =============================================================== #
    #                       14 step14
    # =============================================================== #
    def step14(self, param):
        print('code: step14')
        mem = MEM()
        shape, lhs, rhs, res, block_loop, block, wmma, vec, k_factor, alignment = param
        M_ori, N_ori, K_ori = shape
        AS_align = alignment[0]
        BS_align = alignment[1]
        CS_align = alignment[2] # storage_align(64, 128-1, 128)
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

        M_shared = M # 64
        M = M//block[1]
        N_shared = N # 112
        N = N//block[2]
        N_alignment = ((N_shared + CS_align-1)//CS_align)*CS_align
        
        # A_shared = mem.new((M*K), "zero")
        # B_shared = mem.new((N*K), "zero")
        # T_shared = mem.new((M*N), "zero")
        
        for blockIdx_z in range(BLOCK_Z): # 1
            for blockIdx_y in range(BLOCK_Y): # 4
                T_local = mem.new((M*N), "zero") # 16*16 = 256
                # how to reuse A_shared without compute_at
                T_local_shared = mem.new((M_shared*N_alignment), "zero") # 32*256 = 8192
                for blockIdx_x in range(BLOCK_X): # 7
                    for threadIdx_y in range(block[1]): # 2
                        for threadIdx_z in range(block[2]): # 9
                            cf_ko_factor = K//wmma_k//k_factor # 2048//16//2 = 64
                            K_ko = K//cf_ko_factor # 2048//64 = 32
                            K_ko_alignment = ((K_ko + AS_align -1)//AS_align)*AS_align # 32
                            A_shared = mem.new(2*M_shared*K_ko_alignment, "zero") # 2*32*32 = 2048
                            B_shared = mem.new((N_shared*K_ko_alignment), "zero") # 144*32 = 4608
                            K_m = K//cf_ko_factor//k_factor//(M//wmma_m) # 16
                            A_shared_local = mem.new((2*M*K_m), "zero") # 256*2 = 512
                            B_shared_local = mem.new((N*K_m), "zero") # 256
                            # A_shared double buffer
                            A_shared_local_1 = mem.new((2*M*K_m), "zero") # 256*2 = 512
                            # shared_tail
                            B_shared_local_1 = mem.new((N*K_m), "zero") # 256
                            # shared_tail_local_body
                            B_shared_local_2 = mem.new((N*K_m), "zero") # 256
                            # shared_tail_local_tail
                            B_shared_local_3 = mem.new((N*K_m), "zero") # 256
                            # memset
                            T_local = self.dense_broadcast()

                            # shared_head: lhs HBM -> L2 iter0 for double buffer
                            k_o_o = 0
                            upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                            mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                            for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                for as_threadIdx_z in range(block[2]): # 7
                                    for as_threadIdx_y in range(block[1]): # 4
                                        for as_threadIdx_x in range(block[0]): # 64
                                            for m_k_fuse_i in range(vec): # 1
                                                m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                if m_k_fuse_o_o_o < upper_bound:
                                                    m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                    m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                    m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                    m = m_k_fuse//K_ko
                                                    k = m_k_fuse%K_ko
                                                    left = m*K_ko_alignment + k
                                                    right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + k
                                                    A_shared[left] = A[right]
                            # import pdb;pdb.set_trace()
                            # shared_body: use A_shared double buffer
                            k_o_o_iters = K//wmma_k//k_factor - 1 # 2048//16//2 - 1 = 64 - 1 = 63
                            for k_o_o in range(k_o_o_iters): # 63.
                                # lhs HBM -> L2
                                upper_bound = (M_shared*K_ko)//vec//block[0]//block[1] # 16
                                mk_o_o_o = ((upper_bound+block[2]-1)//block[2]) # 3
                                for m_k_fuse_o_o_o_o in range(mk_o_o_o): # 3
                                    for as_threadIdx_z in range(block[2]): # 7
                                        for as_threadIdx_y in range(block[1]): # 4
                                            for as_threadIdx_x in range(block[0]): # 64
                                                for m_k_fuse_i in range(vec): # 1
                                                    m_k_fuse_o_o_o = m_k_fuse_o_o_o_o*block[2] + as_threadIdx_z
                                                    if m_k_fuse_o_o_o < upper_bound:
                                                        m_k_fuse_o_o = m_k_fuse_o_o_o*block[1] + as_threadIdx_y
                                                        m_k_fuse_o = m_k_fuse_o_o*block[0] + as_threadIdx_x
                                                        m_k_fuse = m_k_fuse_o*vec + m_k_fuse_i
                                                        m = m_k_fuse//K_ko
                                                        k = m_k_fuse%K_ko
                                                        # select ping-pong buffer
                                                        left = ((k_o_o+1)%2)*M_shared*K_ko_alignment + m*K_ko_alignment + k
                                                        right = blockIdx_y*M_shared*K + m*K + k_o_o*K_ko + K_ko + k
                                                        A_shared[left] = A[right]
                                # rhs HBM -> L2
                                nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                                for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                    for bs_threadIdx_z in range(block[2]): # 7
                                        for bs_threadIdx_y in range(block[1]): # 4
                                            for bs_threadIdx_x in range(block[0]): # 64
                                                for n_k_fuse_i in range(vec): # 2
                                                    n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + bs_threadIdx_z
                                                    n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + bs_threadIdx_y
                                                    n_k_fuse_o = n_k_fuse_o_o*block[0] + bs_threadIdx_x
                                                    n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                    n = n_k_fuse//K_ko
                                                    k = n_k_fuse%K_ko
                                                    left = n*K_ko_alignment + k
                                                    right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                    B_shared[left] = B[right]
                                # local_head
                                # 1st lhs L2 -> L1
                                k_o_i = 0
                                for mk_o in range(wmma_m*wmma_k//4): # 64
                                    A_shared_local[((mk_o*4 + 0)//wmma_k)*K_m + ((mk_o*4 + 0)%wmma_k)] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 0)//wmma_k)*K_ko_alignment + k_o_i*K_m + ((mk_o*4 + 0)%wmma_k)]
                                    A_shared_local[((mk_o*4 + 1)//wmma_k)*K_m + ((mk_o*4 + 1)%wmma_k)] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 1)//wmma_k)*K_ko_alignment + k_o_i*K_m + ((mk_o*4 + 1)%wmma_k)]
                                    A_shared_local[((mk_o*4 + 2)//wmma_k)*K_m + ((mk_o*4 + 2)%wmma_k)] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 2)//wmma_k)*K_ko_alignment + k_o_i*K_m + ((mk_o*4 + 2)%wmma_k)]
                                    A_shared_local[((mk_o*4 + 3)//wmma_k)*K_m + ((mk_o*4 + 3)%wmma_k)] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 3)//wmma_k)*K_ko_alignment + k_o_i*K_m + ((mk_o*4 + 3)%wmma_k)]
                                # local_body
                                k_o_i_iters = k_factor - 1 # 2-1=1
                                for k_o_i in range(k_o_i_iters): # 1
                                    for m_o in range(M//wmma_m): # 1
                                        # [1,k_o_i_iters] lhs L2 -> L1
                                        for mk_o in range(wmma_m*wmma_k//4): # 64
                                            A_shared_local[((k_o_i+1)%2)*M*K_m + ((mk_o*4 + 0)//wmma_k)*K_m + ((mk_o*4 + 0)%wmma_k)] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 0)//wmma_k)*K_ko_alignment + k_o_i*K_m + K_m + ((mk_o*4 + 0)%wmma_k)]
                                            A_shared_local[((k_o_i+1)%2)*M*K_m + ((mk_o*4 + 1)//wmma_k)*K_m + ((mk_o*4 + 1)%wmma_k)] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 1)//wmma_k)*K_ko_alignment + k_o_i*K_m + K_m + ((mk_o*4 + 1)%wmma_k)]
                                            A_shared_local[((k_o_i+1)%2)*M*K_m + ((mk_o*4 + 2)//wmma_k)*K_m + ((mk_o*4 + 2)%wmma_k)] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 2)//wmma_k)*K_ko_alignment + k_o_i*K_m + K_m + ((mk_o*4 + 2)%wmma_k)]
                                            A_shared_local[((k_o_i+1)%2)*M*K_m + ((mk_o*4 + 3)//wmma_k)*K_m + ((mk_o*4 + 3)%wmma_k)] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 3)//wmma_k)*K_ko_alignment + k_o_i*K_m + K_m + ((mk_o*4 + 3)%wmma_k)]
                                        # rhs L2 -> L1
                                        for nk_o in range(wmma_n*wmma_k//4): # 64
                                            B_shared_local[((nk_o*4 + 0)//wmma_k)*K_m + (nk_o*4 + 0)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 0)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 0)%wmma_k]
                                            B_shared_local[((nk_o*4 + 1)//wmma_k)*K_m + (nk_o*4 + 1)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 1)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 1)%wmma_k]
                                            B_shared_local[((nk_o*4 + 2)//wmma_k)*K_m + (nk_o*4 + 2)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 2)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 2)%wmma_k]
                                            B_shared_local[((nk_o*4 + 3)//wmma_k)*K_m + (nk_o*4 + 3)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 3)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 3)%wmma_k]

                                        param = N,wmma_n,wmma_m,m_o,wmma_k,K_m
                                        T_local = self.llvm_matrix_mad_f32x4_f32x4(A_shared_local[(k_o_i%2)*M*K_m : (k_o_i%2)*M*K_m+M*K_m],B_shared_local,T_local,param)
                                # local_tail
                                k_o_i = k_o_i_iters
                                for m_o in range(M//wmma_m): # 1
                                    # rhs L2 -> L1
                                    for nk_o in range(wmma_n*wmma_k//4): # 64
                                        B_shared_local_1[((nk_o*4 + 0)//wmma_k)*K_m + (nk_o*4 + 0)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 0)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 0)%wmma_k]
                                        B_shared_local_1[((nk_o*4 + 1)//wmma_k)*K_m + (nk_o*4 + 1)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 1)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 1)%wmma_k]
                                        B_shared_local_1[((nk_o*4 + 2)//wmma_k)*K_m + (nk_o*4 + 2)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 2)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 2)%wmma_k]
                                        B_shared_local_1[((nk_o*4 + 3)//wmma_k)*K_m + (nk_o*4 + 3)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 3)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 3)%wmma_k]

                                    param = N,wmma_n,wmma_m,m_o,wmma_k,K_m
                                    T_local = self.llvm_matrix_mad_f32x4_f32x4(A_shared_local[(k_o_i%2)*M*K_m : (k_o_i%2)*M*K_m+M*K_m],B_shared_local_1,T_local,param)
                            # shared_tail: calculate the last iteration without A_shared load
                            k_o_o = k_o_o_iters
                            # rhs HBM -> L2
                            nk_o_o_o = N_shared*K_ko//vec//block[0]//block[1]//block[2]
                            for n_k_fuse_o_o_o_o in range(nk_o_o_o): # 4
                                for bs_threadIdx_z in range(block[2]): # 7
                                    for bs_threadIdx_y in range(block[1]): # 4
                                        for bs_threadIdx_x in range(block[0]): # 64
                                            for n_k_fuse_i in range(vec): # 2
                                                n_k_fuse_o_o_o = n_k_fuse_o_o_o_o*block[2] + bs_threadIdx_z
                                                n_k_fuse_o_o = n_k_fuse_o_o_o*block[1] + bs_threadIdx_y
                                                n_k_fuse_o = n_k_fuse_o_o*block[0] + bs_threadIdx_x
                                                n_k_fuse = n_k_fuse_o*vec + n_k_fuse_i
                                                n = n_k_fuse//K_ko
                                                k = n_k_fuse%K_ko
                                                left = n*K_ko_alignment + k
                                                right = blockIdx_x*N_shared*K + n*K + k_o_o*K_ko + k
                                                B_shared[left] = B[right]
                            # shared_tail_local_head
                            # 1st lhs L2 -> L1
                            k_o_i = 0
                            for mk_o in range(wmma_m*wmma_k//4): # 64
                                A_shared_local_1[((mk_o*4 + 0)//wmma_k)*K_m + (mk_o*4 + 0)%wmma_k] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 0)//wmma_k)*K_ko_alignment + k_o_i*K_m + (mk_o*4 + 0)%wmma_k]
                                A_shared_local_1[((mk_o*4 + 1)//wmma_k)*K_m + (mk_o*4 + 1)%wmma_k] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 1)//wmma_k)*K_ko_alignment + k_o_i*K_m + (mk_o*4 + 1)%wmma_k]
                                A_shared_local_1[((mk_o*4 + 2)//wmma_k)*K_m + (mk_o*4 + 2)%wmma_k] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 2)//wmma_k)*K_ko_alignment + k_o_i*K_m + (mk_o*4 + 2)%wmma_k]
                                A_shared_local_1[((mk_o*4 + 3)//wmma_k)*K_m + (mk_o*4 + 3)%wmma_k] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 3)//wmma_k)*K_ko_alignment + k_o_i*K_m + (mk_o*4 + 3)%wmma_k]
                            # shared_tail_local_body
                            k_o_i_iters = k_factor - 1 # 2-1=1
                            for k_o_i in range(k_o_i_iters): # 2
                                for m_o in range(M//wmma_m): # 1
                                    # lhs L2 -> L1
                                    for mk_o in range(wmma_m*wmma_k//4): # 64
                                        A_shared_local_1[((k_o_i+1)%2)*M*K_m + ((mk_o*4 + 0)//wmma_k)*K_m + (mk_o*4 + 0)%wmma_k] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 0)//wmma_k)*K_ko_alignment + k_o_i*K_m + K_m + (mk_o*4 + 0)%wmma_k]
                                        A_shared_local_1[((k_o_i+1)%2)*M*K_m + ((mk_o*4 + 1)//wmma_k)*K_m + (mk_o*4 + 1)%wmma_k] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 1)//wmma_k)*K_ko_alignment + k_o_i*K_m + K_m + (mk_o*4 + 1)%wmma_k]
                                        A_shared_local_1[((k_o_i+1)%2)*M*K_m + ((mk_o*4 + 2)//wmma_k)*K_m + (mk_o*4 + 2)%wmma_k] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 2)//wmma_k)*K_ko_alignment + k_o_i*K_m + K_m + (mk_o*4 + 2)%wmma_k]
                                        A_shared_local_1[((k_o_i+1)%2)*M*K_m + ((mk_o*4 + 3)//wmma_k)*K_m + (mk_o*4 + 3)%wmma_k] = A_shared[(k_o_o%2)*M_shared*K_ko_alignment + threadIdx_y*M*K_ko_alignment + ((mk_o*4 + 3)//wmma_k)*K_ko_alignment + k_o_i*K_m + K_m + (mk_o*4 + 3)%wmma_k]
                                    # rhs L2 -> L1
                                    for nk_o in range(wmma_n*wmma_k//4): # 64
                                        B_shared_local_2[((nk_o*4 + 0)//wmma_k)*K_m + (nk_o*4 + 0)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 0)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 0)%wmma_k]
                                        B_shared_local_2[((nk_o*4 + 1)//wmma_k)*K_m + (nk_o*4 + 1)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 1)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 1)%wmma_k]
                                        B_shared_local_2[((nk_o*4 + 2)//wmma_k)*K_m + (nk_o*4 + 2)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 2)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 2)%wmma_k]
                                        B_shared_local_2[((nk_o*4 + 3)//wmma_k)*K_m + (nk_o*4 + 3)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 3)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 3)%wmma_k]
                                    param = N,wmma_n,wmma_m,m_o,wmma_k,K_m
                                    T_local = self.llvm_matrix_mad_f32x4_f32x4(A_shared_local_1[(k_o_i%2)*M*K_m : (k_o_i%2)*M*K_m+M*K_m],B_shared_local_2,T_local,param)
                            # shared_tail_local_tail
                            k_o_i = k_o_i_iters
                            # rhs L2 -> L1
                            for nk_o in range(wmma_n*wmma_k//4): # 64
                                B_shared_local_3[((nk_o*4 + 0)//wmma_k)*K_m + (nk_o*4 + 0)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 0)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 0)%wmma_k]
                                B_shared_local_3[((nk_o*4 + 1)//wmma_k)*K_m + (nk_o*4 + 1)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 1)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 1)%wmma_k]
                                B_shared_local_3[((nk_o*4 + 2)//wmma_k)*K_m + (nk_o*4 + 2)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 2)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 2)%wmma_k]
                                B_shared_local_3[((nk_o*4 + 3)//wmma_k)*K_m + (nk_o*4 + 3)%wmma_k] = B_shared[threadIdx_z*N*K_ko_alignment + ((nk_o*4 + 3)//wmma_k)*K_ko_alignment + k_o_i*K_m + (nk_o*4 + 3)%wmma_k]
                            param = N,wmma_n,wmma_m,m_o,wmma_k,K_m
                            T_local = self.llvm_matrix_mad_f32x4_f32x4(A_shared_local_1[(k_o_i%2)*M*K_m : (k_o_i%2)*M*K_m+M*K_m],B_shared_local_3,T_local,param)
                            # # out L1 -> L2
                            for mn_o in range(wmma_m*wmma_n//4): # 64
                                T_local_shared[threadIdx_y*M*N_alignment + ((mn_o*4 + 0)//wmma_n)*N_alignment + threadIdx_z*N + (mn_o*4 + 0)%wmma_n] = T_local[((mn_o*4 + 0)//wmma_n)*N + (mn_o*4 + 0)%wmma_n]
                                T_local_shared[threadIdx_y*M*N_alignment + ((mn_o*4 + 1)//wmma_n)*N_alignment + threadIdx_z*N + (mn_o*4 + 1)%wmma_n] = T_local[((mn_o*4 + 1)//wmma_n)*N + (mn_o*4 + 1)%wmma_n]
                                T_local_shared[threadIdx_y*M*N_alignment + ((mn_o*4 + 2)//wmma_n)*N_alignment + threadIdx_z*N + (mn_o*4 + 2)%wmma_n] = T_local[((mn_o*4 + 2)//wmma_n)*N + (mn_o*4 + 2)%wmma_n]
                                T_local_shared[threadIdx_y*M*N_alignment + ((mn_o*4 + 3)//wmma_n)*N_alignment + threadIdx_z*N + (mn_o*4 + 3)%wmma_n] = T_local[((mn_o*4 + 3)//wmma_n)*N + (mn_o*4 + 3)%wmma_n]

                    # # out L2 -> HBM
                    C_o_o_o = (M_shared*N_shared)//vec//block[0]//block[1]//block[2]
                    for m_n_fused_o_o_o_o in range(C_o_o_o): # 2
                        for threadIdx_z in range(block[2]): # 7
                            for threadIdx_y in range(block[1]): # 4
                                for threadIdx_x in range(block[0]): # 64
                                    param = m_n_fused_o_o_o_o,block,threadIdx_z,threadIdx_y,threadIdx_x,blockIdx_y,M_shared,N_ori,N_shared,blockIdx_x,N_alignment
                                    self.ramp(T, T_local_shared, vec, param)
        res = np.reshape(T, (M_ori, N_ori))
        return res
