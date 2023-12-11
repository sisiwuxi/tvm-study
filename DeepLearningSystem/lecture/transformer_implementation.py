import numpy as np
import torch
import torch.nn as nn

def softmax(Z):
  # rule
  #   1) The predicted probability is a non-negative number: exp(x)
  #   2) The sum of the probabilities of various predicted results is equal to 1
  # s = exp(A) / sum(exp(A))
  #   = (exp(A)/exp(max)) / (sum(exp(A))/exp(max))
  #   = exp(A-max) / sum(exp(A-max))
  Z = np.exp(Z - Z.max(axis=-1, keepdims=True))
  return Z / Z.sum(axis=-1, keepdims=True)
    
def self_attention(X, mask, W_KQV, W_out):
  # import pdb;pdb.set_trace()
  # X[100,64], mask[100,100], W_KQV[64,192], W_out[64,64]
  # X@W_KQV[100,192] --split--> K[100,64], Q[100,64], V[100,64]
  # KQV: keys, queries, values
  K,Q,V = np.split(X@W_KQV, 3, axis=-1)
  # Q.swapaxes(-1,-2)[64,100], K@Q.swapaxes(-1,-2)[100,100]
  # d = X.shape[-1] = 64, reduce dimension, sqrt(d) = 8
  # (K@Q.swapaxes(-1,-2) / np.sqrt(X.shape[-1]))[100,100]
  # (K@Q.swapaxes(-1,-2) / np.sqrt(X.shape[-1]) + mask)[100,100]
  # attn[100,100]
  attn = softmax(K@Q.swapaxes(-1,-2) / np.sqrt(X.shape[-1]) + mask)
  # attn@V[100,64]
  # attn@V@W_out[100,64]
  return attn@V@W_out, attn

def Implementing_self_attention():
  # T = 5
  # M = torch.triu(-float("inf")*torch.ones(T,T),1) # attn_mask
  # print(M)
  # T = 1 after KV_Cache
  T, d = 100, 64 # d: embed_dim
  # MultiheadAttention(
  #   (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=False)
  # )
  attn = nn.MultiheadAttention(d, 1, bias=False, batch_first=True)
  # attn = nn.MultiheadAttention(embed_dim=d, num_heads=1, bias=False, batch_first=True)
  # attn = nn.MultiheadAttention(d, 1, bias=False)
  M = torch.triu(-float("inf")*torch.ones(T,T),1) # attn_mask[100,100]
  # X[1,100,64]
  X = torch.randn(1,T,d)
  # Y_, A_ = attn(X,X,X)
  # Y_[1,100,64], A_[1,100,100]
  Y_, A_ = attn(X,X,X, attn_mask=M)
  # Y[100,64], A_[100,100]
  Y, A = self_attention(X[0].numpy(), M.numpy(), 
                        attn.in_proj_weight.detach().numpy().T,
                        attn.out_proj.weight.detach().numpy().T)
  print(np.linalg.norm(A - A_[0].detach().numpy()))
  print(np.linalg.norm(Y - Y_[0].detach().numpy()))
  return

def Minibatching_with_batch_matrix_multiply():
  # illustration of batch matmul
  B = np.random.randn(10,3,5,4)
  C = np.random.randn(10,3,4,3)
  print((B@C).shape)
  N = 10
  T, d = 100, 64 # d: embed_dim
  # import pdb;pdb.set_trace()
  attn = nn.MultiheadAttention(d, 1, bias=False, batch_first=True)
  M = torch.triu(-float("inf")*torch.ones(T,T),1) # attn_mask[100,100]
  X = torch.randn(N,T,d)
  Y_, A_ = attn(X,X,X, attn_mask=M)
  Y, A = self_attention(X.numpy(), M.numpy(),
                        attn.in_proj_weight.detach().numpy().T, 
                        attn.out_proj.weight.detach().numpy().T)
  print(np.linalg.norm(A - A_.detach().numpy()))
  print(np.linalg.norm(Y - Y_.detach().numpy()))
  return

def multihead_attention(X, mask, heads, W_KQV, W_out):
  N,T,d = X.shape
  K,Q,V = np.split(X@W_KQV, 3, axis=-1)
  K,Q,V = [a.reshape(N,T,heads,d//heads).swapaxes(1,2) for a in (K,Q,V)]
  
  attn = softmax(K@Q.swapaxes(-1,-2) / np.sqrt(d//heads) + mask)
  import pdb;pdb.set_trace()
  return (attn@V).swapaxes(1,2).reshape(N,T,d) @ W_out, attn

def Multihead_attention():
  heads = 4
  N = 10
  T, d = 100, 64 # d: embed_dim
  attn = nn.MultiheadAttention(d, heads, bias=False, batch_first=True)
  M = torch.triu(-float("inf")*torch.ones(T,T),1)
  X = torch.randn(N,T,d)
  Y_, A_ = attn(X,X,X, attn_mask=M)
  Y, A = multihead_attention(X.numpy(), M.numpy(), 4,
                            attn.in_proj_weight.detach().numpy().T, 
                            attn.out_proj.weight.detach().numpy().T)
  print(A_.shape)
  print(A.shape)
  print(np.linalg.norm(Y - Y_.detach().numpy()))
  print(np.linalg.norm(A.mean(1) - A_.detach().numpy()))
  return

def layer_norm(Z, eps):
  return (Z - Z.mean(axis=-1, keepdims=True)) / np.sqrt(Z.var(axis=-1, keepdims=True) + eps)
    
def relu(Z):
  return np.maximum(Z,0)

def transformer(X, mask, heads, W_KQV, W_out, W_ff1, W_ff2, eps):
  Z = layer_norm(multihead_attention(X, mask, heads, W_KQV, W_out)[0] + X, eps)
  return layer_norm(Z + relu(Z@W_ff1)@W_ff2, eps)

def Transformer_Block():
  heads = 4
  N = 10 #
  T, d = 100, 64 # d: embed_dim
  # import pdb;pdb.set_trace()
  M = torch.triu(-float("inf")*torch.ones(T,T),1) # attn_mask[100,100]
  X = torch.randn(N,T,d) # X[10, 100, 64]
  trans = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128, dropout=0.0, batch_first=True)
  """
    TransformerEncoderLayer(
      (self_attn): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
      )
      (linear1): Linear(in_features=64, out_features=128, bias=True)
      (dropout): Dropout(p=0.0, inplace=False)
      (linear2): Linear(in_features=128, out_features=64, bias=True)
      (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (dropout1): Dropout(p=0.0, inplace=False)
      (dropout2): Dropout(p=0.0, inplace=False)
    )
  """
  trans.linear1.bias.data.zero_() # trans.linear1.bias.data[128]
  trans.linear2.bias.data.zero_() # trans.linear2.bias.data[64]
  Y_ = trans(X, M) # Y_[10, 100, 64]
  Y = transformer(X.numpy(), M.numpy(), heads,
                  trans.self_attn.in_proj_weight.detach().numpy().T, # trans.self_attn.in_proj_weight[192,64]
                  trans.self_attn.out_proj.weight.detach().numpy().T, # trans.self_attn.out_proj.weight[64,64]
                  trans.linear1.weight.detach().numpy().T, # trans.linear1.weight[128,64]
                  trans.linear2.weight.detach().numpy().T, # trans.linear2.weight[64,128]
                  trans.norm1.eps) # 1e-05
  # Y[10, 100, 64]
  print(np.linalg.norm(Y - Y_.detach().numpy()))
  return

def page_attention():
  # https://zhuanlan.zhihu.com/p/668736097
  # https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cu
  # dim3 grid(num_heads, num_seqs)
  # dim3 block(NUM_THREADS) # 128
  # kernel parameters
  #   out[num_seqs, num_heads, head_size]
  #   q[num_seqs, num_heads, head_size]
  #   k_cache[num_blocks, num_kv_heads, head_size/x, block_size, x] # X represents a vectorized size, float16->16, 16/sizeof(float16)=8
  #   v_cache[num_blocks, num_kv_heads, head_size, block_size]
  #   head_mapping[num_heads] # used for MQA,GQA，which KV_head to be used
  #   block_tables[num_seqs, max_num_blocks_per_seq] block_tables mapping table, indicate which blocks each sequence maps to
  #   context_lens[num_seqs] used for variable-length
  # constant definition
  #   THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1) # max(64/16,1)
  #   NUM_TOKENS_PER_THREAD_GROUP = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE # (16+63)/64=1 represents the number of tokens 1 thread_group will handled
  #   NUM_WARPS # the number of warps in 1 threadblock
  #   VEC_SIZE # vectorize size, make sure each thread_group get 16bytes per time, MAX(16/(THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  #   NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE # 表示每个thread要负责多少个数据计算
  #   NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE # 表示每个thread负责的数据经过向量化后, 一共有多少个vec
  #   V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE) # 每个thread一次性读取16bytes
  #   NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE # 对于v_cache[head_size, block_size]，表示一行需要几个V_VEC
  #   NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW # 表示一个warp可以处理多少行
  #   NUM_ROWS_PER_THREAD # 表示每个thread需要负责多少行
  # paged_attention_kernel
  #   part1 加载Query: 根据前面得到每个线程要处理的vector个数, 以及thread_group_idx进行偏移, 获取数据
  #   part2 申请shared_memory
  #     一部分用于存储QK结果来做softmax: shared_mem
  #     另一部分是给blockReduce的smem使用: red_smem
  #   part3 提前偏移block tables等参数
  #   part4 循环计算QK
  #   logits dot V_Cache
  #   最终结果更新
  #   extend
  #     https://github.com/vllm-project/vllm/issues/421
  #     https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/issues/421

  return

if __name__ == "__main__":
  # Implementing_self_attention()
  # Minibatching_with_batch_matrix_multiply()
  # Multihead_attention()
  Transformer_Block()
  # 