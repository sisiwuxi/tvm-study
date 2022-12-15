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
  # X[100,64], mask[100,100], W_KQV[64,192], W_out[64,64]
  # X@W_KQV[100,192], K[100,64], Q[100,64], V[100,64]
  # keys, queries, values
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
  T = 5
  # attn_mask
  M = torch.triu(-float("inf")*torch.ones(T,T),1)
  print(M)

  T, d = 100, 64
  # MultiheadAttention(
  #   (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=False)
  # )
  attn = nn.MultiheadAttention(d, 1, bias=False, batch_first=True)
  # attn = nn.MultiheadAttention(embed_dim=d, num_heads=1, bias=False, batch_first=True)
  # attn = nn.MultiheadAttention(d, 1, bias=False)
  # M[100,100]
  M = torch.triu(-float("inf")*torch.ones(T,T),1)
  # X[1,100,64]
  X = torch.randn(1,T,d)
  Y_, A_ = attn(X,X,X)
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
  T, d = 100, 64
  import pdb;pdb.set_trace()
  attn = nn.MultiheadAttention(d, 1, bias=False, batch_first=True)
  M = torch.triu(-float("inf")*torch.ones(T,T),1)
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
  return (attn@V).swapaxes(1,2).reshape(N,T,d) @ W_out, attn

def Multihead_attention():
  heads = 4
  N = 10
  T, d = 100, 64
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
  N = 10
  T, d = 100, 64
  M = torch.triu(-float("inf")*torch.ones(T,T),1)
  X = torch.randn(N,T,d)  
  trans = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128, dropout=0.0, batch_first=True)
  trans.linear1.bias.data.zero_()
  trans.linear2.bias.data.zero_()
  Y_ = trans(X, M)
  Y = transformer(X.numpy(), M.numpy(), heads,
                  trans.self_attn.in_proj_weight.detach().numpy().T, 
                  trans.self_attn.out_proj.weight.detach().numpy().T,
                  trans.linear1.weight.detach().numpy().T,
                  trans.linear2.weight.detach().numpy().T,
                  trans.norm1.eps)
  print(np.linalg.norm(Y - Y_.detach().numpy()))
  return

if __name__ == "__main__":
  # Implementing_self_attention()
  # Minibatching_with_batch_matrix_multiply()
  # Multihead_attention()
  Transformer_Block()