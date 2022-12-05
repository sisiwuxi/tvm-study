import torch
import torch.nn as nn
import numpy as np

model = nn.LSTMCell(20,100)
print(model.weight_hh.shape)
print(model.weight_ih.shape)

def sigmoid(x):
    return 1/(1+np.exp(-x))

# single lstm cell
def lstm_cell(x, h, c, W_hh, W_ih, b):
  # use single bias instead of b_hi, b_hf, b_hg, b_ho
  i,f,g,o = np.split(W_ih@x + W_hh@h + b, 4)
  i,f,g,o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
  # cell state
  c_out = f*c + i*g
  # hidden state
  h_out = o * np.tanh(c_out)
  return h_out, c_out

def single_lstm_cell():
  # golden
  x = np.random.randn(1,20).astype(np.float32)
  h0 = np.random.randn(1,100).astype(np.float32)
  c0 = np.random.randn(1,100).astype(np.float32)
  h_, c_ = model(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))

  # self lstm
  h, c = lstm_cell(x[0], h0[0], c0[0], 
                  model.weight_hh.detach().numpy(), 
                  model.weight_ih.detach().numpy(), 
                  (model.bias_hh + model.bias_ih).detach().numpy())

  print(np.linalg.norm(h_.detach().numpy() - h), 
        np.linalg.norm(c_.detach().numpy() - c))
  return

# full_sequence_lstm
def lstm(X, h, c, W_hh, W_ih, b):
  H = np.zeros((X.shape[0], h.shape[0]))
  for t in range(X.shape[0]):
      h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
      H[t,:] = h
  return H, c

def full_sequence_lstm():
  model = nn.LSTM(20, 100, num_layers = 1)
  X = np.random.randn(50,20).astype(np.float32)
  h0 = np.random.randn(1,100).astype(np.float32)
  c0 = np.random.randn(1,100).astype(np.float32)
  # golden
  H_, (hn_, cn_) = model(torch.tensor(X)[:,None,:], 
                        (torch.tensor(h0)[:,None,:], 
                          torch.tensor(c0)[:,None,:]))
  H, cn = lstm(X, h0[0], c0[0], 
              model.weight_hh_l0.detach().numpy(), 
              model.weight_ih_l0.detach().numpy(), 
              (model.bias_hh_l0 + model.bias_ih_l0).detach().numpy())
  print(np.linalg.norm(H - H_[:,0,:].detach().numpy()),
        np.linalg.norm(cn - cn_[0,0,:].detach().numpy()))
  return

def batch_lstm_cell(x, h, c, W_hh, W_ih, b):
  i,f,g,o = np.split(x@W_ih + h@W_hh + b, 4, axis=1)
  i,f,g,o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
  c_out = f*c + i*g
  h_out = o * np.tanh(c_out)
  return h_out, c_out

def batching_lstm(X, h, c, W_hh, W_ih, b):
  H = np.zeros((X.shape[0], X.shape[1], h.shape[1]))
  for t in range(X.shape[0]):
    h, c = batch_lstm_cell(X[t], h, c, W_hh, W_ih, b)
    H[t] = h
  return H, c

def batching():
  model = nn.LSTM(20, 100, num_layers = 1)
  # X[B,T,n]
  X = np.random.randn(50,80,20).astype(np.float32)
  h0 = np.random.randn(80,100).astype(np.float32)
  c0 = np.random.randn(80,100).astype(np.float32)

  H_, (hn_, cn_) = model(torch.tensor(X), (torch.tensor(h0)[None,:,:], 
    torch.tensor(c0)[None,:,:]))
                          
  H, cn = batching_lstm(X, h0, c0,
              model.weight_hh_l0.detach().numpy().T, 
              model.weight_ih_l0.detach().numpy().T, 
              (model.bias_hh_l0 + model.bias_ih_l0).detach().numpy())
  print(np.linalg.norm(H - H_.detach().numpy()),
      np.linalg.norm(cn - cn_[0].detach().numpy()))
  return

def train_lstm(X, h0, c0, Y, W_hh, W_ih, b, opt):
  H, cn = lstm(X, h0, c0, parameters)
  l = loss(H, Y)
  l.backward()
  opt.step()
  return

def train_deep_lstm(X, h0, c0, Y, W_hh, W_ih, b, opt):
  H = X
  depth = len(W_hh)
  for d in range(depth):
      H, cn = lstm(H, h0[d], c0[d], W_hh[d], W_ih[d], b[d])
      h0[d] = H[-1].detach().copy()
      c0[d] = cn.detach().copy()
  l = loss(H, Y)
  l.backward()
  opt.step()
  return

def Hidden_repackaging():
  h0, c0 = zeros()
  for i in range(sequence_len//blcok_size):
    h0, c0 = train_deep_lstm(X[i*blcok_size:(i+1)blcok_size], h0, c0,
                             Y[i*blcok_size:(i+1)blcok_size],
                             W_hh, W_ih, b, opt)
  return


def train_lstm(X, Y, h0, c0, parameters):
  H, cn = lstm(X, h0, c0, parameters)
  l = loss(H, Y)
  l.backward()
  opt.step()
  return

def train_deep_lstm(X, Y, h0, c0, parameters):
  H = X
  for i in range(depth):
      H, cn = lstm(H, h0[i], c0[i], parameters[i])
  l = loss(H, Y)
  l.backward()
  opt.step()
  return

def Long_sequences_and_truncated_BPTT():
  # x_1,x_2,...,x_10000
  # x_1,x_2,...,x_100,|   ...,x_10000
  # create RNN just with the first hundred time steps with h0=0
  # x_1,x_2,...,x_100,|x_101,...,x_200,|   ...,x_10000
  # this sequence h0 = h100
  # not creating one big long compute graph through this
  for i in range(0,X.shape[0],BLOCK_SIZE):
    h0, c0 = zeros()
    train_lstm(X[i:i+BLOCK_SIZE], Y[i:i+BLOCK_SIZE], 
               h0, c0, parameters)
  return

def train_lstm(X, Y, h0, c0, parameters):
    H, cn = lstm(X, h0, c0, parameters)
    l = loss(H, Y)
    l.backward()
    opt.step()
    return H[-1].data, cn.data

def Hidden_repackaging():
  h0, c0 = zeros()
  for i in range(0,X.shape[0],BLOCK_SIZE):
    h0, c0 = train_lstm(X[i:i+BLOCK_SIZE], Y[i:i+BLOCK_SIZE], h0, c0, parameters)
  return

def train():
  # time by time replace layer to layer
  opt = optim.SGD([W_hh, W_ih, b])
  train_lstm(X, h0, c0, Y, W_hh, W_ih, b, opt)
  return

if __name__ == "__main__":
  single_lstm_cell()
  full_sequence_lstm()
  batching()
  train()
  Long_sequences_and_truncated_BPTT()
  Hidden_repackaging()