import sys
sys.path.append('./python')

import needle as ndl
import numpy as np
from needle import nn
from matplotlib import pyplot as plt


def prepare_training_dataset(A, mu):
  # total number of sample data to generated
  num_sample = 3200
  lhs = np.random.normal(0, 1, (num_sample, 2))
  data = lhs @ A + mu
  plt.scatter(data[:,0], data[:,1], color="blue", label="real data")
  plt.legend()
  plt.savefig('./training_dataset.png')
  return data

def sample_G(model_G, num_samples):
  Z = ndl.Tensor(np.random.normal(0, 1, (num_samples, 2)))
  fake_X = model_G(Z)
  # return fake_X.numpy()
  return fake_X

def Generator_network_G(data):
  import pdb;pdb.set_trace()
  linear_layer = nn.Linear(2, 2)
  model_G = nn.Sequential(linear_layer)
  fake_data_init = sample_G(model_G, 3200)
  plt.scatter(data[:,0], data[:,1], color="blue", label="real data")
  plt.scatter(fake_data_init[:,0], fake_data_init[:,1], color="red", label="G(z) at init")
  plt.legend()
  plt.savefig('./network_G.png')
  return fake_data_init, model_G

def Discriminator_D():
  model_D = nn.Sequential(
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
  )
  loss_D = nn.SoftmaxLoss()
  return model_D, loss_D

def Generator_update(model_G):
  opt_G = ndl.optim.Adam(model_G.parameters(), lr=0.01)
  return opt_G

def update_G(Z, model_G, model_D, loss_D, opt_G):
  fake_X = model_G(Z)
  fake_Y = model_D(fake_X)
  batch_size = Z.shape[0]
  ones = ndl.ones(batch_size, dtype="int32")
  loss = loss_D(fake_Y, ones)
  loss.backward()
  opt_G.step()
  return

def Discriminator_update():
  opt_D = ndl.optim.Adam(model_D.parameters(), lr=0.01)
  return opt_D
 
def update_D(X, Z, model_G, model_D, loss_D, opt_D):
  fake_X = model_G(Z).detach()
  fake_Y = model_D(fake_X)
  real_Y = model_D(X)
  assert X.shape[0] == Z.shape[0]
  batch_size = X.shape[0]
  ones = ndl.ones(batch_size, dtype="int32")
  zeros = ndl.zeros(batch_size, dtype="int32")
  loss = loss_D(real_Y, ones) + loss_D(fake_Y, zeros)
  loss.backward()
  opt_D.step()
  return

def train_gan(model, data, batch_size, num_epochs):
  model_G, opt_G, model_D, loss_D, opt_D, fake_data_init = model
  assert data.shape[0] % batch_size == 0
  for epoch in range(num_epochs):
    begin = (batch_size * epoch) % data.shape[0]
    X = data[begin: begin+batch_size, :]
    Z = np.random.normal(0, 1, (batch_size, 2))
    X = ndl.Tensor(X)
    Z = ndl.Tensor(Z)
    update_D(X, Z, model_G, model_D, loss_D, opt_D) 
    update_G(Z, model_G, model_D, loss_D, opt_G)
  fake_data_trained = sample_G(model_G, 3200)
  plt.scatter(data[:,0], data[:,1], color="blue", label="real data")
  plt.scatter(fake_data_init[:,0], fake_data_init[:,1], color="red", label="G(z) at init")
  plt.scatter(fake_data_trained[:,0], fake_data_trained[:,1], color="pink", label="G(z) trained")
  plt.legend()
  plt.savefig('./fake_data_trained.png')
  return

def Inspect_the_trained_generator(model_G, A, mu):
  gA, gmu = model_G.parameters()
  gA = gA.numpy()
  covariance_gA = gA.T @ gA
  covariance_A = A.T @A
  print(A, gA)
  print(mu, gmu)
  print(covariance_A, covariance_gA)
  return

def Modularizing_GAN():
  class GANLoss:
    def __init__(self, model_D, opt_D):
      self.model_D = model_D
      self.opt_D = opt_D
      self.loss_D = nn.SoftmaxLoss()

    def _update_D(self, real_X, fake_X):
      real_Y = self.model_D(real_X)
      fake_Y = self.model_D(fake_X.detach())
      batch_size = real_X.shape[0]
      ones = ndl.ones(batch_size, dtype="int32")
      zeros = ndl.zeros(batch_size, dtype="int32")
      loss = self.loss_D(real_Y, ones) + self.loss_D(fake_Y, zeros)
      loss.backward()
      self.opt_D.step()

    def forward(self, fake_X, real_X):
      self._update_D(real_X, fake_X)
      fake_Y = self.model_D(fake_X)
      batch_size = real_X.shape[0]
      ones = ndl.ones(batch_size, dtype="int32")
      loss = self.loss_D(fake_Y, ones)
      return loss
  model_G = nn.Sequential(nn.Linear(2, 2))
  opt_G = ndl.optim.Adam(model_G.parameters(), lr = 0.01)

  model_D = nn.Sequential(
      nn.Linear(2, 20),
      nn.ReLU(),
      nn.Linear(20, 10),
      nn.ReLU(),
      nn.Linear(10, 2)
  )
  opt_D = ndl.optim.Adam(model_D.parameters(), lr=0.01)
  gan_loss = GANLoss(model_D, opt_D)

  def train_gan(data, batch_size, num_epochs):
      assert data.shape[0] % batch_size == 0
      for epoch in range(num_epochs):
          begin = (batch_size * epoch) % data.shape[0]
          X = data[begin: begin+batch_size, :]
          Z = np.random.normal(0, 1, (batch_size, 2))
          X = ndl.Tensor(X)
          Z = ndl.Tensor(Z)
          fake_X = model_G(Z)
          loss = gan_loss.forward(fake_X, X)
          loss.backward()
          opt_G.step()

  train_gan(data, 32, 2000)
  fake_data_trained = sample_G(model_G, 3200)

  plt.scatter(data[:,0], data[:,1], color="blue", label="real data")
  plt.scatter(fake_data_init[:,0], fake_data_init[:,1], color="red", label="G(z) at init")
  plt.scatter(fake_data_trained[:,0], fake_data_trained[:,1], color="pink", label="G(z) trained")
  plt.legend()
  plt.savefig('./Modularizing_GAN.png')
  return

if __name__ == "__main__":
  A = np.array([[1, 2], [-0.2, 0.5]])
  mu = np.array([2, 1])
  data = prepare_training_dataset(A, mu)
  fake_data_init, model_G = Generator_network_G(data)
  # model_D, loss_D = Discriminator_D()
  # opt_G = Generator_update(model_G)
  # opt_D = Discriminator_update(model_D)
  # model = model_G, opt_G, model_D, loss_D, opt_D, fake_data_init
  # train_gan(model, data, 32, 2000)
  # Inspect_the_trained_generator(model_G, A, mu)
  # Modularizing_GAN(fake_data_init)

