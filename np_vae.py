import time
import numpy as np
import mxnet as mx
from tqdm import tqdm, tqdm_notebook
from mxnet import nd, autograd, gluon
from mxnet import np, npx
from mxnet.gluon import nn
import matplotlib.pyplot as plt
from distributions import Normal
npx.set_np()
data_ctx = mx.cpu()
model_ctx = mx.gpu(0)


class VAE(gluon.HybridBlock):
    def __init__(self, n_hidden=256, n_latent=2, n_layers=1, n_output=784, act_type='relu', **kwargs):
      self.soft_zero = 1e-10
      self.n_latent = n_latent
      self.output = None
      self.mu = None
      super(VAE, self).__init__(**kwargs)
      with self.name_scope():
          self.encoder = nn.HybridSequential(prefix='encoder')
          for _ in range(n_layers):
              self.encoder.add(nn.Dense(n_hidden, activation=act_type))
          self.encoder.add(nn.Dense(n_latent*2, activation=None))

          self.decoder = nn.HybridSequential(prefix='decoder')
          for _ in range(n_layers):
              self.decoder.add(nn.Dense(n_hidden, activation=act_type))
          self.decoder.add(nn.Dense(n_output, activation='sigmoid'))

    def hybrid_forward(self, F, x):
      h = self.encoder(x)
      loc_scale = F.np.split(h, 2, 1)
      loc = loc_scale[0]
      log_variance = loc_scale[1]
      scale = F.np.exp(0.5 * log_variance)
      self.loc = loc
      mvn = Normal(loc, scale, F)
      y = self.decoder(mvn.sample())
      self.output = y

      KL = 0.5 * F.np.sum(1 + log_variance - loc ** 2 - F.np.exp(log_variance), axis=1)
      logloss = F.np.sum(x * F.np.log(y+self.soft_zero) + (1-x)
                         * F.np.log(1-y+self.soft_zero), axis=1)
      loss = -logloss-KL
      return loss


def load_data(batch_size):
  mnist_train = gluon.data.vision.MNIST(train=True)
  mnist_test = gluon.data.vision.MNIST(train=False)
  num_worker = 4
  transformer = gluon.data.vision.transforms.ToTensor()
  return (gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                batch_size, shuffle=True,
                                num_workers=num_worker),
          gluon.data.DataLoader(mnist_test.transform_first(transformer),
                                batch_size, shuffle=False,
                                num_workers=num_worker))
                                 


def train(net, n_epoch, print_period, train_iter, test_iter):
  net.collect_params().initialize(mx.init.Xavier(), ctx=model_ctx)
  net.hybridize()
  trainer = gluon.Trainer(net.collect_params(), 'adam',
                          {'learning_rate': .001})
  training_loss = []
  validation_loss = []
  for epoch in tqdm_notebook(range(n_epoch), desc='epochs'):
      epoch_loss = 0
      epoch_val_loss = 0

      n_batch_train = 0
      for batch in train_iter:
          n_batch_train += 1
          data = batch[0].as_in_context(model_ctx).reshape(-1, 28 * 28)
          with autograd.record():
              loss = net(data)
          loss.backward()
          trainer.step(data.shape[0])
          epoch_loss += np.mean(loss)

      n_batch_val = 0
      for batch in test_iter:
          n_batch_val += 1
          data = batch[0].as_in_context(model_ctx).reshape(-1, 28 * 28)
          loss = net(data)
          epoch_val_loss += np.mean(loss)

      epoch_loss /= n_batch_train
      epoch_val_loss /= n_batch_val

      training_loss.append(epoch_loss)
      validation_loss.append(epoch_val_loss)

      if epoch % max(print_period, 1) == 0:
          print('Epoch{}, Training loss {:.2f}, Validation loss {:.2f}'.format(
              epoch, float(epoch_loss), float(epoch_val_loss)))


if __name__ == "__main__":
  n_hidden = 400
  n_latent = 2
  n_layers = 2  # num of dense layers in encoder and decoder respectively
  n_output = 784
  batch_size = 128
  model_prefix = 'vae_gluon_{}d{}l{}h.params'.format(
      n_latent, n_layers, n_hidden)
  net = VAE(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
            n_output=n_output)
  n_epoch = 50
  print_period = n_epoch // 10
  train_set, test_set = load_data(batch_size)
  start = time.time()
  train(net, n_epoch, print_period, train_set, test_set)
  end = time.time()
  print('Time elapsed: {:.2f}s'.format(end - start))
