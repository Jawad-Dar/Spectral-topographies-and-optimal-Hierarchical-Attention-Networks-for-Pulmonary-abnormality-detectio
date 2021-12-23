import numpy as np

def img2col_index(x_shape, k_size, stride=1):
	in_c, in_h, in_w = x_shape
	out_h, out_w = (in_h - k_size) // stride + 1, (in_w - k_size) // stride + 1
	n_rows = out_h*out_w
	c_idices = np.tile(np.repeat(np.arange(in_c), k_size*k_size), (n_rows, 1))
	h_off_set = np.repeat(np.arange(0, in_h - k_size + 1, stride), out_w)
	h_indices = np.tile(np.repeat(np.arange(k_size), k_size), (n_rows, in_c))
	h_indices += h_off_set.reshape(-1,1)
	w_off_set = np.tile(np.arange(0, in_w - k_size + 1, stride), (1, out_h))
	w_indices = np.tile(np.arange(k_size), (n_rows, k_size*in_c))
	w_indices += w_off_set.reshape(-1,1)
	return c_idices, h_indices, w_indices

def img2col(img, k_size, stride=1):
	batch_size, in_c, in_h, in_w = img.shape
	c_idices, h_indices, w_indices = img2col_index([in_c, in_h, in_w], k_size, stride)
	return img[:, c_idices, h_indices, w_indices].transpose(1,0,2).reshape(-1, in_c * k_size * k_size)

def col2img(col, in_shape, k_size, stride):
	in_c, in_h, in_w = in_shape
	out_h, out_w = (in_h - k_size) // stride + 1, (in_w - k_size) // stride + 1
	batch_size = col.shape[0]//out_h//out_w
	c_idices, h_indices, w_indices = img2col_index(in_shape, k_size, stride)
	img = np.zeros((batch_size, in_c, in_h, in_w))
	np.add.at(
		img,
		(slice(None), c_idices, h_indices, w_indices),
		col.reshape(-1, batch_size, in_c * k_size * k_size).transpose(1,0,2)
	)
	return img


class Layer(object):
	def __init__(self, lr=1e-3, optimizer='Adam'):
		self.gradient_funcs = {'Adam':self.adam, "SGD": self.sgd}
		self.learning_rate = lr
		self.weight_decay = 1e-4
		self.eps = 1e-20
		self.optimizer = optimizer

	def init_momentum_cache(self):
		self.mom_w, self.cache_w = np.zeros_like(self.w), np.zeros_like(self.w)
		self.mom_b, self.cache_b = np.zeros_like(self.b), np.zeros_like(self.b)

	def forward(self, x):
		pass

	def gradient(self, grad):
		pass

	def backward(self):
		self.regularize()
		self.gradient_funcs[self.optimizer]()

	def regularize(self):
		self.w *= (1 - self.weight_decay)
		self.b *= (1 - self.weight_decay)

	def adam(self):
		beta1 = 0.9
		beta2 = 0.999
		alpha = self.learning_rate
		self.mom_w = beta1 * self.mom_w + (1 - beta1) * self.grad_w
		self.cache_w = beta2 * self.cache_w + (1 - beta2) * np.square(self.grad_w)
		self.w -= alpha * self.mom_w / (np.sqrt(self.cache_w) + self.eps)
		self.mom_b = beta1 * self.mom_b + (1 - beta1) * self.grad_b
		self.cache_b = beta2 * self.cache_b + (1 - beta2) * np.square(self.grad_b)
		self.b -= alpha * self.mom_b / (np.sqrt(self.cache_b) + self.eps)

	def sgd(self):
		self.w -= self.learning_rate * self.grad_w
		self.b -= self.learning_rate * self.grad_b


class Conv(Layer):
	def __init__(self, in_shape, k_size, k_num, stride=1, padding=0, lr=1e-3):
		super(Conv, self).__init__(lr=lr)
		self.in_shape = in_shape
		channel, height, width = in_shape
		self.k_size = k_size
		self.w = np.random.randn(channel*k_size*k_size, k_num) / np.sqrt(channel / 2) / k_size
		self.b = np.zeros((1, k_num))
		self.init_momentum_cache()

		self.out_shape = (k_num, (height + 2 * padding - k_size)//stride+1, (width + 2 * padding - k_size)//stride+1)
		self.stride, self.padding = stride, padding

	def forward(self, x):
		p = self.padding
		x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')
		self.input = x
		out = self.input
		return out

	def gradient(self, grad):
		batch_size = grad.shape[0]
		grad_out = grad.transpose(2,3,0,1).reshape([-1, self.out_shape[0]])
		self.grad_w = self.input.T.dot(grad_out) / batch_size
		self.grad_b = np.ones((1, grad_out.shape[0])).dot(grad_out) / batch_size
		self.input = None
		grad_padded = col2img(grad_out.dot(self.w.T), self.in_shape, self.k_size, self.stride)
		p = self.padding
		return grad_padded if p == 0 else grad_padded[:, :, p:-p, p:-p]

class TrasposedConv(Layer):
	def __init__(self, in_shape, k_size, k_num, stride=1, padding=0, lr=1e-3):
		pass

	def forward(self, x):
		pass

	def gradient(self, grad):
		pass


class MaxPooling(Layer):
	def __init__(self, in_shape, k_size, stride=None):
		super(MaxPooling, self).__init__()
		self.in_shape = in_shape
		channel, height, width = in_shape
		self.k_size = k_size
		self.stride = k_size if stride is None else stride
		self.out_shape = (channel, (height - k_size)//self.stride + 1, (width - k_size)//self.stride + 1)

	def gradient(self, grad):
		grad = np.repeat(grad, self.k_size, axis=2)
		grad = np.repeat(grad, self.k_size, axis=3)
		return np.multiply(self.mask, grad)

	def forward(self, x):
		col = x
		max_idx = np.argmax(col, axis=1)
		col_mask = np.zeros(col.shape)
		out = col
		return out

	def backward(self):
		pass

class Softmax(Layer):
	def __init__(self):
		super(Softmax, self).__init__()

	def loss(self, out, y):
		return -(np.multiply(y, np.log(out + self.eps))).mean()

	def forward(self, x):
		out = x
		self.out = out
		return self.out

	def gradient(self, y):
		return self.out - y

	def backward(self):
		pass

class FullyConnect(Layer):
	def __init__(self, in_shape, out_dim, lr=1e-3):
		super(FullyConnect, self).__init__(lr=lr)
		self.in_shape = in_shape
		self.w = np.random.randn(np.prod(self.in_shape), out_dim) / np.sqrt(np.prod(self.in_shape) / 2)
		self.b = np.zeros((1, out_dim))
		self.init_momentum_cache()

	def forward(self, x):
		self.input = x
		return self

	def gradient(self, grad):
		batch_size = grad.shape[0]
		self.grad_w = self.input.T.dot(grad) / batch_size
		self.grad_b = np.ones((1, batch_size)).dot(grad) / batch_size
		self.input = None
		return grad.dot(self.w.T).reshape([-1] + list(self.in_shape))


class Activation(Layer):
	def __init__(self, act_type):
		super(Activation, self).__init__()
		self.act_funcs = {'ReLU': self.relu, 'Sigmoid': self.sigmoid, 'Tanh': self.tanh, 'LeakyReLU': self.leaky_relu}
		self.dact_funcs = {'ReLU': self.drelu, 'Sigmoid': self.dsigmoid, 'Tanh': self.dtanh, 'LeakyReLU': self.dleaky_relu}
		self.act_func = self.act_funcs[act_type]
		self.dact_func = self.dact_funcs[act_type]

	def forward(self, x):
		self.out = self.act_func(x)
		return self.out

	def gradient(self, grad):
		return self.dact_func(grad, self.out)

	def relu(self, x):
		return x * (x > 0)

	def leaky_relu(self, x):
		return x * ((x > 0) * 0.99 + 0.01)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def tanh(self, x):
		return np.tanh(x)

	def drelu(self, grad, act):
		return grad * (act > 0)

	def dleaky_relu(self, grad, act):
		return grad * ((act > 0) * 0.99 + 0.01)

	def dsigmoid(self, grad, act):
		return np.multiply(grad, act - np.square(act))

	def dtanh(self, grad, act):
		return np.multiply(grad, 1 - np.square(act))

	def backward(self):
		pass

class BatchNormalization(Layer):
	def __init__(self, in_shape, lr=1e-3):
		super(BatchNormalization, self).__init__(lr=lr)
		self.in_shape = in_shape
		self.param_shape = (1, in_shape[0]) if len(in_shape) == 1 else (1, in_shape[0], 1, 1)
		self.agg_axis = 0 if len(in_shape) == 1 else (0, 2, 3)  # cnn over channel
		self.momentum = 0.99
		self.weight_decay = 0
		self.w, self.b = np.ones(self.param_shape), np.zeros(self.param_shape)
		self.init_momentum_cache()
		self.global_mean, self.global_var = np.zeros(self.param_shape), np.ones(self.param_shape)

	def forward(self, x):
		batch_mean = x.mean(axis=self.agg_axis).reshape(self.param_shape)
		batch_var = x.var(axis=self.agg_axis).reshape(self.param_shape)
		self.global_mean = batch_mean * (1.0 - self.momentum) + self.global_mean * self.momentum
		self.global_var = batch_var * (1.0 - self.momentum) + self.global_var * self.momentum
		self.batch_var_sqrt = np.sqrt(batch_var + self.eps)
		self.x_hat = (x - batch_mean) / self.batch_var_sqrt
		return self.w * self.x_hat + self.b

	def predict_forward(self, x):
		return self.w * self.b

	def gradient(self, grad):
		batch_size = grad.shape[0]
		self.grad_w = (grad * self.x_hat).sum(axis=self.agg_axis).reshape(self.param_shape) / batch_size
		self.grad_b = grad.sum(axis=self.agg_axis).reshape(self.param_shape) / batch_size
		grad_x_hat = grad * self.w
		return (
			grad_x_hat
			- grad_x_hat.mean(axis=self.agg_axis).reshape(self.param_shape)
			- self.x_hat * (grad_x_hat * self.x_hat).mean(axis=self.agg_axis).reshape(self.param_shape)
		) / self.batch_var_sqrt
