from torch.utils.data import Dataset
from hdcob.config import *
import numpy as np
import torch


class Generator(torch.nn.Module):

	def __init__(
			self,
			n_input: int = 2,
			n_target: int = 3,
			seed: int = 100,
	):
		"""
		Generate data through a linear generative model:

		z ~ N(0,I)
		x = W(z)

		where:
		"v" are the conditional features
		"W" is an arbitrary linear mapping z -> x

		:param n_input: Input features [sample locations]
		:param n_target: Output features [target data]
		"""
		super().__init__()

		self.n_input = n_input
		self.n_target = n_target

		#  Save random state (http://archive.is/9glXO)
		np.random.seed(seed)  # or None

		w = np.random.randint(-8, 8, size=(self.n_target, self.n_input))
		W = torch.nn.Linear(self.n_input, self.n_target, bias=False)
		W.requires_grad_(False)
		W.weight.data = torch.FloatTensor(w)
		self.W = W

	def forward(
			self,
			x: tensor):
		"""
		:param x: Sample locations that predict the target data [num_samples, n_input]
		:return: obs: Reconstructed observations according to x = W(z)
		"""
		if type(x) == np.ndarray:
			x = tensor(x)
		assert x.size(1) == self.n_input

		obs = self.W(x)

		return obs


class SyntheticDataset(Dataset):

	def __init__(
			self,
			num_samples: int = 500,
			n_input: int = 2,
			n_target: int = 3,
			seed: int = 42,
	):
		self.num_samples = num_samples
		self.n_input = n_input
		self.n_target = n_target

		np.random.seed(seed)
		self.x = tensor(np.random.normal(size=(self.num_samples, self.n_input)))

		self.generator = Generator(
			n_input=self.n_input,
			n_target=self.n_target,
			seed=seed)

		self.y = self.generator(self.x)

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x = tensor(self.x[idx])
		y = tensor(self.y[idx])

		sample = {
			'x': x,
			'y': y}

		return sample


