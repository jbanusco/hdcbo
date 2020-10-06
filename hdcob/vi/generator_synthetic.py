from torch.utils.data import Dataset
from .. config import *
import numpy as np
import torch


class Generator(torch.nn.Module):

	def __init__(
			self,
			lat_dim: int = 1,
			n_input: int = 2,
			n_cond: int = 3,
			seed: int = 100,
	):
		"""
		Generate data through a linear generative model:

		z ~ N(0,I)
		x = W(z), if conditional: x = W(z|v)

		where:
		"v" are the conditional features
		"W" is an arbitrary linear mapping z|v -> x

		:param lat_dim: Latent dimensions
		:param n_input: Input features to reconstruct
		:param n_cond: Conditional data
		"""
		super().__init__()

		self.n_input = n_input
		self.n_conditional = n_cond
		self.lat_dim = lat_dim

		#  Save random state (http://archive.is/9glXO)
		np.random.seed(seed)  # or None

		w = np.random.randint(-8, 8, size=(self.n_input, self.lat_dim + self.n_conditional))
		W = torch.nn.Linear(self.lat_dim + self.n_conditional, self.n_input, bias=False)
		W.requires_grad_(False)
		W.weight.data = torch.FloatTensor(w)
		self.W = W

	def forward(
			self,
			z: tensor,
			cond_data: tensor = None):
		"""
		:param z: Latent sample from which to generate the data [num_samples, lat_dim]
		:param cond_data: Data to condition the generative process [num_samples, n_cond]
		:return: obs: Reconstructed observations according to x = W(z|v)
		"""
		if type(z) == np.ndarray:
			z = tensor(z)
		assert z.size(1) == self.lat_dim

		if self.n_conditional > 0:
			if type(cond_data) == np.ndarray:
				cond_data = tensor(cond_data)
			assert cond_data.size(1) == self.n_conditional
			input_data = torch.cat((z, cond_data), dim=1)
		else:
			input_data = z

		obs = self.W(input_data)

		return obs


class SyntheticDataset(Dataset):

	def __init__(
			self,
			num_samples: int = 500,
			lat_dim: int = 2,
			n_input: int = 2,
			n_cond: int = 3,
			seed: int = 42,
	):
		self.num_samples = num_samples
		self.lat_dim = lat_dim
		self.n_input = n_input
		self.n_cond = n_cond

		np.random.seed(seed)
		self.z = tensor(np.random.normal(size=(self.num_samples, self.lat_dim)))
		if self.n_cond > 0:
			self.cond = tensor(np.random.normal(size=(self.num_samples, self.n_cond)))
		else:
			self.cond = None

		self.generator = Generator(
			lat_dim=self.lat_dim,
			n_input=self.n_input,
			n_cond=self.n_cond,
			seed=seed)

		self.x = self.generator(self.z, self.cond)

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		data_input = tensor(self.x[idx])
		if self.n_cond > 0:
			data_conditional = tensor(self.cond[idx])
		else:
			data_conditional = tensor([0])  # Don't allow None

		sample = {
			'input': data_input,
			'conditional': data_conditional}

		return sample


