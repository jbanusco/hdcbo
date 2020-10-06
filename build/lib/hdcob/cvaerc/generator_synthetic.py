from hdcob.config import *
from torch.utils.data import Dataset


class GeneratorJoint(torch.nn.Module):
    def __init__(
            self,
            lat_dim=1,  # Latent dimension
            n_input=2,  # Fixed input to GP
            n_conditional=3,  # Data to condition
            n_miss=3,  # Missing data (also input to GP)
            n_target=5,  # GP target data
            seed=100,
    ):
        super().__init__()

        self.n_input = n_input
        self.n_miss = n_miss
        self.n_target = n_target
        self.n_conditional = n_conditional
        self.lat_dim = lat_dim

        #  Save random state (http://archive.is/9glXO)
        np.random.seed(seed)  # or None
        # self.random_state = np.random.get_state()  # get the initial state of the RNG

        # Condition also on the fix input of the GP
        w_ = np.random.uniform(-1, 1, (self.n_miss, self.lat_dim + self.n_conditional + self.n_input))
        # w_ = np.random.uniform(-1, 1, (self.n_miss, self.lat_dim + self.n_conditional))

        u, s, vt = np.linalg.svd(w_, full_matrices=False)
        w = u if self.n_miss >= (self.lat_dim + self.n_conditional + self.n_input) else vt
        W = torch.nn.Linear(self.lat_dim + self.n_conditional + self.n_input, self.n_miss, bias=False)
        W.requires_grad_(False)
        W.weight.data = torch.FloatTensor(w)
        self.W = W

        # 2nd random transformation
        w2_ = np.random.uniform(-1, 1, (self.n_target, self.n_miss + self.n_input))
        # w2_ = np.random.randn(self.n_target, self.n_miss + self.n_input)
        u, s, vt = np.linalg.svd(w2_, full_matrices=False)
        w2 = u if self.n_target >= (self.n_miss + self.n_input) else vt
        # w2 = np.random.uniform(-1, 1, (self.n_target, self.n_miss + self.n_input))
        W2 = torch.nn.Linear(self.n_input + self.n_miss, self.n_target, bias=False)
        W2.requires_grad_(False)
        W2.weight.data = torch.FloatTensor(w2)
        # self.W2 = W2
        self.W2 = lambda x: torch.sin(W2(x))
        # self.W2 = lambda x: W2(torch.sin(x))
        # self.W2 = lambda x: torch.sin(x)

    def forward(self, z, input_data, cond_data, noise_lvl=0.1):
        if type(z) == np.ndarray:
            z = tensor(z)

        assert z.size(1) == self.lat_dim
        assert input_data.size(1) == self.n_input
        assert cond_data.size(1) == self.n_conditional

        # input_to_first = torch.cat((z, cond_data), dim=1)
        input_to_first = torch.cat((z, cond_data, input_data), dim=1)

        missing_data = self.W(input_to_first)
        noise_miss = tensor(np.random.randn(missing_data.size(0), missing_data.size(1))) * noise_lvl
        # missing_data += noise_miss

        # noise_input = tensor(np.random.randn(input_data.size(0), input_data.size(1))) * noise_lvl
        x = torch.cat((input_data, missing_data), dim=1)
        assert x.size(1) == (self.n_input + self.n_miss)

        target_data = self.W2(x)
        noise_target = tensor(np.random.randn(x.size(0), target_data.size(1))) * noise_lvl

        missing_data += noise_miss
        target_data += noise_target

        return missing_data, target_data


class SyntheticDataset(Dataset):
    def __init__(self,
                 num_samples=500,
                 lat_dim=1,
                 input_dim=2,
                 cond_dim=3,
                 miss_dim=3,
                 target_dim=5,
                 seed=100,
                 noise_lvl=0.1):

        self.num_samples = num_samples
        self.lat_dim = lat_dim
        self.cond_dim = cond_dim
        self.input_dim = input_dim
        self.miss_dim = miss_dim
        self.target_dim = target_dim
        self.noise_lvl = noise_lvl

        np.random.seed(seed)

        z = tensor(np.random.normal(size=(self.num_samples, self.lat_dim)))
        self.cond_data = tensor(np.random.normal(size=(self.num_samples, self.cond_dim)))
        self.input_data = tensor(np.random.normal(size=(self.num_samples, self.input_dim)))

        self.generator = GeneratorJoint(
            lat_dim=self.lat_dim,
            n_input=self.input_dim,  # Fixed input to GP
            n_conditional=self.cond_dim,  # Data to condition
            n_miss=self.miss_dim,  # Missing data (also input to GP)
            n_target=self.target_dim,  # GP target data
            seed=seed,
        )

        self.missing_data, self.target_data = self.generator(z, self.input_data, self.cond_data, self.noise_lvl)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input': self.input_data[idx],
                  'conditional': self.cond_data[idx],
                  'missing': self.missing_data[idx],
                  'target': self.target_data[idx],
                  }
        return sample


if __name__ == '__main__':
    print("eh")

    dataset = SyntheticDataset(num_samples=500, lat_dim=1, input_dim=2, cond_dim=3, miss_dim=3,
                               target_dim=5, noise_lvl=0.1)
