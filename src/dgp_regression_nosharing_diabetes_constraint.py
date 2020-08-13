# NOTE: GPytorch version for this script `pip install git+https://github.com/cornellius-gp/gpytorch.git@02fc8dd366760ec92ed74f889626e55f21a395b3 --upgrade`
""" Experiment of monitor DGPs using recurrence
Data set:
    The Boston housing data_
Setting:
    No sharing
"""
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns;
import torch
import tqdm
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.constraints import GreaterThan, Positive, LessThan
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from matplotlib import rc
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

sns.set(style='white')


matplotlib.rcParams.update({
    'font.size': 18,
    'figure.subplot.bottom': 0.125,
})
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

font = {'family': 'serif',
        'weight': 'normal',
        'size': 18,
        }

class PreprocessingData(Dataset):

    def __init__(self, X, y):
        if not torch.is_tensor(X):
            X = StandardScaler().fit_transform(X)
            X = X.astype(np.float32)
            self.X = torch.from_numpy(X)
        else:
            raise ValueError("X should be numpy")
        if not torch.is_tensor(y):
            y = y.astype(np.float32)
            self.y = torch.from_numpy(y)
        else:
            raise ValueError("y should be numpy")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RBFConstraint(RBFKernel):

    def __init__(self, m, **kwargs):
        # self.m = m
        scale_constraint = LessThan(0.2)
        super(RBFConstraint, self).__init__(lengthscale_constraint=scale_constraint, **kwargs)
        outputscale = torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.0)
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(outputscale))
        outputscale_constraint = Positive()
        self.register_constraint("raw_outputscale", outputscale_constraint)
        self.register_buffer("m", torch.tensor(m))

    @property
    def sigma2(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)
        
    @property
    def lengthscale(self):
        # scale constraint betweent sigma^2 and ell^2
        scale = self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
        return self.sigma2 * self.m * scale
    
    def forward(self, x1, x2, diag=False, **params):
        return self.sigma2 * super(RBFConstraint, self).forward(x1, x2, diag=diag, **params)
        


class HiddenLayer(DeepGPLayer):

    def __init__(self, input_dims, output_dims, num_inducing=128):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            if torch.cuda.is_available():
                inducing_points = inducing_points.cuda()
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            if torch.cuda.is_available():
                inducing_points = inducing_points.cuda()
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = RBFConstraint(m=input_dims, ard_num_dims=None)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def get_sigma2(self):
        return self.covar_module.sigma2.data

    def get_lengthscale(self):
        return self.covar_module.lengthscale.data


class DeepGPRegression(DeepGP):

    def __init__(self, input_dims, n_layer=5, inter_dims=20):
        """All dimension are the same"""
        assert n_layer >= 2
        super().__init__()
        self.first_layer = HiddenLayer(input_dims=input_dims,
                                       output_dims=inter_dims)
        if n_layer == 2:
            self.hiddens = []
        else:
            self.hiddens = torch.nn.ModuleList(
                [HiddenLayer(input_dims=inter_dims, output_dims=inter_dims) for _ in range(n_layer - 2)])

        self.last_layer = HiddenLayer(input_dims=inter_dims,
                                      output_dims=None)

        self.likelihood = GaussianLikelihood()

    def forward(self, x):
        x = self.first_layer(x, are_samples=True)
        for hidden in self.hiddens:
            x = hidden(x)
        output = self.last_layer(x)
        return output

    def retrieve_all_hyperparameter(self):
        all_sigma2 = []
        all_sigma2.append(self.first_layer.get_sigma2())
        for h in self.hiddens:
            all_sigma2.append(h.get_sigma2())
        all_sigma2.append(self.last_layer.get_sigma2())

        all_lengthscale = []
        all_lengthscale.append(self.first_layer.get_lengthscale())
        for h in self.hiddens:
            all_lengthscale.append(h.get_lengthscale())
        all_lengthscale.append(self.last_layer.get_lengthscale())

        return all_sigma2, all_lengthscale

    def get_K_1(self, x):
        return self.first_layer.covar_module(x).evaluate().data


def train(model, train_loader, n_iter=50):
    num_data = train_loader.dataset.X.shape[0]
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    elbo = DeepApproximateMLL(VariationalELBO(model.likelihood,
                                              model,
                                              num_data=num_data))
    for i in range(n_iter):
        for x, y in train_loader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            output = model(x)
            loss = -elbo(output, y)
            loss.backward()
            optimizer.step()
        print("Iter={}\t Loss:{:.3f}".format(i, loss.item()))


def save_model(model, save_dir):
    state_dict = model.state_dict()
    torch.save(state_dict, save_dir)
    print("Save model to {}".format(save_dir))


def load_model(model, save_dir):
    if not torch.cuda.is_available():
        state_dict = torch.load(save_dir, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(save_dir)
    model.load_state_dict(state_dict)
    print("Load model from {}".format(save_dir))


def test(model, test_loader):
    model.eval()
    num_test = test_loader.dataset.X.shape[0]
    error = 0.
    ll = 0.
    with torch.no_grad():
        for x, y in test_loader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            output = model(x)
            pred = model.likelihood(output)
            error += ((y - pred.mean) ** 2).sum().cpu()
            ll += pred.log_prob(y).cpu()
        rmse = torch.sqrt(error / num_test)
        mll = ll / num_test
        print("RMSE={:.3f}\t Mean Log Likelihood={:.3f}".format(rmse.numpy(), mll.numpy()))
        return rmse.numpy(), mll.numpy()


def compute_EZ_trajectory(K, m, sigma2s, lengthscales):
    def recur(x, sigma2, lengthscale):
        floor_m_div_2 = np.floor(m / 2)
        if m % 2 == 0:
            # return 2 * sigma2 * (1. - 1. / np.power(1 + x / lengthscale ** 2, floor_m_div_2))
            return (1. - 1. / np.power(1 + x / lengthscale ** 2, floor_m_div_2))
        else:
            # return 2 * sigma2 * (1. - 1. / np.power(1 + x / lengthscale ** 2, floor_m_div_2 + 0.5))
            return (1. - 1. / np.power(1 + x / lengthscale ** 2, floor_m_div_2 + 0.5))

    x = K
    EZ = []
    for sigma2, lengthscale in zip(sigma2s, lengthscales):
        temp = recur(x, sigma2, lengthscale)
        EZ.append(temp)
        x = temp
    return EZ


# random state for reproducibility
random_state = 123
# batch_size
batch_size = 128
# predict_sample
n_predict_sample = 10

# # save dir
save_dir = "../data/dgp_regression_models_diabetes_constraint_0.3"
dims = [2, 3, 4, 5, 6, 7, 8, 9]
dims = [9]
# dims = [8]
# dims = [2]
n_layers = [2, 3, 4, 5, 6]

X, y = load_diabetes(return_X_y=True)
train_idx, test_idx = train_test_split(list(range(X.shape[0])), test_size=.1, random_state=random_state)
dataset = PreprocessingData(X, y)
train_loader = DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=SubsetRandomSampler(train_idx))
test_loader = DataLoader(dataset,
                         batch_size=batch_size,
                         sampler=SubsetRandomSampler(test_idx))

input_dims = train_loader.dataset.X.shape[-1]
RMSEs = np.zeros((len(dims), len(n_layers)))
MLLs = np.zeros((len(dims), len(n_layers)))

# # train model
for i, d in enumerate(tqdm.tqdm(dims, desc='Dimension')):
    for j, n_layer in enumerate(tqdm.tqdm(n_layers, desc='Layer')):
        file_name = "dim_{}_layer{}".format(d, n_layer)
        save_file = os.path.join(save_dir, file_name)
        if os.path.exists(save_file):
            print("File {} already exists. Skip!!!".format(save_file))
            continue

        model = DeepGPRegression(input_dims=input_dims, n_layer=n_layer, inter_dims=d)
        if torch.cuda.is_available():
            model = model.cuda()

        train(model, train_loader, n_iter=100)
        save_model(model, save_file)

# load model
for i, d in enumerate(tqdm.tqdm(dims)):
    for j, n_layer in enumerate(tqdm.tqdm(n_layers)):
        print("dDim={}\t # Layer={}".format(d, n_layer))
        model = DeepGPRegression(input_dims=input_dims, n_layer=n_layer, inter_dims=d)
        if torch.cuda.is_available():
            model = model.cuda()
        file_name = "dim_{}_layer{}".format(d, n_layer)
        load_model(model, os.path.join(save_dir, file_name))
        RMSEs = np.zeros((n_predict_sample,))
        MLLs = np.zeros((n_predict_sample,))
        for s in range(n_predict_sample):
            rmse, mll = test(model, test_loader)
            RMSEs[s] = rmse
            MLLs[s] = mll

        with open(os.path.join(save_dir, file_name + "_rmse.pkl"), 'wb') as f:
            pickle.dump(RMSEs, f)
            print("save RMSE result")

        with open(os.path.join(save_dir, file_name + "_mll.pk"), 'wb') as f:
            pickle.dump(MLLs, f)
            print("save MLL result")

        sigma2, lengthscale = model.retrieve_all_hyperparameter()
        K_1 = model.get_K_1(torch.zeros(1,1))
        EZ = compute_EZ_trajectory(K_1, m=d, sigma2s=sigma2, lengthscales=lengthscale)
        EZ = np.array([ez.numpy() for ez in EZ])
        with open(os.path.join(save_dir, file_name + "_EZ.pkl"), 'wb') as f:
            pickle.dump(EZ, f)
            print("Save EZ")


from src.plot_utils import plot_dual_ez_rmse_by_dimension, get_legend
selected_dim = dims[-1]
save_file = "../figure/experiment_diabetes/no_sharing_constraint/m_{}_dgp.png".format(selected_dim)
plot_dual_ez_rmse_by_dimension(save_dir, n_layers, save_file, selected_dim=selected_dim)

# get_legend(n_layers)
plt.show()

