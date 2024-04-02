import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import NMF

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from multichannel_net import MultipathwayNet


from multichannel_net import MultipathwayNet
torch.manual_seed(5)




X_default = torch.eye(8)
Y_default = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 1, 1],
                          [1, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1],
                          ]).T

Y_alt = torch.Tensor([[0, 0, 0, 0, 0, 1, 1, 1],
                      [0, 1, 1, 1, 0, 1, 0, 1],
                      [0, 0, 1, 0, 1, 0, 1, 0],
                      [0, 1, 1, 1, 0, 1, 1, 0],
                      [0, 1, 0, 0, 1, 0, 1, 1],
                      [1, 1, 1, 0, 1, 0, 0, 1],
                      [0, 1, 1, 1, 0, 0, 0, 1],
                      [0, 0, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 0, 0, 0, 1, 0],
                      [1, 1, 0, 0, 0, 1, 1, 1],
                      [1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1],
                      [0, 0, 1, 1, 1, 0, 0, 1]]).T

# 2 groups
diag_values1 = torch.zeros(8) + 1.0
mat1 = torch.zeros((8, 15))
mat1[range(8), range(8)] = diag_values1
diag_values2 = torch.zeros(8) - 1.0
mat2 = torch.zeros((8, 15))
mat2[range(8), range(8)] = diag_values2
# x1 = X_default[:4,:]
# x2 = X_default[4:,:]
# x1 = X_default[:,:4]
# x2 = X_default[:,4:]
# y1 = x1.mm(mat1)
# y2 = x2.mm(mat2)
x1 = torch.diag(torch.randint(1, 5, (8,))).float() / 5
x2 = torch.diag(torch.randint(-5, -1, (8,))).float() / 5
y1 = Y_default
y2 = Y_default
# X_2group = torch.cat([x1, x2],dim=0)
# Y_2group = torch.cat([y1, y2],dim=0)


class MPNAnalysis(object):
    def __init__(self, mcn, X=X_default, Y=Y_default, device=None):

        self.mcn = mcn

        self.X = X
        self.Y = Y

        if device is None:
            if torch.backends.mps.is_built():
                self.device = "mps"
            elif torch.has_cuda:
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print("Using device {}".format(self.device))

        self.mcn.to(self.device)
        self.X = self.X.to(self.device)
        self.Y = self.Y.to(self.device)

        sigma_yx = self.Y.T.mm(self.X) / self.Y.shape[0]

        U, S, V = torch.svd(sigma_yx, some=False)

        self.U = U
        self.S = S
        self.V = V

        self.loss_history = None
        self.omega_history = None
        self.K_history = None

    def omega2K(self, omega):

        with torch.no_grad():
            k = omega.mm(self.V)
            k = self.U.T.mm(k)

        return k

    def train_mcn(self, timesteps=1000, lr=0.01):
        torch.manual_seed(5)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(5)

        loss = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(params=self.mcn.parameters(), lr=lr)

        self.mcn.train()

        loss_history = []
        omega_history = []

        for t in range(timesteps):
            output = self.mcn(self.X)
            # print(output.shape)
            loss_val = loss(output, self.Y)

            loss_history.append(loss_val.to("cpu").detach().numpy())
            omega_history.append(self.mcn.omega())

            loss_val.backward()
            optimizer.step()

            optimizer.zero_grad()

        omega_history = zip(*omega_history)

        # convert omegas to Ks
        K_history = []
        for pathway in omega_history:
            K_history.append([self.omega2K(om) for om in pathway])

        self.loss_history = loss_history
        self.omega_history = omega_history
        self.K_history = K_history

        return loss_history, K_history

    def plot_K(self, ax, savedir='', labels=None, savename=None, savelabel='',
               min_val=0, max_val=2):

        if self.K_history is None:
            raise Exception(
                "MultipathwayNet must be trained before visualization.")

        num_K = len(self.K_history)

        K_list = [pathway[-1].to("cpu") for pathway in self.K_history]

        # min_val = np.min([torch.min(K) for K in K_list])
        # max_val = np.max([torch.max(K) for K in K_list])

        if labels is None:
            labels = [i for i in range(len(K_list))]

        for i, K in enumerate(K_list):
            im = ax[i].imshow(K, vmin=min_val, vmax=max_val,
                              cmap='magma')  # 'inferno'
            ax[i].set_title(r'$\bf K_{}$'.format(labels[i]), fontsize=20)
            ax[i].axis('off')

        plt.colorbar(im, ax=ax, shrink=0.6)

    def plot_K_history(self, ax, savename=None, D='unknown', savelabel=''):

        if self.K_history is None:
            raise Exception(
                "MultipathwayNet must be trained before visualization.")

        num_pathways = len(self.K_history)
        timesteps = len(self.K_history[0])

        for i in range(min(self.mcn.input_dim, self.mcn.output_dim)):

            z1 = np.array([K[i, i].to("cpu") for K in self.K_history[0]])
            z2 = np.array([K[i, i].to("cpu") for K in self.K_history[1]])

            x = np.ones(timesteps) * i
            y = np.arange(timesteps)
            if i == 0:
                ax.plot3D(x, y, z1, 'C0', linewidth=4, label=r'$K_{a\alpha}$')
                line = \
                ax.plot3D(x, y, z2, 'C1', linewidth=4, label=r'$K_{b\alpha}$')[
                    0]
                line.set_dashes([1, 1, 1, 1])
            ax.plot3D(x, y, z1, 'C0', linewidth=4)
            line = ax.plot3D(x, y, z2, 'C1', linewidth=4)[0]
            line.set_dashes([2, 1, 2, 1])
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)
        ax.set_xlabel(r'dimension $\alpha$', fontsize=15)
        ax.set_ylabel('epoch', fontsize=15)
        ax.set_zlabel(r'$K_{a,b\alpha}$', fontsize=15)
        ax.legend(fontsize=17, loc='upper left')

        ax.set_box_aspect((2.25, 1.75, 1))



class MPNAnalysi2Group(object):
    def __init__(self, mcn, X1=X_default, Y1=Y_default, X2=X_default, Y2=Y_default, device=None):

        self.mcn = mcn

        self.X1 = X1
        self.Y1 = Y1
        self.X2 = X2
        self.Y2 = Y2
        self.X = torch.cat([X1, X2],dim=0)
        self.Y = torch.cat([Y1, Y2],dim=0)

        if device is None:
            if torch.backends.mps.is_built():
                self.device = "mps"
            elif torch.has_cuda:
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print("Using device {}".format(self.device))

        self.mcn.to(self.device)
        self.X1 = self.X1.to(self.device)
        self.Y1 = self.Y1.to(self.device)
        self.X2 = self.X2.to(self.device)
        self.Y2 = self.Y2.to(self.device)
        self.X = self.X.to(self.device)
        self.Y = self.Y.to(self.device)

        sigma_yx1 = self.Y1.T.mm(self.X1) / self.Y1.shape[0]
        sigma_yx2 = self.Y2.T.mm(self.X2) / self.Y2.shape[0]
        sigma_yx = self.Y.T.mm(self.X) / self.Y.shape[0]

        U1, S1, V1 = torch.svd(sigma_yx1, some=False)
        U2, S2, V2 = torch.svd(sigma_yx2, some=False)
        U, S, V = torch.svd(sigma_yx, some=False)

        self.U1 = U1
        self.S1 = S1
        self.V1 = V1

        self.U2 = U2
        self.S2 = S2
        self.V2 = V2

        self.U = U
        self.S = S
        self.V = V

        self.loss_history = None
        self.omega_history = None
        self.K_history = None
        self.K_history1 = None
        self.K_history2 = None

    def omega2K(self, omega):

        with torch.no_grad():
            k = omega.mm(self.V)
            k = self.U.T.mm(k)

        return k
    
    def omega2K1(self, omega):

        with torch.no_grad():
            k = omega.mm(self.V1)
            k = self.U1.T.mm(k)

        return k
    
    def omega2K2(self, omega):

        with torch.no_grad():
            k = omega.mm(self.V2)
            k = self.U2.T.mm(k)

        return k

    def train_mcn(self, timesteps=1000, lr=0.01):
        torch.manual_seed(5)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(5)

        loss = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(params=self.mcn.parameters(), lr=lr)

        self.mcn.train()

        loss_history = []
        omega_history = []

        for t in range(timesteps):
            output = self.mcn(self.X)
            # print(output.shape)
            loss_val = loss(output, self.Y)

            loss_history.append(loss_val.to("cpu").detach().numpy())
            omega_history.append(self.mcn.omega())

            loss_val.backward()
            optimizer.step()

            optimizer.zero_grad()

        omega_history = zip(*omega_history)

        # convert omegas to Ks
        K_history = []
        K_history1 = []
        K_history2 = []

        for pathway in omega_history:
            K_history.append([self.omega2K(om) for om in pathway])
            K_history1.append([self.omega2K1(om) for om in pathway])
            K_history2.append([self.omega2K2(om) for om in pathway])

        self.loss_history = loss_history
        self.omega_history = omega_history
        self.K_history = K_history
        self.K_history1 = K_history1
        self.K_history2 = K_history2

        return loss_history, K_history, K_history1, K_history2
    
    def train_mcn_separate(self, timesteps=1000, lr=0.01):
        torch.manual_seed(5)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(5)

        loss = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(params=self.mcn.parameters(), lr=lr)

        self.mcn.train()

        loss_history = []
        omega_history = []

        for t in range(timesteps):
            if t % 2 == 0:
                output = self.mcn(self.X1)
                loss_val = loss(output, self.Y1)
            else:
                output = self.mcn(self.X2)
                loss_val = loss(output, self.Y2)


            loss_history.append(loss_val.to("cpu").detach().numpy())
            omega_history.append(self.mcn.omega())

            loss_val.backward()
            optimizer.step()

            optimizer.zero_grad()

        omega_history = zip(*omega_history)

        # convert omegas to Ks
        K_history = []
        K_history1 = []
        K_history2 = []

        for pathway in omega_history:
            K_history.append([self.omega2K(om) for om in pathway])
            K_history1.append([self.omega2K1(om) for om in pathway])
            K_history2.append([self.omega2K2(om) for om in pathway])

        self.loss_history = loss_history
        self.omega_history = omega_history
        self.K_history = K_history
        self.K_history1 = K_history1
        self.K_history2 = K_history2

        return loss_history, K_history, K_history1, K_history2

    def train_mcn_guide(self, timesteps=1000, lr=0.01):
        torch.manual_seed(5)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(5)

        loss = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(params=self.mcn.parameters(), lr=lr)

        self.mcn.train()

        loss_history = []
        omega_history = []

        for t in range(timesteps):

            output1 = self.mcn(self.X1, 0)
            output2 = self.mcn(self.X2, 1)
            output = output1 + output2
            loss_val = loss(output,  self.Y1+self.Y2)

            loss_history.append(loss_val.to("cpu").detach().numpy())
            omega_history.append(self.mcn.omega())

            loss_val.backward()
            optimizer.step()

            optimizer.zero_grad()

        omega_history = zip(*omega_history)

        # convert omegas to Ks
        K_history = []
        K_history1 = []
        K_history2 = []

        for pathway in omega_history:
            K_history.append([self.omega2K(om) for om in pathway])
            K_history1.append([self.omega2K1(om) for om in pathway])
            K_history2.append([self.omega2K2(om) for om in pathway])

        self.loss_history = loss_history
        self.omega_history = omega_history
        self.K_history = K_history
        self.K_history1 = K_history1
        self.K_history2 = K_history2

        return loss_history, K_history, K_history1, K_history2

    def plot_K(self, ax, savedir='', labels=None, savename=None, savelabel='',
               min_val=0, max_val=2):

        if self.K_history is None:
            raise Exception(
                "MultipathwayNet must be trained before visualization.")

        num_K = len(self.K_history)

        K_list = [pathway[-1].to("cpu") for pathway in self.K_history]

        # min_val = np.min([torch.min(K) for K in K_list])
        # max_val = np.max([torch.max(K) for K in K_list])

        if labels is None:
            labels = [i for i in range(len(K_list))]

        for i, K in enumerate(K_list):
            im = ax[i].imshow(K, vmin=min_val, vmax=max_val,
                              cmap='magma')  # 'inferno'
            ax[i].set_title(r'$\bf K_{}$'.format(labels[i]), fontsize=20)
            ax[i].axis('off')

        plt.colorbar(im, ax=ax, shrink=0.6)

    def plot_K1(self, ax, savedir='', labels=None, savename=None, savelabel='',
               min_val=0, max_val=2):

        if self.K_history1 is None:
            raise Exception(
                "MultipathwayNet must be trained before visualization.")

        num_K = len(self.K_history1)

        K_list = [pathway[-1].to("cpu") for pathway in self.K_history1]

        # min_val = np.min([torch.min(K) for K in K_list])
        # max_val = np.max([torch.max(K) for K in K_list])

        if labels is None:
            labels = [i for i in range(len(K_list))]

        for i, K in enumerate(K_list):
            im = ax[i].imshow(K, vmin=min_val, vmax=max_val,
                              cmap='magma')  # 'inferno'
            ax[i].set_title(r'$\bf K_{}$'.format(labels[i]), fontsize=20)
            ax[i].axis('off')

        plt.colorbar(im, ax=ax, shrink=0.6)

    def plot_K2(self, ax, savedir='', labels=None, savename=None, savelabel='',
               min_val=0, max_val=2):

        if self.K_history2 is None:
            raise Exception(
                "MultipathwayNet must be trained before visualization.")

        num_K = len(self.K_history2)

        K_list = [pathway[-1].to("cpu") for pathway in self.K_history2]

        # min_val = np.min([torch.min(K) for K in K_list])
        # max_val = np.max([torch.max(K) for K in K_list])

        if labels is None:
            labels = [i for i in range(len(K_list))]

        for i, K in enumerate(K_list):
            im = ax[i].imshow(K, vmin=min_val, vmax=max_val,
                              cmap='magma')  # 'inferno'
            ax[i].set_title(r'$\bf K_{}$'.format(labels[i]), fontsize=20)
            ax[i].axis('off')

        plt.colorbar(im, ax=ax, shrink=0.6)

    def plot_K_history(self, ax, savename=None, D='unknown', savelabel=''):

        if self.K_history is None:
            raise Exception(
                "MultipathwayNet must be trained before visualization.")

        num_pathways = len(self.K_history)
        timesteps = len(self.K_history[0])

        for i in range(min(self.mcn.input_dim, self.mcn.output_dim)):

            z1 = np.array([K[i, i].to("cpu") for K in self.K_history[0]])
            z2 = np.array([K[i, i].to("cpu") for K in self.K_history[1]])

            x = np.ones(timesteps) * i
            y = np.arange(timesteps)
            if i == 0:
                ax.plot3D(x, y, z1, 'C0', linewidth=4, label=r'$K_{a\alpha}$')
                line = \
                ax.plot3D(x, y, z2, 'C1', linewidth=4, label=r'$K_{b\alpha}$')[
                    0]
                line.set_dashes([1, 1, 1, 1])
            ax.plot3D(x, y, z1, 'C0', linewidth=4)
            line = ax.plot3D(x, y, z2, 'C1', linewidth=4)[0]
            line.set_dashes([2, 1, 2, 1])
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)
        ax.set_xlabel(r'dimension $\alpha$', fontsize=15)
        ax.set_ylabel('epoch', fontsize=15)
        ax.set_zlabel(r'$K_{a,b\alpha}$', fontsize=15)
        ax.legend(fontsize=17, loc='upper left')

        ax.set_box_aspect((2.25, 1.75, 1))


    def plot_K1_history(self, ax, savename=None, D='unknown', savelabel=''):

        if self.K_history1 is None:
            raise Exception(
                "MultipathwayNet must be trained before visualization.")

        num_pathways = len(self.K_history1)
        timesteps = len(self.K_history1[0])

        for i in range(min(self.mcn.input_dim, self.mcn.output_dim)):

            z1 = np.array([K[i, i].to("cpu") for K in self.K_history1[0]])
            z2 = np.array([K[i, i].to("cpu") for K in self.K_history1[1]])

            x = np.ones(timesteps) * i
            y = np.arange(timesteps)
            if i == 0:
                ax.plot3D(x, y, z1, 'C0', linewidth=4, label=r'$K_{a\alpha}$')
                line = \
                ax.plot3D(x, y, z2, 'C1', linewidth=4, label=r'$K_{b\alpha}$')[
                    0]
                line.set_dashes([1, 1, 1, 1])
            ax.plot3D(x, y, z1, 'C0', linewidth=4)
            line = ax.plot3D(x, y, z2, 'C1', linewidth=4)[0]
            line.set_dashes([2, 1, 2, 1])
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)
        ax.set_xlabel(r'dimension $\alpha$', fontsize=15)
        ax.set_ylabel('epoch', fontsize=15)
        ax.set_zlabel(r'$K_{a,b\alpha}$', fontsize=15)
        ax.legend(fontsize=17, loc='upper left')

        ax.set_box_aspect((2.25, 1.75, 1))

    def plot_K2_history(self, ax, savename=None, D='unknown', savelabel=''):

        if self.K_history2 is None:
            raise Exception(
                "MultipathwayNet must be trained before visualization.")

        num_pathways = len(self.K_history2)
        timesteps = len(self.K_history2[0])

        for i in range(min(self.mcn.input_dim, self.mcn.output_dim)):

            z1 = np.array([K[i, i].to("cpu") for K in self.K_history2[0]])
            z2 = np.array([K[i, i].to("cpu") for K in self.K_history2[1]])

            x = np.ones(timesteps) * i
            y = np.arange(timesteps)
            if i == 0:
                ax.plot3D(x, y, z1, 'C0', linewidth=4, label=r'$K_{a\alpha}$')
                line = \
                ax.plot3D(x, y, z2, 'C1', linewidth=4, label=r'$K_{b\alpha}$')[
                    0]
                line.set_dashes([1, 1, 1, 1])
            ax.plot3D(x, y, z1, 'C0', linewidth=4)
            line = ax.plot3D(x, y, z2, 'C1', linewidth=4)[0]
            line.set_dashes([2, 1, 2, 1])
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)
        ax.set_xlabel(r'dimension $\alpha$', fontsize=15)
        ax.set_ylabel('epoch', fontsize=15)
        ax.set_zlabel(r'$K_{a,b\alpha}$', fontsize=15)
        ax.legend(fontsize=17, loc='upper left')

        ax.set_box_aspect((2.25, 1.75, 1))


if __name__ == '__main__':

    import argparse

    torch.manual_seed(345345)

    plt.rc('font', size=20)
    plt.rcParams['figure.constrained_layout.use'] = True
    import matplotlib

    matplotlib.rcParams["mathtext.fontset"] = 'cm'

    parser = argparse.ArgumentParser()

    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--twogroup', action='store_true', default=False)
    parser.add_argument('--separate', action='store_true', default=False)
    parser.add_argument('--guide', action='store_true', default=False)

    # parser.add_argument('--nonlinearity', type=str, default='relu')

    args = parser.parse_args()

    nonlin = None
    # if args.nonlinearity=='relu':
    #     nonlin = torch.nn.ReLU()
    # if args.nonlinearity=='tanh':
    #     nonlin = torch.nn.Tanh()

    timesteps = args.timesteps

    depth_list = [2, 3, 4, 7]

    fig_train, ax_train = plt.subplots(1, len(depth_list), figsize=(25, 8))
    ax_train[0].set_ylabel('training error')

    fig_history = plt.figure(figsize=(24, 10))
    gs = gridspec.GridSpec(2, 6, width_ratios=[2.2, 1, 1, 2.2, 1, 1],
                           figure=fig_history)
    
    if args.twogroup:
        fig_history1 = plt.figure(figsize=(24, 10))
        gs1 = gridspec.GridSpec(2, 6, width_ratios=[2.2, 1, 1, 2.2, 1, 1],
                            figure=fig_history1)
        
        fig_history2 = plt.figure(figsize=(24, 10))
        gs2 = gridspec.GridSpec(2, 6, width_ratios=[2.2, 1, 1, 2.2, 1, 1],
                            figure=fig_history2)

    timestep_list = [1000, 1000, 1400, 10000]

    min_val = 0.0
    max_val = 0.0
    min_val1 = 0.0
    max_val1 = 0.0
    min_val2 = 0.0
    max_val2 = 0.0

    mpna_list = []

    for di, depth in enumerate(depth_list):
        ax3d = fig_history.add_subplot(gs[di * 3], projection='3d')
        if args.twogroup:
            ax3d1 = fig_history1.add_subplot(gs1[di * 3], projection='3d')
            ax3d2 = fig_history2.add_subplot(gs2[di * 3], projection='3d')

        mcn = MultipathwayNet(8, 15, depth=depth, num_pathways=2, width=1000,
                              bias=False, nonlinearity=nonlin)
        # mcn = MultipathwayNet(8, 15, depth=depth, num_pathways=2, width=1000,
        #                       bias=False, nonlinearity=nonlin)
        if not args.twogroup:
            mpna = MPNAnalysis(mcn, X=X_default, Y=Y_default)
            name = 'default'
        else:
            mpna = MPNAnalysi2Group(mcn, X1=x1, Y1=y1, X2=x2, Y2=y2)
            name = '2group'
        
        if args.separate and args.twogroup:
            mpna.train_mcn_separate(timesteps=timestep_list[di], lr=0.01)
        elif args.guide and args.twogroup:
            mpna.train_mcn_guide(timesteps=timestep_list[di], lr=0.01)
        else:
            mpna.train_mcn(timesteps=timestep_list[di], lr=0.01)

        ax_train[di].plot(mpna.loss_history)
        ax_train[di].set_xlabel('epoch')
        ax_train[di].set_title("$D={}$".format(depth))

        ax3d.set_title("$D={}$".format(depth))
        mpna.plot_K_history(ax3d, D=depth)
        if args.twogroup:
            ax3d1.set_title("$D={}$".format(depth))
            mpna.plot_K1_history(ax3d1, D=depth)
            ax3d2.set_title("$D={}$".format(depth))
            mpna.plot_K2_history(ax3d2, D=depth)

        mpna_list.append(mpna)

        K_list = [pathway[-1].to("cpu") for pathway in mpna.K_history]
        min_val_temp = np.min([torch.min(K) for K in K_list])
        max_val_temp = np.max([torch.max(K) for K in K_list])
        if min_val_temp == min_val_temp:
            min_val = min(min_val_temp, min_val)
        if max_val_temp == max_val_temp:
            max_val = max(max_val_temp, max_val)

        if args.twogroup:
            K_list1 = [pathway[-1].to("cpu") for pathway in mpna.K_history1]
            min_val_temp1 = np.min([torch.min(K) for K in K_list1])
            max_val_temp1 = np.max([torch.max(K) for K in K_list1])
            if min_val_temp1 == min_val_temp1:
                min_val1 = min(min_val_temp1, min_val1)
            if max_val_temp1 == max_val_temp1:
                max_val1 = max(max_val_temp1, max_val1)

            K_list2 = [pathway[-1].to("cpu") for pathway in mpna.K_history2]
            min_val_temp2 = np.min([torch.min(K) for K in K_list2])
            max_val_temp2 = np.max([torch.max(K) for K in K_list2])
            if min_val_temp2 == min_val_temp2:
                min_val2 = min(min_val_temp2, min_val2)
            if max_val_temp2 == max_val_temp2:
                max_val2 = max(max_val_temp2, max_val2)

    for di, depth in enumerate(depth_list):
        mpna = mpna_list[di]

        ax2 = fig_history.add_subplot(gs[di * 3 + 1])
        ax3 = fig_history.add_subplot(gs[di * 3 + 2])

        if args.twogroup:
            ax21 = fig_history1.add_subplot(gs1[di * 3 + 1])
            ax31 = fig_history1.add_subplot(gs1[di * 3 + 2])

            ax22 = fig_history2.add_subplot(gs2[di * 3 + 1])
            ax32 = fig_history2.add_subplot(gs2[di * 3 + 2])

        mpna.plot_K([ax2, ax3], labels=['a', 'b'], min_val=min_val,
                    max_val=max_val)
        
        if args.twogroup:
            mpna.plot_K1([ax21, ax31], labels=['a', 'b'], min_val=min_val1,
                        max_val=max_val1)
            
            mpna.plot_K2([ax22, ax32], labels=['a', 'b'], min_val=min_val2,
                        max_val=max_val2)

    fig_train.suptitle("Training loss")
    fig_train.savefig(f'split_train_loss_{name}_separate{args.separate}_guide{args.guide}.pdf')
    fig_history.savefig(f'split_test_{name}_separate{args.separate}_guide{args.guide}.pdf')
    if args.twogroup:
        fig_history1.savefig(f'split_test1_{name}_separate{args.separate}_guide{args.guide}.pdf')
        fig_history2.savefig(f'split_test2_{name}_separate{args.separate}_guide{args.guide}.pdf')


    plt.show()
