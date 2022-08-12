import torch
from torch import autograd
import numpy as np
from pyDOE import lhs  # Latin Hypercube Sampling
import torch.nn as nn
import time
import scipy.io
import argparse



def training_points(n_bd, n_int):
    # n_bd number of points on boundary, n_int number of points in the interior domain

    # Boundary condition at x=y
    x = np.linspace(0.0, 1.0, n_bd).reshape(n_bd, 1)
    diagonal_points = np.hstack((x, x))

    # Boundary condition at y=0
    bottom_points = np.hstack((x, np.zeros((n_bd, 1))))

    # Points in the interior domain
    # Latin Hypercube sampling for collocation points
    # n_int sets of tuples(x,t)
    interior_points = lhs(2, n_int)

    # Change from square to triangular by mirroring infeasible points: if y>x, (x,y) -> (y,x)
    for point in interior_points:
        if point[1] > point[0]:
            point[0], point[1] = point[1], point[0]

    return diagonal_points, bottom_points, interior_points


class SequentialModel(nn.Module):

    def __init__(self, layers):
        self.q = 100.0
        super().__init__()  # call __init__ from parent class

        'activation function'
        self.activation = nn.Tanh()

        'loss function'
        self.loss_function = nn.MSELoss(reduction='mean')

        'Initialise neural network as a list using nn.Modulelist'
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        self.iter = 0

        'Xavier Normal Initialization'
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers) - 1):
            # weights from a normal distribution with
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1)

            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

    'forward pass'

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        # convert to float
        a = x.float()

        for i in range(len(layers) - 2):
            z = self.linears[i](a)

            a = self.activation(z)

        a = self.linears[-1](a)

        return a

    def eps1(self, x):
        return 1 + torch.sin(x**2)

    def eps2(self, x):
        return 1 + x**3

    def c1(self, x):
        return torch.cos(2*x)

    def c2(self, x):
        return 2 + torch.arctan(torch.sin(x)) + torch.exp(torch.sin(x)) + torch.sin(2*x)

    def loss_BC_diagonal(self, points):
        KvuKvv = self.forward(points)
        Kvu = KvuKvv[:, 0]
        x = points[:, 0]

        bc_diag_func = -(self.c2(x) / (self.eps1(x) + self.eps2(x)))
        loss = self.loss_function(Kvu, bc_diag_func)
        return loss

    def loss_BC_bottom(self, points):
        KvuKvv = self.forward(points)
        Kvu = KvuKvv[:, 0]
        Kvv = KvuKvv[:, 1]

        zero = torch.zeros(n_bd)

        bottom = (self.q * self.eps1(zero) / self.eps2(zero)) * Kvu

        loss = self.loss_function(Kvv, bottom)
        return loss

    def eps1_grad(self, points):
        p = points.clone()
        p.requires_grad_(True)

        eps1xi = self.eps1(p)
        eps1_xi = autograd.grad(outputs=eps1xi, inputs=p, grad_outputs=torch.ones_like(eps1xi), create_graph=True)[0]

        return eps1_xi

    def eps2_grad(self, points):
        p = points.clone()
        p.requires_grad_(True)

        eps2xi = self.eps2(p)
        eps2_xi = autograd.grad(outputs=eps2xi, inputs=p, grad_outputs=torch.ones_like(eps2xi), create_graph=True)[0]

        return eps2_xi

    def loss_Kvu(self, points):
        p = points.clone()
        p.requires_grad_(True)

        KvuKvv = self.forward(p)
        Kvu = KvuKvv[:, [0]]
        Kvv = KvuKvv[:, [1]]

        Kvu_grad = \
            autograd.grad(outputs=Kvu, inputs=p, grad_outputs=torch.ones_like(Kvu), create_graph=True)[0]

        Kvu_x = Kvu_grad[:, [0]]
        Kvu_xi = Kvu_grad[:, [1]]

        x = points[:, [0]]
        xi = points[:, [1]]

        eps1_xi = self.eps1_grad(xi)

        f1 = self.eps2(x) * Kvu_x - self.eps1(xi) * Kvu_xi - eps1_xi * Kvu - self.c2(xi) * Kvv

        loss_f1 = self.loss_function(f1, f1_hat)

        return loss_f1

    def loss_Kvv(self, points):
        p = points.clone()
        p.requires_grad_(True)

        KvuKvv = self.forward(p)
        Kvu = KvuKvv[:, [0]]
        Kvv = KvuKvv[:, [1]]

        Kvv_grad = \
            autograd.grad(outputs=Kvv, inputs=p, grad_outputs=torch.ones_like(Kvv), create_graph=True)[0]

        Kvv_x = Kvv_grad[:, [0]]
        Kvv_xi = Kvv_grad[:, [1]]

        x = points[:, [0]]
        xi = points[:, [1]]

        eps2_xi = self.eps2_grad(xi)

        f2 = self.eps2(x) * Kvv_x + self.eps2(xi) * Kvv_xi + eps2_xi * Kvv - self.c1(xi) * Kvu

        loss_f2 = self.loss_function(f2, f2_hat)

        return loss_f2

    def loss(self, d_points, b_points, int_points):

        loss_val = 100*self.loss_BC_diagonal(d_points) + self.loss_BC_bottom(b_points) + \
                   self.loss_Kvu(int_points) + self.loss_Kvv(int_points)

        return loss_val

    # Callable for optimizer
    def closure(self):

        optimizer.zero_grad()

        loss = self.loss(diagonal_points, bottom_points, interior_points)

        loss.backward(retain_graph=True)

        self.iter += 1

        if self.iter % 100 == 0:
            print(f'Loss is: {loss}')

        return loss

    def save_prediction_to_mat_file(self):
        # Preparing data
        nx, ny = (100, 100)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # points = [X,Y]
        points = np.dstack((X, Y))
        p = torch.from_numpy(points).float().to(device)
        p.requires_grad_(True)
        KvuKvv = self.forward(p)

        Kvu = KvuKvv[:, :, [0]]
        Kvv = KvuKvv[:, :, [1]]

        Kvu_grad = \
            autograd.grad(outputs=Kvu, inputs=p, grad_outputs=torch.ones_like(Kvu), create_graph=True)[0]
        Kvv_grad = \
            autograd.grad(outputs=Kvv, inputs=p, grad_outputs=torch.ones_like(Kvv), create_graph=True)[0]

        Kvu_x = Kvu_grad[:, :, [0]]
        Kvu_xi = Kvu_grad[:, :, [1]]
        Kvv_x = Kvv_grad[:, :, [0]]
        Kvv_xi = Kvv_grad[:, :, [1]]

        Z1 = Kvu.cpu().detach().numpy().reshape((100, 100))
        Z2 = Kvv.cpu().detach().numpy().reshape((100, 100))
        Z3 = Kvu_x.cpu().detach().numpy().reshape((100, 100))
        Z4 = Kvu_xi.cpu().detach().numpy().reshape((100, 100))
        Z5 = Kvv_x.cpu().detach().numpy().reshape((100, 100))
        Z6 = Kvv_xi.cpu().detach().numpy().reshape((100, 100))

        for nx in range(100):
            for ny in range(100):
                if nx > ny:
                    Z1[nx, ny] = np.nan
                    Z2[nx, ny] = np.nan
                    Z3[nx, ny] = np.nan
                    Z4[nx, ny] = np.nan
                    Z5[nx, ny] = np.nan
                    Z6[nx, ny] = np.nan

        scipy.io.savemat('f.mat', {'Kvu_PINN': Z1})
        scipy.io.savemat('g.mat', {'Kvv_PINN': Z2})
        scipy.io.savemat('Kvu_x.mat', {'Kvu_x_PINN': Z3})
        scipy.io.savemat('Kvu_xi.mat', {'Kvu_xi_PINN': Z4})
        scipy.io.savemat('Kvv_x.mat', {'Kvv_x_PINN': Z5})
        scipy.io.savemat('Kvv_xi.mat', {'Kvv_xi_PINN': Z6})

        return 0

    def print_loss(self, d_points, b_points, int_points):
        print(f'Diagonal loss is: {self.loss_BC_diagonal(d_points)}')
        print(f'Bottom loss is: {self.loss_BC_bottom(b_points)}')
        print(f'Kvu loss is: {self.loss_Kvu(int_points)}')
        print(f'Kvv loss is: {self.loss_Kvv(int_points)}')

        return 0


if __name__ == '__main__':
    ################
    # Arguments
    ################

    parser = argparse.ArgumentParser(description='PINN for solving the infinite-dimensional backstepping kernel')

    parser.add_argument('--n_bd', type=int, default=100, help='Number of training points at the boundary conditions ')
    parser.add_argument('--n_int', type=int, default=1000, help='Number of training points at the interior ')
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[200, 200], help='Number of units')

    args = parser.parse_args()
    # Set default dtype to float32
    torch.set_default_dtype(torch.float)

    # PyTorch random number generator
    torch.manual_seed(1234)

    # Random number generators in other libraries
    np.random.seed(1234)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    if device == 'cuda':
        print(torch.cuda.get_device_name())

    # Generate points
    n_bd = args.n_bd  # Total number of collocation points at boundary.
    n_int = args.n_int  # Total number of collocation points in the interior.
    diagonal_points_np_array, bottom_points_np_array, interior_points_np_array = training_points(n_bd, n_int)

    # Convert to tensor and send to GPU
    diagonal_points = torch.from_numpy(diagonal_points_np_array).float().to(device)
    bottom_points = torch.from_numpy(bottom_points_np_array).float().to(device)
    interior_points = torch.from_numpy(interior_points_np_array).float().to(device)
    bc_zeros = torch.zeros(bottom_points.shape[0]).to(device)



    f1_hat = torch.zeros(interior_points.shape[0], 1).to(device)
    f2_hat = torch.zeros(interior_points.shape[0], 1).to(device)

    hidden_layers = np.array(args.hidden_layers)
    layers = np.concatenate(([2], hidden_layers, [2]))

    PINN = SequentialModel(layers)
    PINN.to(device)

    # Neural Network Summary
    print(PINN)

    params = list(PINN.parameters())

    # Optimization
    # L-BFGS Optimizer
    optimizer = torch.optim.LBFGS(PINN.parameters(), lr=0.1,
                                  max_iter=20000,
                                  max_eval=None,
                                  tolerance_grad=1e-05,
                                  tolerance_change=1e-09,
                                  history_size=100,
                                  line_search_fn='strong_wolfe')

    start_time = time.time()

    optimizer.step(PINN.closure)

    elapsed = time.time() - start_time
    print('Training time: %.2f' % (elapsed))

    PINN.save_prediction_to_mat_file()
    PINN.print_loss(diagonal_points, bottom_points, interior_points)


