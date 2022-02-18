import torch
from torch import autograd
import numpy as np
from pyDOE import lhs  # Latin Hypercube Sampling
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import scipy.special as sc
import scipy.io


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
        self.a = 1.0
        self.b = 1.0
        self.eps = 1.0
        self.gamma = 2 * self.b/self.eps
        self.q = -1.0
        super().__init__()  # call __init__ from parent class

        'activation function'
        self.activation = nn.Tanh()

        'loss function'
        self.loss_function = nn.MSELoss(reduction='mean')

        'Initialise neural network as a list using nn.Modulelist'
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        self.iter = 0

        '''
        Alternatively:

        *all layers are callable

        Simple linear Layers
        self.fc1 = nn.Linear(2,50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50,1)

        '''

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

        '''    
        Alternatively:

        a = self.activation(self.fc1(a))
        a = self.activation(self.fc2(a))
        a = self.activation(self.fc3(a))
        a = self.fc4(a)

        '''

        for i in range(len(layers) - 2):
            z = self.linears[i](a)

            a = self.activation(z)

        a = self.linears[-1](a)

        return a

    def loss_BC_diagonal(self, points):
        KvuKvv = self.forward(points)
        Kvu = KvuKvv[:, 0]

        x = points[:, 0]


        bc_diag_func = (-0.5 * self.b) * np.exp(-self.gamma * x)
        loss = self.loss_function(Kvu, bc_diag_func)
        return loss

    def loss_BC_bottom(self, points):
        KvuKvv = self.forward(points)
        Kvu = KvuKvv[:, 0]
        Kvv = KvuKvv[:, 1]

        bottom = self.q * Kvu

        loss = self.loss_function(Kvv, bottom)
        return loss

    def loss_Kvu(self, points):
        p = points.clone()
        p.requires_grad_(True)

        KvuKvv = self.forward(p)
        Kvu = KvuKvv[:, [0]]
        Kvv = KvuKvv[:, [1]]

        Kvu_grad = \
            autograd.grad(outputs=Kvu, inputs=p, grad_outputs=torch.ones_like(Kvu), create_graph=True)[0]

        self.Kvu_x = Kvu_grad[:, [0]]
        self.Kvu_xi = Kvu_grad[:, [1]]

        # PDE equation

        xi = points[:, [1]]

        f1 = self.Kvu_x - self.Kvu_xi - self.b * np.exp(-self.gamma * xi) * Kvv

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

        self.Kvv_x = Kvv_grad[:, [0]]
        self.Kvv_xi = Kvv_grad[:, [1]]

        # PDE equation

        xi = points[:, [1]]

        f2 = self.Kvv_x + self.Kvv_xi - self.a * np.exp(self.gamma * xi) * Kvu

        loss_f2 = self.loss_function(f2, f2_hat)

        return loss_f2

    def loss(self, d_points, b_points, int_points):

        loss_val = self.loss_BC_diagonal(d_points) + self.loss_BC_bottom(b_points) + \
                   self.loss_Kvu(int_points) + self.loss_Kvv(int_points)

        return loss_val

    # Callable for optimizer
    def closure(self):

        optimizer.zero_grad()

        loss = self.loss(diagonal_points, bottom_points, interior_points)

        loss.backward()

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
    n_bd = 1000  # Total number of collocation points at boundary.
    n_int = 10000  # Total number of collocation points in the interior.
    diagonal_points_np_array, bottom_points_np_array, interior_points_np_array = training_points(n_bd, n_int)

    # X_f_train_np_array, X_u_train_np_array, u_train_np_array = trainingdata(N_u, N_f)

    # Convert to tensor and send to GPU
    diagonal_points = torch.from_numpy(diagonal_points_np_array).float().to(device)
    bottom_points = torch.from_numpy(bottom_points_np_array).float().to(device)
    interior_points = torch.from_numpy(interior_points_np_array).float().to(device)
    bc_zeros = torch.zeros(bottom_points.shape[0]).to(device)
    #test_data = torch.from_numpy(X_u_test).float().to(device)
    #u = torch.from_numpy(u_true).float().to(device)
    f1_hat = torch.zeros(interior_points.shape[0], 1).to(device)
    f2_hat = torch.zeros(interior_points.shape[0], 1).to(device)

    layers = np.array([2, 40, 40, 40, 40, 40, 40, 40, 40, 2])  # 8 hidden layers
    #layers = np.array([2, 40, 40, 40, 40, 2]) # 4 hidden layers

    PINN = SequentialModel(layers)
    PINN.to(device)

    """Saving model"""
    # Specify a path
    #PATH = "PINN.state_dict_model.pt"

    # Save
    #torch.save(PINN.state_dict(), PATH)

    # Load
    #load_PINN = SequentialModel(layers)
    #load_PINN.load_state_dict(torch.load(PATH))

    # Neural Network Summary
    print(PINN)

    params = list(PINN.parameters())
    #loaded_params = list(load_PINN.parameters())

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


