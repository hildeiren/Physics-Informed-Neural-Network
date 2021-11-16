import torch
from torch import autograd
import numpy as np
from pyDOE import lhs  # Latin Hypercube Sampling
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import scipy.special as sc


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
        super().__init__()  # call __init__ from parent class

        'activation function'
        self.activation = nn.Tanh()

        'loss function'
        self.loss_function = nn.MSELoss(reduction = 'mean')

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
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)

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
        fg = self.forward(points)
        f = fg[:, 0]

        x = points[:, 0]

        self.c = 1.0
        self.eps = 1.0
        bc_diag_func = (-self.c / 2.0 * self.eps) * np.exp((2.0 * self.c / self.eps) * x)
        loss = self.loss_function(f, bc_diag_func)

        return loss

    def loss_BC_bottom(self, points):
        fg = self.forward(points)
        f = fg[:, 0]
        g = fg[:, 1]

        loss = self.loss_function(f + g, bc_zeros)
        return loss

    def loss_PDE(self, points):
        p = points.clone()
        p.requires_grad = True

        fg = self.forward(p)
        f = fg[:, [0]]
        g = fg[:, [1]]

        f_grad = \
            autograd.grad(outputs=f, inputs=p, grad_outputs=torch.ones_like(f), retain_graph=True,
                          create_graph=True)[0]
        g_grad = \
            autograd.grad(outputs=g, inputs=p, grad_outputs=torch.ones_like(g), retain_graph=True,
                          create_graph=True)[0]

        f_x = f_grad[:, [0]]
        f_y = f_grad[:, [1]]

        g_x = g_grad[:, [0]]
        g_y = g_grad[:, [1]]

        # PDE equation

        y = points[:, [0]]

        eps = self.eps
        c = self.c

        f1 = eps * f_x - eps * f_y - c * np.exp((2 * c / eps) * y) * g
        f2 = eps * g_x + eps * g_y - c * np.exp((-2 * c / eps) * y) * f

        loss_f1 = self.loss_function(f1, f1_hat)
        loss_f2 = self.loss_function(f2, f2_hat)

        return loss_f1 + loss_f2

    def loss(self, d_points, b_points, int_points):

        loss_val = self.loss_BC_diagonal(d_points) + self.loss_BC_bottom(b_points) + self.loss_PDE(int_points)

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

    def solution(self, X, Y):

        c = self.c
        eps = self.eps
        i0_arg = (np.abs(c)/eps)*np.sqrt(np.power(X, 2)-np.power(Y, 2))
        i0 = sc.i0(i0_arg)
        i1_arg = (np.abs(c)/eps)*np.sqrt(np.power(X, 2)-np.power(Y, 2))
        i1 = sc.i1(i1_arg)
        sq_f = np.sqrt((X-Y)/(X+Y))
        exp_f = np.exp((c/eps)*(X+Y))
        sq_g = np.sqrt((X+Y)/(X-Y))
        exp_g = np.exp((c/eps)*(X-Y))

        f_sol = (-1/2 * eps) * exp_f * (c * i0-np.abs(c) * sq_f * i1)
        g_sol = (1/2 * eps) * exp_g * (c * i0-np.abs(c) * sq_g * i1)

        return f_sol, g_sol

    def plot_prediction(self):
        # Preparing data for plot
        nx, ny = (100, 100)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        self.X, self.Y = np.meshgrid(x, y)

        # points = [X,Y]
        points = np.dstack((X, Y))
        fg = self.forward(points)

        F = fg[:, :, [0]]
        G = fg[:, :, [1]]

        self.Z1 = F.cpu().detach().numpy().reshape((100, 100))
        self.Z2 = G.cpu().detach().numpy().reshape((100, 100))

        Z1 = self.Z1
        Z2 = self.Z2

        BOUND = bottom_points_np_array
        x_bound = BOUND[:, 0]
        y_bound = BOUND[:, 1]
        z_bound = np.zeros(np.shape(y_bound))

        for nx in range(100):
            for ny in range(100):
                if nx > ny:
                    Z1[nx, ny] = np.nan
                    Z2[nx, ny] = np.nan

        Z3 = Z1 + Z2  # F+G

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Prediction')
        plt.xlabel("x")
        plt.ylabel("y")
        ax.plot_surface(self.X, self.Y, Z3, rstride=1, cstride=1, cmap='viridis', edgecolor='none', vmin=0, vmax=1)
        ax.plot(x_bound, y_bound, z_bound)

        ax.view_init(30, 60)

        plt.savefig('.', dpi=500, transparent=True)
        plt.show()

        return 0

    def plot_solution(self):
        BOUND = bottom_points_np_array
        x_bound = BOUND[:, 0]
        y_bound = BOUND[:, 1]
        z_bound = np.zeros(np.shape(y_bound))

        [self.f_sol, self.g_sol] = self.solution(X, Y)

        Z3 = self.f_sol+self.g_sol

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Solution')
        plt.xlabel("x")
        plt.ylabel("y")
        ax.plot_surface(self.X, self.Y, Z3, rstride=1, cstride=1, cmap='viridis', edgecolor='none', vmin=0, vmax=1)
        ax.plot(x_bound, y_bound, z_bound)

        ax.view_init(30, 60)

        plt.savefig('.', dpi=500, transparent=True)
        plt.show()

        return 0

    def plot_compare(self, pred, sol, title):
        BOUND = bottom_points_np_array
        x_bound = BOUND[:, 0]
        y_bound = BOUND[:, 1]
        z_bound = np.zeros(np.shape(y_bound))

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.plot_surface(self.X, self.Y, pred, rstride=1, cstride=1, cmap='viridis', edgecolor='none', vmin=0, vmax=1)
        ax.plot_surface(self.X, self.Y, pred - sol, rstride=1, cstride=1, cmap='viridis', edgecolor='none', vmin=0, vmax=1)
        ax.plot(x_bound, y_bound, z_bound)

        ax.view_init(30, 60)

        plt.show()

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
    n_bd = 10000  # Total number of collocation points at boundary.
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

    layers = np.array([2, 40, 40, 40, 40, 2])  # 8 hidden layers (for time being there is 4 layers)

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
                                  max_iter=1500,
                                  max_eval=None,
                                  tolerance_grad=1e-05,
                                  tolerance_change=1e-09,
                                  history_size=100,
                                  line_search_fn='strong_wolfe')

    start_time = time.time()

    optimizer.step(PINN.closure)

    elapsed = time.time() - start_time
    print('Training time: %.2f' % (elapsed))

    plot_pred = PINN.plot_prediction()
    plot_sol = PINN.plot_solution()
    plot_compare_1 = PINN.plot_compare(PINN.Z1, PINN.f_sol, title='Error F')
    plot_compare_2 = PINN.plot_compare(PINN.Z2, PINN.g_sol, title='Error G')
