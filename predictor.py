import torch
import numpy as np
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from scipy.ndimage import gaussian_filter

# We will use the simplest form of GP model, exact inference
class DirichletGPModel(ExactGP):
#     def __init__(self, train_x, train_y, likelihood, num_classes):
#         super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
#         self.covar_module = gpytorch.kernels.GridInterpolationKernel(
#             RBFKernel(batch_shape=torch.Size((num_classes,))),
#             grid_size=50, num_dims=2, grid_bounds=((-3, 3), (-3, 3)),
#         )
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def Drichlet_fit(train_x, train_y):
    # initialize likelihood and model
    # we let the DirichletClassificationLikelihood compute the targets for us
    likelihood = DirichletClassificationLikelihood(train_y.flatten(), learn_additional_noise=True)
    model = DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)
    training_iter = 50

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, likelihood.transformed_targets).sum()
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):
        return model, likelihood


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def fit(model, likelihood, train_x, train_y):
    if model is None:
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultitaskGPModel(train_x, train_y, likelihood)
    else:
        model.set_train_data(train_x, train_y, strict=False)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    training_iterations = 50

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):
        return model, likelihood


def measured_vs_pos_test(measured_positions, test_pos):
    return (np.sqrt((np.array(measured_positions)[:,0] - test_pos[0])**2 + (np.array(measured_positions)[:,1] - test_pos[1])**2) < 0.1).any()

def choose_next_position(model, likelihood, test_x, measured_positions):
    test_dist = model(test_x)
    g = 0.25

    means = test_dist.mean[:,0].detach().numpy()
    means = means/means.sum()
    stddevs = likelihood(test_dist).stddev[:,1].detach().numpy()
    stddevs = stddevs/stddevs.sum()
    choice_pdf = g * means + (1 - g) * stddevs
    choice_pdf = choice_pdf/choice_pdf.sum()

    least_certain_test_index = np.random.choice(np.arange(choice_pdf.shape[0]),p=choice_pdf.flatten())

#     if measured_vs_pos_test(measured_positions, test_x[least_certain_test_index].numpy()):
#         for i in np.argsort(choice_pdf.flatten())[::-1]:
#             if measured_vs_pos_test(measured_positions, test_x[i].numpy()):
#                 continue
#             else:
#                 least_certain_test_index = i
#                 break

#     fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))
#     ax1.imshow(stddevs.reshape(91,91), origin='lower', cmap='terrain')
#     ax2.imshow(means.reshape(91,91), origin='lower', cmap='terrain')
#     ax3.imshow(choice_pdf.reshape(91,91), origin='lower', cmap='terrain')
#     ax3.scatter(least_certain_test_index%91, least_certain_test_index//91, color='red')
#     plt.show()
    return least_certain_test_index, choice_pdf


def Dirichlet_choose_next_position(model, likelihood, test_x, measured_positions):
    test_dist = model(test_x)

    stddev_pdf = likelihood(test_dist).stddev.detach().numpy().prod(0)
    choice_pdf = stddev_pdf/stddev_pdf.sum()
    least_certain_test_index = np.argmax(choice_pdf.flatten())

    if measured_vs_pos_test(measured_positions, test_x[least_certain_test_index].numpy()):
        for i in np.argsort(choice_pdf.flatten())[::-1]:
            if measured_vs_pos_test(measured_positions, test_x[i].numpy()):
                continue
            else:
                least_certain_test_index = i
                break

    
    return least_certain_test_index, choice_pdf
