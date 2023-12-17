import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlayers.regularization import L1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('Warning, training model on CPU')


class SparseClassification(nn.Module):
    def __init__(self, latent_dims, hidden_dims=200, regularization_weight=1e-4):
        """
        Sparse classification module to prioritize relevant image properties
        :param latent_dims: amount of latent dimensions in the autoencoder
        :param hidden_dims: amount of hidden dimensions to use
        """
        super(SparseClassification, self).__init__()
        self.linear1 = L1(nn.Linear(latent_dims, hidden_dims), regularization_weight)
        self.linear2 = nn.Linear(hidden_dims, 1)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = self.linear2(z)
        return z


class Encoder(nn.Module):

    def __init__(self, latent_dims):
        """
        Encoder module for STE-TCVAE
        :param latent_dims: number of latent dimensions
        """
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(265, 512, 3)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 128, 1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 256, 4, 2, (0, 1))
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, 3)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv9 = nn.Conv2d(512, 64, 1)
        self.bn9 = nn.BatchNorm2d(64)

        self.linear1 = nn.Linear(12*10*64, latent_dims)
        self.linear2 = nn.Linear(12*10*64, latent_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = torch.flatten(x, start_dim=1)

        mu = self.linear1(x)
        log_var = self.linear2(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(latent_dims, 10*12*64)
        self.bn10 = nn.BatchNorm1d(10*12*64)

        self.conv9 = nn.ConvTranspose2d(64, 512, 1)
        self.bn9 = nn.BatchNorm2d(512)

        self.conv8 = nn.ConvTranspose2d(512, 256, 3)
        self.bn8 = nn.BatchNorm2d(256)

        self.conv7 = nn.ConvTranspose2d(256, 128, 4, 2, (0, 1))
        self.bn7 = nn.BatchNorm2d(128)

        self.conv6 = nn.ConvTranspose2d(128, 512, 1)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv5 = nn.ConvTranspose2d(512, 256, 3)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv4 = nn.ConvTranspose2d(256, 128, 3)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv2 = nn.ConvTranspose2d(64, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv1 = nn.ConvTranspose2d(32, 3, 3)

    def forward(self, z):
        z = F.relu(self.bn10(self.linear1(z)))
        z = z.reshape(-1, 64, 12, 10)
        z = F.relu(self.bn9(self.conv9(z)))
        z = F.relu(self.bn8(self.conv8(z)))
        z = F.relu(self.bn7(self.conv7(z)))
        z = F.relu(self.bn6(self.conv6(z)))
        z = F.relu(self.bn5(self.conv5(z)))
        z = F.relu(self.bn4(self.conv4(z)))
        z = F.relu(self.bn3(self.conv3(z)))
        z = F.relu(self.bn2(self.conv2(z)))

        z = self.conv1(z)
        z = torch.sigmoid(z)
        return z


class STETCVAE(nn.Module):
    def __init__(self, dataset_size, positive_weight, latent_dims=200, encoder=Encoder, decoder=Decoder,
                 classifier=SparseClassification, classifier_hidden_dims=200, beta_init=20, gamma_init=1,
                 polarize_percent=50, classification_weight=0.1, Kp_beta=1e-2, Ki_beta=1e-4, Kp_gamma=1e-2,
                 Ki_gamma=1e-4, max_beta=30, min_beta=1, max_gamma=10000, min_gamma=0, verbose=False):
        """
        STE-TCVAE model module
        :param dataset_size: Size of the training dataset
        :param positive_weight: Weight of the positive samples
        :param latent_dims: amount of latent dimensions, default: 200
        :param encoder: The encoder module
        :param decoder: The decoder module
        :param classifier: The classification module
        :param classifier_hidden_dims: The amount of hidden dimensions for the classifier, default: 200
        :param beta_init: Initial value for the beta parameter, default: 20
        :param gamma_init: Initial value for the gamma parameter, default: 1
        :param polarize_percent: Value for the target polarization amount, default: 50
        :param classification_weight: Value on [0,1] for the max contribution of classification to the loss, default: .1
        :param Kp_beta: Proportion parameter for PI controller beta, default: 1e-2
        :param Ki_beta: Integral parameter for PI controller beta, default 1e-4
        :param Kp_gamma: Proportion parameter for PI controller gamma, default: 1e-2
        :param Ki_gamma: Integral parameter for PI controller gamma, default: 1e-4
        :param max_beta: Maximum value for beta, default: 30
        :param min_beta: Minimum value for beta, default: 1
        :param max_gamma: Maximum value for gamma: default: 10000
        :param min_gamma: Minimum value for gamma: default: 0
        :param verbose: whether to print extra information, default False
        """
        super(STETCVAE, self).__init__()

        self.encoder = encoder(latent_dims)
        self.decoder = decoder(latent_dims)
        self.classifier = classifier(latent_dims, classifier_hidden_dims)

        self.beta = beta_init
        self.gamma = gamma_init

        self.size = dataset_size
        self.latent_dims = latent_dims

        # beta parameter
        self.polarize_percent = polarize_percent
        self.Kp_beta = Kp_beta
        self.Ki_beta = Ki_beta
        self.max_beta = max_beta
        self.min_beta = min_beta
        self.I_beta = beta_init

        # classification parameters
        self.class_percent = classification_weight
        self.Kp_gamma = Kp_gamma
        self.Ki_gamma = Ki_gamma
        self.max_gamma = max_gamma
        self.min_gamma = min_gamma
        self.I_gamma = gamma_init
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

        self.verbose = verbose

    def sample(self, mu, std):
        eps = torch.randn_like(std)
        return mu+eps*std

    def pi_controller_polarization(self, mu):
        mu_var = mu.detach().std(dim=0)
        actual_polarization = F.relu(0.5 - mu_var, inplace=True).sum()*(200 / self.latent_dims)

        error = actual_polarization - self.polarize_percent
        P = self.Kp_beta / (1 + torch.exp(error))
        if (self.I_beta >= self.min_beta) and (self.I_beta <= self.max_beta):
            self.I_beta -= self.Ki_beta*error
        beta_old = self.beta
        self.beta = P + self.I_beta
        if self.beta > self.max_beta:
            self.beta = self.max_beta
        elif self.beta < self.min_beta:
            self.beta = self.min_beta
        return error, self.beta-beta_old

    def pi_controller_classification(self, loss, bceloss):
        loss = loss.detach()
        bceloss = bceloss.detach()
        error = self.gamma * bceloss - self.class_percent * loss

        P = self.Kp_gamma / (1 + torch.exp(error))
        if (self.I_gamma >= self.min_gamma) and (self.I_gamma <= self.max_gamma):
            self.I_gamma -= self.Ki_gamma * error

        gamma_old = self.gamma
        self.gamma = P + self.I_gamma
        if self.gamma > self.max_gamma:
            self.gamma = self.max_gamma
        elif self.gamma < self.min_gamma:
            self.gamma = self.min_gamma

        return error, self.gamma - gamma_old

    def _log_importance_w_matrix(self, batch_size, dataset_size):
        """
        Function directly taken from: https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py

        :param batch_size: batch size
        :param dataset_size: size of full dataset
        :return: importance matrix
        """
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def log_gauss_density(self, z, mu, log_var):
        """
        Function adapted from https://github.com/rtqichen/beta-tcvae/blob/master/lib/dist.py
        :param z: latent representation
        :param mu: mean of latents
        :param log_var: variance of latents
        :return: density estimate
        """
        return -0.5 * (torch.log(torch.tensor([2*torch.pi]).to(device)) +
                       log_var + (z - mu).pow(2) * torch.exp(-log_var))

    def total_correlation_decomposition_loss(self, x, x_hat, log_var, mu, z):
        """
        Function adapted from https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py
        :param x: image batch
        :param x_hat: reconstructed image batch
        :param log_var: variance of latent space batch
        :param mu: means of latent space batch
        :param z: samples from latent distributions
        :return: total correlation loss
        """
        reconstruction_loss = F.mse_loss(x_hat.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1),
                                         reduction='none').sum(dim=-1)

        log_qz_condx = self.log_gauss_density(z, mu, log_var).sum(dim=-1)
        log_prior = self.log_gauss_density(z, torch.zeros_like(z), torch.zeros_like(z)).sum(dim=-1)

        log_q_batch = self.log_gauss_density(z.reshape(z.shape[0], 1, -1),
                                             mu.reshape(1, z.shape[0], -1),
                                             log_var.reshape(1, z.shape[0], -1))

        logiw_mat = self._log_importance_w_matrix(z.shape[0], self.size)

        log_qz = torch.logsumexp(logiw_mat + log_q_batch.sum(dim=-1), dim=-1)
        log_prod_qz = torch.logsumexp(logiw_mat.reshape(z.shape[0], z.shape[0], -1) + log_q_batch, dim=1).sum(dim=-1)

        mutual_info_loss = log_qz_condx - log_qz
        total_correlation_loss = log_qz - log_prod_qz
        dim_wise_kl = log_prod_qz - log_prior

        elbo_loss = (reconstruction_loss + mutual_info_loss +
                     self.beta * total_correlation_loss + dim_wise_kl).mean(dim=0)
        kl_loss = (mutual_info_loss + self.beta * total_correlation_loss + dim_wise_kl).mean(dim=0)

        return elbo_loss, reconstruction_loss.mean(dim=0), kl_loss, (self.beta * total_correlation_loss).mean(dim=0)

    def forward(self, image_batch, output_batch):
        mu, log_var = self.encoder(image_batch)
        std = torch.exp(0.5*log_var)
        z = self.sample(mu, std)

        # update the beta parameter
        b_error, b_delta = self.pi_controller_polarization(mu)

        x_hat = self.decoder(z)
        loss, rl, kl, tc = self.total_correlation_decomposition_loss(image_batch, x_hat, log_var, mu, z)

        classification = self.classifier(z).squeeze()
        bceloss = self.bce_loss(classification, output_batch)
        g_error, g_delta = self.pi_controller_classification(loss, bceloss)
        bceloss = self.gamma * bceloss

        if self.verbose:
            return x_hat, (loss, rl, kl, tc, self.beta, b_error, b_delta, bceloss, self.gamma, g_error, g_delta)
