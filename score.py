import torch
import torch.nn as nn
import numpy as np


class ScoreNet(nn.Module):
    """Score matching model"""

    def __init__(self, scorenet, sigma_begin, sigma_end, noise_level, sigma_type='geometric'):
        """
        :param scorenet: an `nn.Module` instance that computes the score of the input images
        :param sigma_begin: the largest sigma value
        :param sigma_end: the smallest sigma value
        :param noise_level: the number of noise levels
        :param sigma_type: the type of sigma distribution, 'geometric' or 'linear'
        """
        super().__init__()
        self.scorenet = scorenet

        self.sigmas: torch.Tensor
        sigmas = self.get_sigmas(sigma_begin, sigma_end, noise_level, sigma_type)
        self.register_buffer('sigmas', sigmas)  # (num_noise_level,)

    @staticmethod
    def get_sigmas(sigma_begin, sigma_end, noise_level, sigma_type='geometric'):
        """
        Get the sigmas used to perturb the images
        :param sigma_begin: the largest sigma value
        :param sigma_end: the smallest sigma value
        :param noise_level: the number of noise levels
        :param sigma_type: the type of sigma distribution, 'geometric' or 'linear'
        :return: sigmas of shape (num_noise_level,)
        """
        if sigma_type == 'geometric':
            sigmas = torch.FloatTensor(np.geomspace(
                sigma_begin, sigma_end,
                noise_level
            ))
        elif sigma_type == 'linear':
            sigmas = torch.FloatTensor(np.linspace(
                sigma_begin, sigma_end, noise_level
            ))
        else:
            raise NotImplementedError(f'sigma distribution {sigma_type} not supported')
        return sigmas

    def perturb(self, batch):
        """
        Perturb images with Gaussian noise.
        You should randomly choose a sigma from `self.sigmas` for each image in the batch.
        Use that sigma as the standard deviation of the Gaussian noise added to the image.
        :param batch: batch of images of shape (N, D)
        :return: noises added to images (N, D)
                 sigmas used to perturb the images (N, 1)
        """
        batch_size = batch.size(0)
        device = batch.device
    
        # Randomly select a sigma for each sample in the batch
        indices = torch.randint(low=0, high=len(self.sigmas), size=(batch_size,), device=device)
        used_sigmas = self.sigmas[indices].view(batch_size, 1)  # Shape: (N, 1)
    
        # Generate Gaussian noise with the selected sigmas
        noise = torch.randn_like(batch) * used_sigmas  # Shape: (N, D)
    
        # Return the added noise and the used sigmas
        return noise, used_sigmas


    @torch.no_grad()
    def sample(self, batch_size, img_size, sigmas=None, n_steps_each=10, step_lr=2e-5):
        self.eval()
    
        if sigmas is None:
            sigmas = self.sigmas
        else:
            sigmas = sigmas.to(self.sigmas.device)
    
        if not isinstance(sigmas, torch.Tensor) or sigmas.numel() == 0:
            raise ValueError("`sigmas` must be a non-empty torch.Tensor")
    
        x = torch.rand(batch_size, img_size, device=sigmas.device)
    
        traj = []
        print(f"sigmas: {sigmas}")
    
        for sigma in sigmas:
            sigma = sigma.to(x.device)
            print(f"Processing sigma: {sigma.item()}")
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
    
            for step in range(n_steps_each):
                print(f"Step: {step}")
                try:
                    score = self.get_score(x, sigma)
                    noise = torch.randn_like(x)
                    x = x + step_size * score + (2 * step_size) ** 0.5 * noise
                    traj.append(x.clone())
                except Exception as e:
                    print(f"Exception at step {step}: {e}")
                    raise
    
        if len(traj) == 0:
            raise RuntimeError("Trajectory is empty. Loops may not be executing properly.")
    
        traj = torch.stack(traj, dim=0).view(len(sigmas), n_steps_each, *x.size())
        return traj






    def get_score(self, x, sigma):
        """
        Calculate the score of the input images
        :param x: images of (N, D)
        :param sigma: the sigma used to perturb the images, either a float or a tensor of shape (N, 1)
        :return: the score of the input images, of shape (N, D)
        """
        # In NCSNv2, the score is divided by sigma (i.e., noise-conditioned)
        out = self.scorenet(x) / sigma
        return out

    def get_loss(self, x):
        """
        Calculate the score loss.
        The loss should be averaged over the batch dimension and the image dimension.
        :param x: images of (N, D)
        :return: score loss, a scalar tensor
        """
        # Obtain noise and perturbed data
        noise, sigmas = self.perturb(x)
        perturbed_data = x + noise  # Add noise to original data to get perturbed data
    
        # Obtain the estimated scores for the perturbed data
        estimated_scores = self.get_score(perturbed_data, sigmas)
    
        # Compute the target score: - (noise / sigma^2)
        target_scores = -noise / (sigmas ** 2)
    
        # Compute the score loss
        loss = 0.5 * torch.mean(((estimated_scores - target_scores) ** 2) * (sigmas ** 2))
    
        return loss


    def forward(self, x):
        """
        Calculate the result of the score net (not noise-conditioned)
        :param x: images of (N, D)
        :return: the result of the score net, of shape (N, D)
        """
        return self.scorenet(x)
