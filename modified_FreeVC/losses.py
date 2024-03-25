import torch 
from torch.nn import functional as F

import commons


def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_divergence(mu_p, logvar_p, mu_q, logvar_q):
    """
    Calculate KL divergence between two diagonal multivariate normal distributions
    with diagonal covariance matrices.
    
    Parameters:
        mu_p (Tensor): Mean of distribution P.
        logvar_p (Tensor): Log variance of distribution P.
        mu_q (Tensor): Mean of distribution Q.
        logvar_q (Tensor): Log variance of distribution Q.
        
    Returns:
        Tensor: KL divergence between distribution P and Q.
    """
    var_p = torch.exp(logvar_p)
    var_q = torch.exp(logvar_q)
    
    kl_div = 0.5 * torch.sum((var_p / var_q + (mu_q - mu_p)**2 / var_q - 1 - logvar_p + logvar_q), dim=1)
    
    return kl_div

def kl_loss(z_p, m_p, logs_p, z_mask, multi_samples = False):
  """
  z_p: [32, b, h, t_t]
  logs_q, m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()
  
  # print("logs_p.shape",logs_p.shape)
  
  # if not multi_samples:
  #   kl = logs_p - logs_q - 0.5
  #   # Here, z_p should be the mean, why not use m_q?
  #   # It's measuring the difference bewteen z' and mu_p, logs_p
  #   kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)

  # Assuming z is your tensor with size [N, a, b, c]
  N, b, h, t_t = z_p.size()  # Get the sizes dynamically
  # N is the number of samples
  # b is the batch size

  # Initialize lists to store means and variances for each batch and each dimension
  batch_means = []
  batch_log_stds = []

  # Iterate over each batch
  for i in range(b):
    # Get the tensor corresponding to the current batch
    z_batch = z_p[:, i, :, :]  # Shape: [N, h, t_t]
    
    # Compute mean and variance along the sample dimension for each dimension separately
    mean_batch = torch.mean(z_batch, dim=0)  # Shape: [h, t_t]
    std_batch = torch.std(z_batch, dim=0)  # Shape: [h, t_t]
    
    # Compute the standard deviation and take the logarithm
    log_std_batch = torch.log(std_batch)# Compute log std
    
    # Append mean and log std to the respective lists
    batch_means.append(mean_batch)
    batch_log_stds.append(log_std_batch)

  # Convert lists to tensors for all batches
  batch_means = torch.stack(batch_means, dim=0)  # Shape: [b, h, t_t]
  batch_log_stds = torch.stack(batch_log_stds, dim=0)  # Shape: [b, h, t_t]

  # print("batch_means.shape",batch_means.shape)
  # print("batch_log_stds.shape",batch_log_stds.shape)
  # kl = logs_p - batch_log_stds - 0.5
  # kl += 0.5 * ((batch_means - m_p)**2) * torch.exp(-2. * logs_p)
  
  kl = kl_divergence(m_p, logs_p, batch_means, batch_log_stds)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l
