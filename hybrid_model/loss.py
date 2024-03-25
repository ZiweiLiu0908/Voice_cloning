import torch
import torch.nn.functional as F


def loss_function(target_audio, z_q, z_p, z_q_head, z_p_head, z_q_mu, z_q_log_var, z_p_mu, z_p_log_var,
                  generated_audio):
    # 1. Reconstruction Loss
    reconstruction_loss = F.mse_loss(generated_audio, target_audio)

    # 2. KL Divergence Loss
    z_q_log_var_clipped = torch.clamp(z_q_log_var, min=-10, max=10)  # 限制log_var在[-10, 10]范围内
    z_p_log_var_clipped = torch.clamp(z_p_log_var, min=-10, max=10)  # 限制log_var在[-10, 10]范围内
    kl_divergence_loss = 0.5 * torch.sum(torch.exp(z_q_log_var_clipped) + z_q_mu ** 2 - 1.0 - z_q_log_var_clipped)
    kl_divergence_loss += 0.5 * torch.sum(torch.exp(z_p_log_var_clipped) + z_p_mu ** 2 - 1.0 - z_p_log_var_clipped)

    # 3. z_p_head and z_p MSE Loss
    z_p_head_loss = F.mse_loss(z_p_head, z_p)

    # 4. z_q_head and z_q MSE Loss
    z_q_head_loss = F.mse_loss(z_q_head, z_q)

    total_loss = reconstruction_loss * 50 + kl_divergence_loss + z_p_head_loss * 5 + z_q_head_loss * 5

    return total_loss, reconstruction_loss * 50, kl_divergence_loss, z_p_head_loss * 5, z_q_head_loss * 5
