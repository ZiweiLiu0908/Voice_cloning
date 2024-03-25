import matplotlib.pyplot as plt


filename = "training(17).log"


target_date_time = "Epoch"


total_losses = []
reconstruction_losses = []
kl_divergence_losses = []
z_p_head_losses = []
z_q_head_losses = []
Generator_Loss = []
Discriminator_Loss = []


with open(filename, 'r') as file:
    for line in file:
        if target_date_time in line:

            parts = line.split(',')
            for part in parts:
                # if 'Total Loss' in part:
                #     total_losses.append(float(part.split(':')[1].strip()))
                # if 'Generator Loss' in part:
                #     Generator_Loss.append(float(part.split(':')[1].strip()))
                # if 'Discriminator Loss' in part:
                #     Discriminator_Loss.append(float(part.split(':')[1].strip()))
                # if 'Reconstruction Loss' in part:
                #     reconstruction_losses.append(float(part.split(':')[1].strip())/50.0)
                if 'KL Divergence Loss' in part:
                    kl_divergence_losses.append(float(part.split(':')[1].strip()))
                # if 'z_p_head Loss' in part:
                #     z_p_head_losses.append(float(part.split(':')[1].strip())/50.0)
                # elif 'z_q_head Loss' in part:
                #     z_q_head_losses.append(float(part.split(':')[1].strip())/50.0)


plt.figure(figsize=(10, 6))
# plt.plot(total_losses, label='Total Loss')
# plt.plot(Generator_Loss, label='Generator Loss')
# plt.plot(Discriminator_Loss, label='Discriminator Loss')
plt.plot(reconstruction_losses, label='Reconstruction Loss')
plt.plot(kl_divergence_losses, label='KL Divergence Loss')
plt.plot(z_p_head_losses, label='z_p_head Loss')
plt.plot(z_q_head_losses, label='z_q_head Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.title('Loss Metrics Over Time')
plt.legend()
plt.show()
