from opacus.accountants import RDPAccountant

def estimate_sigma(target_epsilon, target_delta, sample_rate, steps):
    from opacus.accountants.utils import get_noise_multiplier

    sigma = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=steps,
        accountant="rdp"
    )
    return sigma

# Parameters
all_epsilon = 10
total_rounds =3
total_epochs = 3
target_epsilon = 10/(total_rounds*total_epochs)
target_delta = 1e-5
CIFAR_10_SIZE = 50000  # CIFAR-10 dataset size
batch_size = 1024  # Batch size for training
sample_rate = batch_size / CIFAR_10_SIZE
steps = 2 * 3 * int(CIFAR_10_SIZE / batch_size)

sigma = estimate_sigma(target_epsilon, target_delta, sample_rate, steps)
print(f"Noise multiplier σ to achieve ε={target_epsilon}, δ={target_delta}: {sigma:.3f}")


from opacus.accountants import RDPAccountant

def compute_epsilon(sigma, sample_rate, steps, delta):
    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
    return accountant.get_epsilon(delta)

# Example:
epsilon = compute_epsilon(
    sigma=sigma,
    sample_rate=sample_rate,
    steps=2 * 3 * int(CIFAR_10_SIZE / batch_size),
    delta=1e-5
)
print("Effective ε:", epsilon)
