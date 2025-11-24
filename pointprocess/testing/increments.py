from pointprocess.testing.transformations import build_eta_on_grid, transform_T_eta_univariate, increments_from_Zhat
import numpy as np

def naive_increments(events, mu_hat, alpha_hat, beta_hat, T, n, tau, grid_size, H0):

    # Grid u in [0, tau]
    grid_u_full = np.linspace(0.0, 1.0, grid_size)
    mask = grid_u_full <= tau
    grid_u = grid_u_full[mask]

    # Empirical process η̂(u)
    eta_full, _, _ = build_eta_on_grid(
        events, mu_hat, alpha_hat, beta_hat, T, grid_u_full, H0
    )
    eta_tau = eta_full[mask]

    # Standardized empirical process W̃(u)
    mean_intensity_hat = len(events) / T
    Wtilde = eta_tau / np.sqrt(mean_intensity_hat)

    # Increments Z̃_i (eq. 24)
    Ztilde = increments_from_Zhat(Wtilde, grid_u, n, tau=tau)
    return Ztilde


def khmaladze_increments(events, mu_hat, alpha_hat, beta_hat, T, n, tau, grid_size, H0):

    # Grid u
    grid_u_full = np.linspace(0.0, 1.0, grid_size)
    mask = grid_u_full <= tau
    grid_u = grid_u_full[mask]

    # Empirical process η̂(u)
    eta_full, _, _ = build_eta_on_grid(
        events, mu_hat, alpha_hat, beta_hat, T, grid_u_full, H0
    )
    eta_tau = eta_full[mask]

    # Khmaladze transform Ŵ(u)
    mean_intensity_hat = len(events) / T
    What = transform_T_eta_univariate(eta_tau, grid_u, mean_intensity_hat)

    # Increments Ẑ_i
    Zhat = increments_from_Zhat(What, grid_u, n, tau=tau)
    return Zhat
