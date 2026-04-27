from src.models.NMMR.kernel_utils import calculate_kernel_matrix_batched
from src.models.NMMR.mar_imputer import imputed_residual


def NMMR_loss(model_output, target, kernel_matrix, loss_name: str):  # batch_indices=None):
    residual = target - model_output
    n = residual.shape[0]
    K = kernel_matrix

    if loss_name == "U_statistic":
        # calculate U statistic (see Serfling 1980)
        K.fill_diagonal_(0)
        loss = (residual.T @ K @ residual) / (n * (n-1))
    elif loss_name == "V_statistic":
        # calculate V statistic (see Serfling 1980)
        loss = (residual.T @ K @ residual) / (n ** 2)
    else:
        raise ValueError(f"{loss_name} is not valid. Must be 'U_statistic' or 'V_statistic'.")

    return loss[0, 0]


def NMMR_loss_mar(model_output, target, delta_w, mar_weights, kernel_matrix, loss_name: str):
    """MAR-imputed U-/V-statistic loss (paper §4.4).

    R̂_MAR(θ) = (1 / N) · Σ_{i,j} r̃_i · r̃_j · k(L_i, L_j)

    where r̃_i = δ_i · r_i + (1 − δ_i) · Σ_j W[i,j] · r_j and N is
    n(n−1) for the U-statistic (diagonal zeroed) or n² for the V-statistic.

    Reduces exactly to the upstream NMMR_loss when δ ≡ 1.
    """
    residual = target - model_output
    r_tilde = imputed_residual(residual, delta_w, mar_weights)
    n = r_tilde.shape[0]
    K = kernel_matrix

    if loss_name == "U_statistic":
        K.fill_diagonal_(0)
        loss = (r_tilde.T @ K @ r_tilde) / (n * (n - 1))
    elif loss_name == "V_statistic":
        loss = (r_tilde.T @ K @ r_tilde) / (n ** 2)
    else:
        raise ValueError(f"{loss_name} is not valid. Must be 'U_statistic' or 'V_statistic'.")

    return loss[0, 0]


def NMMR_loss_batched(model_output, target, kernel_inputs, kernel, batch_size: int, loss_name: str):
    residual = target - model_output
    n = residual.shape[0]

    loss = 0
    for i in range(0, n, batch_size):
        partial_kernel_matrix = calculate_kernel_matrix_batched(kernel_inputs, (i, i+batch_size), kernel)
        if loss_name == "V_statistic":
            factor = n ** 2
        if loss_name == "U_statistic":
            factor = n * (n-1)
            # zero out the main diagonal of the full matrix
            for row_idx in range(partial_kernel_matrix.shape[0]):
                partial_kernel_matrix[row_idx, row_idx+i] = 0
        temp_loss = residual[i:(i+batch_size)].T @ partial_kernel_matrix @ residual / factor
        loss += temp_loss[0, 0]
    return loss
