import os
import numpy as np
import cupy as cp
import polars as pl
from cupyx.scipy.special import ndtr
from cupyx.scipy.special import chdtrc
from cupyx.scipy.special import gammaincc
import argparse
import psutil
import threading
import time

def transform_to_pvalues(z_scores_matrix):
    z_scores = cp.asarray(z_scores_matrix, dtype=cp.float32)
    z_64 = z_scores.astype(cp.float64)

    p1 = cp.where(z_64 > 0, ndtr(-z_64), 1.0)
    p2 = cp.where(z_64 < 0, ndtr(z_64), 1.0)

    gpu_tiny = 1e-300
    p1 = cp.clip(p1, gpu_tiny, 1.0)
    p2 = cp.clip(p2, gpu_tiny, 1.0)

    return p1, p2


def meanp(values_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    if values_matrix.ndim == 1:
        values_matrix = values_matrix[None, :]

    k = values_matrix.shape[1]
    gpu_tiny = 1e-300

    if data_type == "z-scores":
        p1, p2 = transform_to_pvalues(values_matrix)

        z1 = (0.5 - cp.mean(p1, axis=1)) * cp.sqrt(12.0 * k)
        z2 = (0.5 - cp.mean(p2, axis=1)) * cp.sqrt(12.0 * k)

        combined_p1 = 1.0 - ndtr(z1)
        combined_p2 = 1.0 - ndtr(z2)
        p_final = 2.0 * cp.minimum(combined_p1, combined_p2)
    else:
        p_values = cp.clip(values_matrix, gpu_tiny, 1.0)
        z = (0.5 - cp.mean(p_values, axis=1)) * cp.sqrt(12.0 * k)
        p_final = 1.0 - ndtr(z)

    return cp.clip(p_final, gpu_tiny, 1.0)


def fisher_method(values_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    k = values_matrix.shape[1]
    df = 2.0 * k
    gpu_tiny = 1e-300

    if data_type == "z-scores":
        p1, p2 = transform_to_pvalues(values_matrix)
        chi_squared1 = -2.0 * cp.sum(cp.log(p1), axis=1)
        chi_squared2 = -2.0 * cp.sum(cp.log(p2), axis=1)
        combined_p1 = chdtrc(df, chi_squared1)
        combined_p2 = chdtrc(df, chi_squared2)
        p_final = 2.0 * cp.minimum(combined_p1, combined_p2)
    else:
        p_values = cp.clip(values_matrix, gpu_tiny, 1.0)
        chi_squared = -2.0 * cp.sum(cp.log(p_values), axis=1)
        p_final = chdtrc(df, chi_squared)

    return cp.clip(p_final, gpu_tiny, 1.0)


def logit(values_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    k = values_matrix.shape[1]
    C = cp.sqrt(k * cp.pi ** 2 * (5.0 * k + 2.0) / (3.0 * (5.0 * k + 4.0)))
    df = 2.0 * k
    gpu_tiny = 1e-300

    from cupyx.scipy.special import betainc

    if data_type == "z-scores":
        p1, p2 = transform_to_pvalues(values_matrix)
        t1_value = -cp.sum(cp.log(p1) - cp.log(p2), axis=1)
        t2_value = -cp.sum(cp.log(p2) - cp.log(p1), axis=1)
        x1 = df / (df + (t1_value / C) ** 2)
        x2 = df / (df + (t2_value / C) ** 2)
        combined_p1 = betainc(df / 2.0, 0.5, x1)
        combined_p2 = betainc(df / 2.0, 0.5, x2)
        p_final = 2.0 * cp.minimum(combined_p1, combined_p2)
    else:
        p_values = cp.clip(values_matrix, gpu_tiny, 1.0 - gpu_tiny)
        t_value = -cp.sum(cp.log(p_values) - cp.log(1.0 - p_values), axis=1)
        x = df / (df + (t_value / C) ** 2)
        p_final = betainc(df / 2.0, 0.5, x)

    return cp.clip(p_final, gpu_tiny, 1.0)


def stouffer(values_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    k = values_matrix.shape[1]
    gpu_tiny = 1e-300

    if data_type == "p-values":
        from cupyx.scipy.special import ndtri
        z_scores = -ndtri(cp.clip(values_matrix, gpu_tiny, 1.0))
    else:
        z_scores = values_matrix

    combined_z = cp.sum(z_scores, axis=1) / cp.sqrt(k)
    combined_p = ndtr(-combined_z)

    return cp.clip(combined_p, gpu_tiny, 1.0)


def weighted_stouffer(values_matrix, data_matrix, data_type="z-scores", weight_matrix=None):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    k = values_matrix.shape[1]
    gpu_tiny = 1e-300

    if weight_matrix is None:
        weights = cp.ones(k, dtype=cp.float64)
    else:
        weights = cp.asarray(weight_matrix, dtype=cp.float64).flatten()

    if data_type == "p-values":
        from cupyx.scipy.special import ndtri
        z_scores = -ndtri(cp.clip(values_matrix, gpu_tiny, 1.0))
    else:
        z_scores = values_matrix

    weighted_z = cp.sum(weights * z_scores, axis=1)
    weighted_std_dev = cp.sqrt(cp.sum(weights ** 2))

    combined_z = weighted_z / weighted_std_dev
    combined_p = ndtr(-combined_z)

    return cp.clip(combined_p, gpu_tiny, 1.0)


def inverse_chi2(values_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    k = values_matrix.shape[1]
    df = k
    gpu_tiny = 1e-300

    from cupyx.scipy.special import chdtri

    if data_type == "z-scores":
        p1, p2 = transform_to_pvalues(values_matrix)
        chi_squared1 = cp.sum(chdtri(1.0, p1), axis=1)
        combined_p1 = chdtrc(df, chi_squared1)
        chi_squared2 = cp.sum(chdtri(1.0, p2), axis=1)
        combined_p2 = chdtrc(df, chi_squared2)
        p_final = 2.0 * cp.minimum(combined_p1, combined_p2)
    else:
        p_values = cp.clip(values_matrix, gpu_tiny, 1.0)
        chi_squared = cp.sum(chdtri(1.0, p_values), axis=1)
        p_final = chdtrc(df, chi_squared)

    return cp.clip(p_final, gpu_tiny, 1.0)


def lancaster_method(values_matrix, data_matrix, data_type="z-scores", weight_matrix=None):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    k = values_matrix.shape[1]
    gpu_tiny = 1e-300

    if weight_matrix is None:
        weights = cp.ones(k, dtype=cp.float64)
    else:
        weights = cp.asarray(weight_matrix, dtype=cp.float64).flatten()

    df_total = cp.sum(weights)

    from cupyx.scipy.special import chdtri, chdtrc

    if data_type == "z-scores":
        p1, p2 = transform_to_pvalues(values_matrix)
        chi_statistic1 = cp.sum(chdtri(weights, p1), axis=1)
        combined_p1 = chdtrc(df_total, chi_statistic1)
        chi_statistic2 = cp.sum(chdtri(weights, p2), axis=1)
        combined_p2 = chdtrc(df_total, chi_statistic2)
        p_final = 2.0 * cp.minimum(combined_p1, combined_p2)
    else:
        p_values = cp.clip(values_matrix, gpu_tiny, 1.0)
        chi_statistic = cp.sum(chdtri(weights, p_values), axis=1)
        p_final = chdtrc(df_total, chi_statistic)

    return cp.clip(p_final, gpu_tiny, 1.0)


def binomial_test(values_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    k = values_matrix.shape[1]
    alpha = 0.05
    gpu_tiny = 1e-300

    from cupyx.scipy.special import bdtrc

    if data_type == "z-scores":
        p1, p2 = transform_to_pvalues(values_matrix)
        r1 = cp.sum(p1 < alpha, axis=1)
        r2 = cp.sum(p2 < alpha, axis=1)
        combined_p1 = cp.where(r1 > 0, bdtrc(r1 - 1, k, alpha), 1.0)
        combined_p2 = cp.where(r2 > 0, bdtrc(r2 - 1, k, alpha), 1.0)
        p_final = 2.0 * cp.minimum(combined_p1, combined_p2)
    else:
        r = cp.sum(values_matrix < alpha, axis=1)
        p_final = cp.where(r > 0, bdtrc(r - 1, k, alpha), 1.0)

    return cp.clip(p_final, gpu_tiny, 1.0)


def cauchy_method(values_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    k = values_matrix.shape[1]
    gpu_tiny = 1e-300

    if data_type == "z-scores":
        p1, p2 = transform_to_pvalues(values_matrix)
        T1 = cp.tan((0.5 - p1) * cp.pi)
        T2 = cp.tan((0.5 - p2) * cp.pi)
        t1 = cp.sum(T1, axis=1) / k
        t2 = cp.sum(T2, axis=1) / k
        combined_p1 = 0.5 - (cp.arctan(t1) / cp.pi)
        combined_p2 = 0.5 - (cp.arctan(t2) / cp.pi)
        p_final = 2.0 * cp.minimum(combined_p1, combined_p2)
    else:
        p_values = cp.clip(values_matrix, gpu_tiny, 1.0)
        T = cp.tan((0.5 - p_values) * cp.pi)
        t = cp.sum(T, axis=1) / k
        p_final = 0.5 - (cp.arctan(t) / cp.pi)

    return cp.clip(p_final, gpu_tiny, 1.0)


def minP(values_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    k = values_matrix.shape[1]
    gpu_tiny = 1e-300

    if data_type == "z-scores":
        p1, p2 = transform_to_pvalues(values_matrix)
        min_p1 = cp.min(p1, axis=1)
        min_p2 = cp.min(p2, axis=1)
        combined_p1 = 1.0 - (1.0 - min_p1) ** k
        combined_p2 = 1.0 - (1.0 - min_p2) ** k
        p_final = 2.0 * cp.minimum(combined_p1, combined_p2)
    else:
        p_values = cp.clip(values_matrix, gpu_tiny, 1.0)
        min_p = cp.min(p_values, axis=1)
        p_final = 1.0 - (1.0 - min_p) ** k

    return cp.clip(p_final, gpu_tiny, 1.0)


def CMC(values_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    gpu_tiny = 1e-300

    p_value_cauchy = cauchy_method(values_matrix, data_type)
    p_value_minp = minP(values_matrix, data_type)

    combined_values = cp.vstack((p_value_cauchy, p_value_minp))
    k = 2.0

    combined_values = cp.clip(combined_values, gpu_tiny, 1.0)
    T = cp.tan((0.5 - combined_values) * cp.pi)
    t = cp.sum(T, axis=0) / k
    combined_p = 0.5 - (cp.arctan(t) / cp.pi)

    return cp.clip(combined_p, gpu_tiny, 1.0)


def MCM(values_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    gpu_tiny = 1e-300

    p_value_cauchy = cauchy_method(values_matrix, data_type)
    p_value_minp = minP(values_matrix, data_type)

    min_1 = cp.minimum(p_value_cauchy, p_value_minp)
    min_final = cp.minimum(min_1, 0.5)

    combined_p = 2.0 * min_final
    return cp.clip(combined_p, gpu_tiny, 1.0)


def HMP(values_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    k = values_matrix.shape[1]
    gpu_tiny = 1e-300

    if data_type == "z-scores":
        p1, p2 = transform_to_pvalues(values_matrix)
        p1 = cp.clip(p1, gpu_tiny, 1.0)
        p2 = cp.clip(p2, gpu_tiny, 1.0)

        combined_p1 = k / cp.sum(1.0 / p1, axis=1)
        combined_p2 = k / cp.sum(1.0 / p2, axis=1)
        p_final = 2.0 * cp.minimum(combined_p1, combined_p2)
    else:
        p_values = cp.clip(values_matrix, gpu_tiny, 1.0)
        p_final = k / cp.sum(1.0 / p_values, axis=1)

    return cp.clip(p_final, gpu_tiny, 1.0)


def EmpiricalBrownsMethod(values_matrix, data_matrix, data_type="z-scores", custom_cov_matrix=None):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    data_matrix = cp.asarray(data_matrix, dtype=cp.float64)

    k = data_matrix.shape[1]
    Expected = 2.0 * k
    df_fisher = 2.0 * k
    gpu_tiny = 1e-300

    if data_type == "z-scores":
        p1, p2 = transform_to_pvalues(values_matrix)
        log_p1 = -cp.log(cp.clip(p1, gpu_tiny, 1.0))
        log_p2 = -cp.log(cp.clip(p2, gpu_tiny, 1.0))

        covar_matrix1 = custom_cov_matrix if custom_cov_matrix is not None else cp.cov(log_p1, rowvar=False)
        covar_matrix2 = custom_cov_matrix if custom_cov_matrix is not None else cp.cov(log_p2, rowvar=False)

        off_diag_sum1 = cp.sum(covar_matrix1) - cp.trace(covar_matrix1)
        off_diag_sum2 = cp.sum(covar_matrix2) - cp.trace(covar_matrix2)

        Var1 = 4.0 * k + off_diag_sum1
        Var2 = 4.0 * k + off_diag_sum2

        c1 = Var1 / (2.0 * Expected)
        df_brown1 = min((2.0 * Expected ** 2) / Var1, df_fisher)
        c1 = 1.0 if df_brown1 == df_fisher else c1

        c2 = Var2 / (2.0 * Expected)
        df_brown2 = min((2.0 * Expected ** 2) / Var2, df_fisher)
        c2 = 1.0 if df_brown2 == df_fisher else c2

        chi_squared1 = 2.0 * cp.sum(log_p1, axis=1)
        chi_squared2 = 2.0 * cp.sum(log_p2, axis=1)

        combined_p1 = chdtrc(df_brown1, chi_squared1 / c1)
        combined_p2 = chdtrc(df_brown2, chi_squared2 / c2)

        p_brown_final = 2.0 * cp.minimum(combined_p1, combined_p2)

    else:
        log_p = -cp.log(cp.clip(values_matrix, gpu_tiny, 1.0))
        covar_matrix = custom_cov_matrix if custom_cov_matrix is not None else cp.cov(log_p, rowvar=False)
        off_diag_sum = cp.sum(covar_matrix) - cp.trace(covar_matrix)
        Var = 4.0 * k + off_diag_sum

        c = Var / (2.0 * Expected)
        df_brown = min((2.0 * Expected ** 2) / Var, df_fisher)
        c = 1.0 if df_brown == df_fisher else c

        x = 2.0 * cp.sum(log_p, axis=1)
        p_brown_final = chdtrc(df_brown, x / c)

    return cp.clip(p_brown_final, gpu_tiny, 1.0)


def KostsMethod(values_matrix, data_matrix, data_type="z-scores"):
    data_matrix_64 = cp.asarray(data_matrix, dtype=cp.float64)
    cor = cp.corrcoef(data_matrix_64, rowvar=False)
    covar_matrix = 3.263 * cor + 0.710 * cor ** 2 + 0.027 * cor ** 3
    cp.fill_diagonal(covar_matrix, 0.0)
    return EmpiricalBrownsMethod(values_matrix, data_matrix_64, data_type, custom_cov_matrix=covar_matrix)


def BrownsMethodbyYang(values_matrix, data_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    data_matrix_64 = cp.asarray(data_matrix, dtype=cp.float64)
    k = data_matrix_64.shape[1]
    c1 = 3.9081
    gpu_tiny = 1e-300

    if data_type == "z-scores":
        abs_z = cp.abs(values_matrix)
        p_values = 2.0 * ndtr(-abs_z)
    else:
        p_values = values_matrix

    cor = cp.corrcoef(data_matrix_64, rowvar=False)
    cp.fill_diagonal(cor, 1.0)

    bcc = cor * (1.0 + (1.0 - cor ** 2) / (2.0 * (k - 3.0)))
    f_r = 3.9081 * bcc ** 2 + 0.0313 * bcc ** 4 + 0.1022 * bcc ** 6 - 0.1378 * bcc ** 8 + 0.0941 * bcc ** 10
    bias = (c1 / k) * (1.0 - bcc ** 2) ** 2

    delta_matrix = f_r - bias
    cp.fill_diagonal(delta_matrix, 0.0)

    delta_sum = cp.sum(delta_matrix)
    mean_val = 2.0 * k
    Var = 4.0 * k + delta_sum

    v_gamma = 2.0 * (mean_val ** 2 / Var)
    gamma_scale = Var / mean_val

    T = 2.0 * cp.sum(-cp.log(cp.clip(p_values, gpu_tiny, 1.0)), axis=1)
    p_yang_final = gammaincc(v_gamma / 2.0, T / (2.0 * gamma_scale))

    return cp.clip(p_yang_final, gpu_tiny, 1.0)


def correlated_Stouffer(values_matrix, data_matrix, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    data_matrix_64 = cp.asarray(data_matrix, dtype=cp.float64)
    k = data_matrix_64.shape[1]
    gpu_tiny = 1e-300

    if data_type == "p-values":
        from cupyx.scipy.special import ndtri
        z_scores = -ndtri(cp.clip(values_matrix, gpu_tiny, 1.0))
    else:
        z_scores = values_matrix

    cor_matrix = cp.corrcoef(data_matrix_64, rowvar=False)
    cor_sum = (cp.sum(cor_matrix) - k) / 2.0
    total_variance = k + 2.0 * cor_sum

    combined_z = cp.sum(z_scores, axis=1) / cp.sqrt(total_variance)
    combined_p_value = ndtr(-combined_z)

    return cp.clip(combined_p_value, gpu_tiny, 1.0)


def weighted_correlated_Stouffer(values_matrix, data_matrix, data_type="z-scores", weight_matrix=None):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    data_matrix_64 = cp.asarray(data_matrix, dtype=cp.float64)
    k = data_matrix_64.shape[1]
    gpu_tiny = 1e-300

    if weight_matrix is None:
        weights = cp.ones(k, dtype=cp.float64)
    else:
        weights = cp.asarray(weight_matrix, dtype=cp.float64).flatten()

    if data_type == "p-values":
        from cupyx.scipy.special import ndtri
        z_scores = -ndtri(cp.clip(values_matrix, gpu_tiny, 1.0))
    else:
        z_scores = values_matrix

    cor_matrix = cp.corrcoef(data_matrix_64, rowvar=False)
    weighted_z = cp.sum(weights * z_scores, axis=1)

    weights_2d = weights[:, None] * weights[None, :]
    weighted_variance = cp.sum(weights_2d * cor_matrix)
    weighted_std_dev = cp.sqrt(weighted_variance)

    combined_z = weighted_z / weighted_std_dev
    combined_p_value = ndtr(-combined_z)

    return cp.clip(combined_p_value, gpu_tiny, 1.0)


def Bonferronis_correction_g(data_matrix):
    data_matrix_64 = cp.asarray(data_matrix, dtype=cp.float64)
    k = data_matrix_64.shape[1]
    cor = cp.corrcoef(data_matrix_64, rowvar=False)
    cp.fill_diagonal(cor, -cp.inf)
    ICC = cp.max(cor)
    g_adjusted = (k + 1.0) - (1.0 + (k - 1.0) * ICC)
    return float(g_adjusted)


def get_eigenvalues(data_matrix, correlation_matrix_path=None):
    if correlation_matrix_path:
        cor = read_matrix_from_file(correlation_matrix_path)
    else:
        data_matrix_64 = cp.asarray(data_matrix, dtype=cp.float64)
        cor = cp.corrcoef(data_matrix_64, rowvar=False)
    eigenvalues = cp.linalg.eigvalsh(cor)
    return cp.sort(eigenvalues)[::-1]


def effective_number_of_tests_cheverud_nyholt(eigenvalues):
    k = len(eigenvalues)
    sample_variance = cp.var(eigenvalues)
    return float(1.0 + (k - 1.0) * (1.0 - sample_variance / k))


def h_function_gpu(x_array):
    floor_x = cp.floor(x_array)
    return cp.where(x_array >= 1.0, 1.0 + (x_array - floor_x), x_array - floor_x)


def effective_number_of_tests_li_ji(eigenvalues):
    abs_eigenvalues = cp.abs(eigenvalues)
    return float(cp.sum(h_function_gpu(abs_eigenvalues)))


def effective_number_of_tests_gao(eigenvalues, C=0.995):
    total_sum = cp.sum(eigenvalues)
    cumulative_sum = cp.cumsum(eigenvalues)
    ratio = cumulative_sum / total_sum
    idx = cp.where(ratio > C)[0]
    if len(idx) > 0:
        return float(idx[0] + 1)
    return float(len(eigenvalues))


def effective_number_of_tests_galwey(eigenvalues):
    lambda_prime = cp.maximum(0.0, eigenvalues)
    sum_squared_lambda_prime = (cp.sum(cp.sqrt(lambda_prime))) ** 2
    return float(sum_squared_lambda_prime / cp.sum(lambda_prime))


def bonferroni_method_with_effective_tests(values_matrix, effective_number_of_tests, data_type="z-scores"):
    values_matrix = cp.asarray(values_matrix, dtype=cp.float64)
    gpu_tiny = 1e-300

    if data_type == "z-scores":
        p1, p2 = transform_to_pvalues(values_matrix)
        min_p1 = cp.min(p1, axis=1)
        min_p2 = cp.min(p2, axis=1)
        combined_p1 = cp.minimum(1.0, min_p1 * effective_number_of_tests)
        combined_p2 = cp.minimum(1.0, min_p2 * effective_number_of_tests)
        p_final = 2.0 * cp.minimum(combined_p1, combined_p2)
    else:
        p_values = cp.clip(values_matrix, gpu_tiny, 1.0)
        min_p = cp.min(p_values, axis=1)
        p_final = cp.minimum(1.0, min_p * effective_number_of_tests)

    return cp.clip(p_final, gpu_tiny, 1.0)


def default():
    print("Invalid choice")


def read_matrix_from_file(input_file_path):
    with open(input_file_path, 'r') as file:
        return cp.array([[float(value) for value in line.strip().split()] for line in file], dtype=cp.float64)


def read_data_hybrid(file_path, data_type="z-scores"):
    print(f"\n[*] Polars Lazy Engine: Building execution plan for {file_path}...")
    lf = pl.scan_csv(file_path, has_header=False, separator=' ')

    print("[*] Executing Lazy Plan & Streaming to RAM...")
    df = lf.collect()

    snp_list = df.get_column("column_1").to_list()
    math_df = df.drop("column_1")

    if data_type == "p-values":
        np_dtype = np.float64
        cp_dtype = cp.float64
        print("[*] Data Type: P-Values detected. Enforcing strict Float64 precision.")
    else:
        np_dtype = np.float32
        cp_dtype = cp.float32
        print("[*] Data Type: Z-Scores detected. Using Float32 for memory optimization.")

    cpu_matrix = math_df.to_numpy().astype(np_dtype)

    num_snps, num_studies = cpu_matrix.shape
    print(f"[*] Memory Bridge: Transferring {num_snps} SNPs x {num_studies} studies to GPU...")

    gpu_matrix = cp.asarray(cpu_matrix, dtype=cp_dtype)

    print(f"[+] Transfer Complete. VRAM Payload: {gpu_matrix.nbytes / (1024 ** 2):.2f} MB")

    return snp_list, gpu_matrix


def save_results_hybrid(snp_list, p_values_gpu, method_name, output_file_path):
    print(f"[*] Memory Bridge: Pulling {len(snp_list)} results back to CPU...")

    p_values_gpu = p_values_gpu.astype(cp.float64)
    gpu_tiny = 1e-300
    p_values_gpu = cp.clip(p_values_gpu, gpu_tiny, 1.0)

    p_values_cpu = cp.asnumpy(p_values_gpu)

    df_result = pl.DataFrame({
        "SNP": snp_list,
        "Combined_P": p_values_cpu
    })

    base_name, ext = os.path.splitext(output_file_path)
    out_file = f"{base_name}_{method_name}{ext}"

    print(f"[*] SSD Streamer: Writing to {out_file}...")
    df_result.write_csv(out_file, separator="\t")
    print(f"[+] {method_name} Complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text files and data type.')
    parser.add_argument('input_file_path', type=str, help='Path to the input file')
    parser.add_argument('output_file_path', type=str, help='Path to the output file')
    parser.add_argument('data_type', type=str, choices=['p-values', 'z-scores'],
                        help="Type of data in the file ('p-values' or 'z-scores')")
    parser.add_argument('meta_choice', type=str, choices=['Yes', 'No'],
                        help="Do you want the program to perform meta-analysis? ('Yes' or 'No')")
    parser.add_argument('correlation_matrix_path', type=str, nargs='?', default=None,
                        help="Path to the correlation matrix file (optional)")
    parser.add_argument('weights_matrix_path', type=str, nargs='?', default=None,
                        help="Path to the correlation matrix file (optional)")
    parser.add_argument('methods', type=str, nargs='+',
                        choices=['logit', 'meanp', 'fisher', 'lancaster', 'stouffer', 'wstouffer', 'invchi', 'binomial',
                                 'cct', 'minp', 'mcm', 'hmp', 'cmc', 'bonferroni', 'ebm', 'kost', 'yang', 'corstouffer',
                                 'wcorstouffer'],
                        help="Which method(s) would you like to use to combine your p-values?")
    parser.add_argument('--runs', type=int, default=1, help='Number of benchmark iterations (default: 1)')
    args = parser.parse_args()

    time_records = []
    ram_records = []

    for run in range(args.runs):
        if args.runs > 1:
            print(f"\n{'=' * 50}")
            print(f"BENCHMARK RUN {run + 1} OF {args.runs}")
            print(f"{'=' * 50}")


        class MemoryMonitor:
            def __init__(self):
                self.keep_measuring = True
                self.peak_memory = 0
                self.process = psutil.Process(os.getpid())

            def measure_memory(self):
                while self.keep_measuring:
                    try:
                        total_mem = self.process.memory_info().rss
                        for child in self.process.children(recursive=True):
                            try:
                                total_mem += child.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                        current_mem_mb = total_mem / (1024 * 1024)
                        if current_mem_mb > self.peak_memory:
                            self.peak_memory = current_mem_mb
                    except psutil.NoSuchProcess:
                        pass
                    import time
                    time.sleep(0.025)


        monitor = MemoryMonitor()
        mem_thread = threading.Thread(target=monitor.measure_memory, daemon=True)
        mem_thread.start()

        run_start_time = time.time()

        t_read_start = time.time()
        snp_list, gpu_data_matrix = read_data_hybrid(args.input_file_path, args.data_type)
        t_read_end = time.time()
        print(f"[TIME] Polars Read & PCIe Transfer took: {t_read_end - t_read_start:.2f} seconds")

        metanalysis_needed = args.meta_choice.lower() == 'yes'

        weights_matrix = None
        if 'wcorstouffer' in args.methods or 'wstouffer' in args.methods:
            if args.weights_matrix_path:
                weights_matrix = read_matrix_from_file(args.weights_matrix_path)

        method_functions = {
            'logit': logit, 'meanp': meanp, 'fisher': fisher_method, 'lancaster': lancaster_method,
            'stouffer': stouffer, 'wstouffer': weighted_stouffer, 'invchi': inverse_chi2,
            'binomial': binomial_test, 'cct': cauchy_method, 'minp': minP, 'cmc': CMC,
            'mcm': MCM, 'hmp': HMP, 'ebm': EmpiricalBrownsMethod, 'kost': KostsMethod,
            'yang': BrownsMethodbyYang, 'bonferroni': bonferroni_method_with_effective_tests,
            'corstouffer': correlated_Stouffer, 'wcorstouffer': weighted_correlated_Stouffer
        }
        selected_methods = args.methods

        for method in selected_methods:
            if method in method_functions:
                combine_function = method_functions[method]
                t_math_start = time.time()

                if method == 'ebm':
                    combined_p_gpu = combine_function(gpu_data_matrix, gpu_data_matrix, args.data_type)
                elif method in ['kost', 'yang', 'corstouffer']:
                    combined_p_gpu = combine_function(gpu_data_matrix, gpu_data_matrix, args.data_type)
                elif method in ['lancaster', 'wstouffer', 'wcorstouffer']:
                    combined_p_gpu = combine_function(gpu_data_matrix, gpu_data_matrix, args.data_type, weights_matrix)
                elif method == 'bonferroni':
                    eigenvalues = get_eigenvalues(gpu_data_matrix, args.correlation_matrix_path)
                    cn = effective_number_of_tests_cheverud_nyholt(eigenvalues)
                    gao = effective_number_of_tests_gao(eigenvalues)
                    gal = effective_number_of_tests_galwey(eigenvalues)
                    li_ji = effective_number_of_tests_li_ji(eigenvalues)
                    g_adj = Bonferronis_correction_g(gpu_data_matrix)

                    p_cn = bonferroni_method_with_effective_tests(gpu_data_matrix, cn, args.data_type)
                    save_results_hybrid(snp_list, p_cn, "bonf_CN", args.output_file_path)
                    p_gao = bonferroni_method_with_effective_tests(gpu_data_matrix, gao, args.data_type)
                    save_results_hybrid(snp_list, p_gao, "bonf_Gao", args.output_file_path)
                    p_gal = bonferroni_method_with_effective_tests(gpu_data_matrix, gal, args.data_type)
                    save_results_hybrid(snp_list, p_gal, "bonf_Galwey", args.output_file_path)
                    p_li_ji = bonferroni_method_with_effective_tests(gpu_data_matrix, li_ji, args.data_type)
                    save_results_hybrid(snp_list, p_li_ji, "bonf_LiJi", args.output_file_path)
                    p_bc = bonferroni_method_with_effective_tests(gpu_data_matrix, g_adj, args.data_type)
                    save_results_hybrid(snp_list, p_bc, "bonf_G_Adj", args.output_file_path)

                    cp.cuda.Device().synchronize()
                    t_math_end = time.time()
                    print(f"[TIME] GPU Math Engine took: {t_math_end - t_math_start:.2f} seconds")
                    continue
                else:
                    combined_p_gpu = combine_function(gpu_data_matrix, args.data_type)

                cp.cuda.Device().synchronize()
                t_math_end = time.time()
                print(f"[TIME] GPU Math Engine took: {t_math_end - t_math_start:.2f} seconds")

                t_write_start = time.time()
                save_results_hybrid(snp_list, combined_p_gpu, method, args.output_file_path)
                t_write_end = time.time()
                print(f"[TIME] Polars SSD Writer took: {t_write_end - t_write_start:.2f} seconds\n")

        monitor.keep_measuring = False
        mem_thread.join()
        run_end_time = time.time()

        total_run_time = run_end_time - run_start_time
        peak_ram = monitor.peak_memory

        time_records.append(total_run_time)
        ram_records.append(peak_ram)

        print(f"Run {run + 1} PEAK RAM USAGE: {peak_ram:.2f} MB")
        print(f"Run {run + 1} TOTAL TIME: {total_run_time:.2f} seconds\n")

        cp.get_default_memory_pool().free_all_blocks()

    if args.runs > 1:
        print(f"{'=' * 50}")
        print(f"FINAL BENCHMARK RESULTS (Over {args.runs} runs)")
        print(f"{'=' * 50}")
        print(f"Mean Execution Time: {np.mean(time_records):.2f} seconds (± {np.std(time_records):.2f}s)")
        print(f"Mean Peak RAM Usage: {np.mean(ram_records):.2f} MB (± {np.std(ram_records):.2f} MB)")
        print(f"{'=' * 50}\n")
