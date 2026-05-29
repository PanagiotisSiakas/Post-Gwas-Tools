import numpy as np
import os
os.environ["POLARS_MAX_THREADS"] = "10"
import polars as pl
import argparse
import math
import sys
import gc
import psutil
import threading
from scipy.stats import gamma
from scipy.stats import chi2
import cProfile
import pstats
import time
from scipy.special import ndtr
import concurrent.futures
from numba import jit

def main():
    parser = argparse.ArgumentParser(description='Merge GWAS summary statistics files by SNPID_UKB')

    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help='Input GWAS files (at least 2). Example: gwas1.tsv.gz, gwas2.tsv.gz')

    parser.add_argument('-o', '--output', default='results.tsv.gz', help='Output file path. Example: results.tsv.gz')

    parser.add_argument('--method', type=str,
                        choices=['CAUCHY', 'MINP', 'HMP', 'MCM', 'CMC',
                                 'YANG', 'WALD_Z_SCORES', 'WALD_BETAS_SES'],
                        help='Method for combining p-values: CAUCHY, MINP, HMP, MCM, CMC, YANG, WALD_Z_SCORES, WALD_BETAS_SES')

    parser.add_argument('--R2_cut_off', type=float,
                        help='R2 threshold for LD-based pruning when estimating R correlation matrix. Default: no pruning.')

    parser.add_argument('--maf_threshold', type=float, default=0.01,
                        help='Minor Allele Frequency threshold for LD pruning. Default: 0.01')

    parser.add_argument('--population', type=str, default='EUR',
                        help='Reference population for LD pruning (EUR, AFR, EAS, SAS). Default is EUR.')

    parser.add_argument('--z_cut_off', type=float, default=None,
                        help='Z-score threshold for null SNP filtering when estimating R. Default: no filtering.')

    parser.add_argument('--study_fraction', type=float, default=None,
                        help='Minimum number of studies a SNP must appear in.')


    args = parser.parse_args()

    start_time = time.time()

    # ---RAM PROFILER ---
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

                    # Convert to Megabytes
                    current_mem_mb = total_mem / (1024 * 1024)

                    # Update peak
                    if current_mem_mb > self.peak_memory:
                        self.peak_memory = current_mem_mb

                except psutil.NoSuchProcess:
                    pass

                time.sleep(0.025)  # Check every 25 milliseconds

    monitor = MemoryMonitor()
    # Adding daemon=True tells Python to kill this thread instantly when the script ends
    mem_thread = threading.Thread(target=monitor.measure_memory, daemon=True)
    mem_thread.start()

    # Wrap the main execution in a profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Check if there are more than 2 GWAS input files
    if len(args.input) < 2:
        print("More than two input files are required.", file=sys.stderr)
        sys.exit(1)

    K = len(args.input)  # total number of GWAS studies
    if args.study_fraction is not None:
        min_study_count = max(1, math.ceil(args.study_fraction * K))
        print(f" Filtering SNPs: must appear in at least {min_study_count}/{K} studies.")
    else:
        min_study_count = 1

        t_read_start = time.time()
        print(f"[{time.strftime('%X')}] Scanning and Merging {K} files lazily with Polars...", flush=True)

        pl.enable_string_cache()

        required_cols = ['SNP', 'CHR', 'POS', 'A1', 'A2', 'BETA', 'SE', 'Z']

        pl_dtypes = {'CHR': pl.UInt8, 'POS': pl.Int32, 'A1': pl.Categorical, 'A2': pl.Categorical,
                     'BETA': pl.Float32, 'SE': pl.Float32, 'Z': pl.Float32}

        lf_merged = pl.scan_csv(args.input[0], separator='\t', schema_overrides=pl_dtypes).select(required_cols)
        rename_map = {'BETA': 'BETA1', 'Z': 'Z1', 'SE': 'SE1'}
        lf_merged = lf_merged.rename(rename_map)

        join_keys = ['SNP', 'CHR', 'POS', 'A1', 'A2']

        for i, file_path in enumerate(args.input[1:], start=2):
            lf_next = pl.scan_csv(file_path, separator='\t', schema_overrides=pl_dtypes).select(required_cols)
            rename_map = {'BETA': f'BETA{i}', 'Z': f'Z{i}', 'SE': f'SE{i}'}
            lf_next = lf_next.rename(rename_map)
            lf_merged = lf_merged.join(lf_next, on=join_keys, how="full", coalesce=True)

        beta_cols_pl = [f'BETA{i}' for i in range(1, K + 1)]

        lf_merged = lf_merged.with_columns(
            pl.sum_horizontal(pl.col(beta_cols_pl).is_not_null()).alias("N")
        ).filter(pl.col("N") >= min_study_count)

        print(f" [DEBUG] Polars execution graph constructed. Executing data stream into RAM...")

        df_merged = lf_merged.collect(engine="streaming")

        pl.disable_string_cache()

        # Sort after the collect to organize the final RAM block
        df_merged = df_merged.sort(["CHR", "POS"])

        z_cols = [f'Z{i}' for i in range(1, K + 1)]
        beta_cols = [f'BETA{i}' for i in range(1, K + 1)]
        se_cols = [f'SE{i}' for i in range(1, K + 1)]

        final_columns = ['SNP', 'CHR', 'POS', 'A1', 'A2'] + beta_cols + se_cols + z_cols + ['N']

        df_merged = df_merged.select(final_columns)

        shared_snps = df_merged.filter(pl.col("N") == 2).height
        file1_only = df_merged.filter(pl.col("N") == 1).height
        print(f" [DEBUG] Shared SNPs: {shared_snps}")
        print(f" [DEBUG] SNPs only in File 1: {file1_only}")

        df_for_corr = df_merged.select(["SNP", "CHR"] + z_cols)
        t_read_end = time.time()
        print(f"[TIME] Lazy Reading & Merging took: {t_read_end - t_read_start:.2f} seconds\n")

    # Z-BASED FILTERING (optional)
    # Avoids polygenic inflation, keeps only SNPs |Z| <= z_cut_off for the estimation of the R correlation matrix
    # Only if the user provides a value for the z-threshold will the z_cut_off be performed
    if args.z_cut_off is not None:
        print(f"Applying Z-based filtering (|Z| <= {args.z_cut_off}) for R estimation.")

        z_cols = [col for col in df_for_corr.columns if col.startswith('Z')]
        # if z_cols:
        #     Z_matrix = df_for_corr[z_cols].values.astype(np.float32)
        #     # A SNP is kept if all its observed z-scores satisfy |Z| <= threshold
        #     within_threshold = (np.abs(Z_matrix) <= args.z_cut_off) | (~np.isfinite(Z_matrix))
        #     keep_rows = np.all(within_threshold, axis=1)
        #     df_for_corr = df_for_corr[keep_rows].copy()
        if z_cols:
            Z_matrix = df_for_corr.select(z_cols).to_numpy().astype(np.float32)
            within_threshold = (np.abs(Z_matrix) <= args.z_cut_off) | (~np.isfinite(Z_matrix))
            keep_rows = np.all(within_threshold, axis=1)
            # Polars native boolean filtering
            df_for_corr = df_for_corr.filter(pl.Series(keep_rows))
        else:
            print(f"No Z-columns found. Skipping Z-filtering.")
    else:
        print(f"Skipping Z-based filtering.")

    # R2-BASED FILTERING (optional)
    # Removes SNPs in linkage disequilibrium (LD) to ensure independence among SNPs, before the calculation of the correlation matrix R
    # Only if the user provides a value for the R2-threshold will the R2_cut_off be performed
        #R2-BASED FILTERING (Optimized: Parallel)
        if args.R2_cut_off is not None:
            print(f"Applying R2-based filtering (Pruning R2 >= {args.R2_cut_off})...")
            pruning_start_time = time.time()  # <--- START THE DEDICATED TIMER

            if 'SNP' not in df_for_corr.columns:
                print("No SNP columns found. Skipping R2-based filtering.")
                sys.exit(1)

            # ---> POLARS DEBUG COUNTER <---
            total_snps = df_for_corr.height
            print(f" [DEBUG] Total SNPs in merged dataset before pruning: {total_snps}")

            snps_to_exclude = []

            LD_POPULATION = args.population
            CHROM_LIST = [str(c) for c in range(1, 23)]
            safe_cores = round(os.cpu_count() * 0.75)

            # Polars native check: completely bypasses Python memory allocation
            if total_snps == 0:
                print("No SNPs to check for LD.")

            else:
                # --- CHROMOSOME PARTITIONING (Pure Polars) ---

                # 1. Drop nulls, strip 'rs', and cast to pure Int64 natively
                df_valid = (
                    df_for_corr.select(["SNP", "CHR"])
                    .drop_nulls()
                    .with_columns(
                        pl.col("SNP").str.replace("rs", "").cast(pl.Int64, strict=False).alias("Target_SNP")
                    )
                    .drop_nulls(subset=["Target_SNP"])
                )

                # 2. Extract the unique chromosomes (usually 1 through 22)
                unique_chrs = df_valid.get_column("CHR").unique().to_list()

                # 3. Build the dictionary using pure C-contiguous NumPy arrays
                chrom_dict = {}
                for c in unique_chrs:
                    # Instantly filter for the chromosome and extract a raw numpy array
                    c_array = df_valid.filter(pl.col("CHR") == c).get_column("Target_SNP").unique().to_numpy()
                    chrom_dict[str(c)] = c_array
                # --- OPTIMIZATION: PARALLEL PROCESSING ---
                # 1. Manually start the executor WITHOUT the "with" statement
                # 2. GET RID OF THE NOW USELESS DATAFRAME ( eixa empneush ekeino to bradu )
                del df_valid
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
                future_to_chrom = {}

                try:
                    def get_ld_path(chrom):
                        base_dir = r"C:\Users\pansi\Desktop\PTUXIAKH\MetaCP\metacp-main\metacp-main"
                        return f"{base_dir}/UNIVERSAL_LD_chr{chrom}.parquet"

                    valid_chroms = [
                        c for c in CHROM_LIST
                        if os.path.exists(get_ld_path(c))
                    ]

                    if not valid_chroms:
                        print(" No LD files found. Skipping.")
                        executor.shutdown(wait=True)
                    else:
                        print(f" Found {len(valid_chroms)} valid LD files. Starting partitioned threads...")

                        # 2. Submit tasks
                        future_to_chrom = {
                            executor.submit(TOP_LD_info, chrom_dict.get(chrom, []), chrom,
                                            args.R2_cut_off): chrom
                            for chrom in valid_chroms
                        }

                        # 3. Collect results
                        for future in concurrent.futures.as_completed(future_to_chrom):
                            chrom = future_to_chrom[future]
                            try:
                                loser_arr = future.result()
                                if len(loser_arr) > 0:
                                    snps_to_exclude.append(loser_arr)
                                    #print(f"  [DONE] Thread for Chr {chrom} finished. Pruned {len(rsids)} SNPs. TIME:{time.time() - start_time:.2f}")
                            except Exception as exc:
                                print(f"  - Chr {chrom} generated an exception: {exc}")

                        # 4. Normal shutdown if everything finishes safely
                        executor.shutdown(wait=True)

                except KeyboardInterrupt:
                    print("\n[!] Ctrl+C detected: Terminating processes...")

                    # Cancel pending tasks
                    for future in future_to_chrom:
                        future.cancel()

                    # Force pool closure without waiting for active Rust threads
                    executor.shutdown(wait=False, cancel_futures=True)
                    print("[!] Pool closed: Bypassing Windows cleanup locks...")

                    # os._exit is a brutal OS-level kill.
                    # It completely ignores Python's polite cleanup protocols and instantly terminates the terminal lock.
                    os._exit(1)

                gc.collect()

            # Apply Exclusion (Optimized)
                # When you call .astype(str), Pandas literally creates a brand new,
                # temporary 10.3-million row array of text in your RAM, and then compares every single row against
                # an 8.4-million item string dictionary. It is choking on text processing.
                # THE FIX (Instant Exclusion):
                # Because your SNP column is already text, we drop the .astype conversion entirely.
                # Furthermore, converting snps_to_exclude to a frozenset optimizes Pandas' internal C-hashing.
                # --- THE PANDAS-FREE EXCLUSION ---
                before = df_for_corr.height

                if snps_to_exclude:
                    # 1. Smash all 22 integer arrays into one solid C-array
                    flat_losers = np.concatenate(snps_to_exclude)

                    # 2. Let Polars construct the strings natively without Python lists
                    loser_series = pl.Series("losers", flat_losers)
                    loser_strings = pl.lit("rs") + loser_series.cast(pl.Utf8)

                    # 3. Apply the filter ONLY to the correlation dataset
                    df_for_corr = df_for_corr.filter(~pl.col("SNP").is_in(loser_strings))

                after = df_for_corr.height
                pruning_duration = time.time() - pruning_start_time  # <--- STOP THE TIMER
            print(f"Excluded {before - after} SNPs in LD (R2 >= {args.R2_cut_off}) {pruning_duration:.2f} seconds.")
        else:
            print("Skipping R2-based LD filtering.")


    method_names = {
        'CAUCHY': 'CAUCHY',
        'MINP': 'MINP',
        'HMP': 'HMP',
        'MCM': 'MCM',
        'CMC': 'CMC',
        'YANG': 'YANG',
        'WALD_Z_SCORES': 'WALD_Z_SCORES',
        'WALD_BETAS_SES': 'WALD_BETAS_SES'
    }

    # Check if the user provides a method
    if args.method is None:
        print("No method specified. Default method will be used: CAUCHY")
        choice = 'CAUCHY'
    else:
        # Apply the selected method from the user to the dataframe
        choice = args.method

    # Map each method name to their function
    method_functions = {
        'CAUCHY': cauchy_combine_gwas,
        'MINP': minp_gwas,
        'HMP': hmp_gwas,
        'MCM': mcm_gwas,
        'CMC': cmc_gwas,
        'YANG': lambda df_merged: yang_gwas(df_merged, df_for_corr=df_for_corr),
        'WALD_Z_SCORES': lambda df_merged: wald_test_gwas(df_merged, df_for_corr=df_for_corr),
        'WALD_BETAS_SES': lambda df_merged: wald_test_gwas_from_beta_se(df_merged, df_for_corr=df_for_corr)
    }

    method_func = method_functions[choice]

    t_math_start = time.time()  # <--- START TIMER 3
    print(f"[{time.strftime('%X')}] Executing {choice} Math Engine...", flush=True)

    df_result = method_func(df_merged)

    t_math_end = time.time()  # <--- END TIMER 3
    print(f"[TIME] Math Engine ({choice}) took: {t_math_end - t_math_start:.2f} seconds\n")

    # ---> STRICT DOWNCASTING FOR I/O OPTIMIZATION <---
    # Separate P-values (Keep Float64) from Test Statistics (Downcast to Float32)
    p_cols = [col for col in df_result.columns if col.endswith('_p')]
    stat_cols = [col for col in df_result.columns if 'Wald' in col and not col.endswith('_p')]

    # Build the exact casting instructions (Omit p_cols so they stay Float64)
    cast_exprs = [
                     pl.col("CHR").cast(pl.Int8),
                     pl.col("N").cast(pl.Int8)
                 ] + [pl.col(col).cast(pl.Float32) for col in stat_cols]

    # 3. Apply the cast natively in Polars
    df_result = df_result.with_columns(cast_exprs).drop(beta_cols + se_cols + z_cols)

    output_base = args.output
    method_suffix = method_names[choice]
    if output_base.endswith('.tsv.gz'):
        base_name = output_base[:-7]
    elif output_base.endswith('.gz'):
        base_name = output_base[:-3]
    else:
        base_name = os.path.splitext(output_base)[0]

    t_write_start = time.time()  # <--- START TIMER 4
    output_filename = f"{base_name}_{method_suffix}2.tsv.gz"
    print(f"[{time.strftime('%X')}] Saving to {output_filename} (Pure Polars Write)...", flush=True)

    # Write directly from Polars to disk (Zero Pandas memory conversions)
    df_result.write_csv(output_filename, separator='\t', float_precision=None, compression="gzip")

    print(f"Done saving data to {output_filename}...!")
    t_write_end = time.time()  # <--- END TIMER 4
    print(f"[TIME] Downcasting & Saving took: {t_write_end - t_write_start:.2f} seconds\n")
    profiler.disable()
    end_time = time.time()  # End timer

    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    # --- ADD THIS TO STOP AND PRINT RAM ---
    monitor.keep_measuring = False
    mem_thread.join()
    print(f"TRUE PEAK RAM USAGE: {monitor.peak_memory:.2f} MB")
    # --------------------------------------
    # Show top 20 functions by cumulative time
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    print("\nTop 20 functions by cumulative execution time:")
    stats.print_stats(20)
# --------------------------------------------------------------------------------------------------------
    run_ram = monitor.peak_memory
    run_time = end_time - start_time
    return run_ram, run_time

@jit(nopython=True)
def fast_grim_reaper(s1_array, s2_array):
    losers = {np.int64(-1)}
    for i in range(len(s1_array)):
        s1 = s1_array[i]
        s2 = s2_array[i]
        # Numba handles integer sets extremely fast
        if s1 in losers or s2 in losers:
            continue
        losers.add(s2)
    losers.remove(np.int64(-1))
    return losers

def TOP_LD_info(chrom_target_list, chrom, R2_threshold):
    """
    Partitioned Worker Process (Pure-Integer Optimized).
    """
    if pl is None: raise ImportError("Polars is required.")

    base_dir = r"C:\Users\pansi\Desktop\PTUXIAKH\MetaCP\metacp-main\metacp-main"
    universal_ld_path = f"{base_dir}/UNIVERSAL_LD_chr{chrom}.parquet"

    if not os.path.exists(universal_ld_path) or len(chrom_target_list) == 0:
        return set()

    scaled_threshold = int(round(R2_threshold * 100))

    try:
        target_lf = pl.DataFrame({"Target_SNP": chrom_target_list}).lazy()

        matched_df = (
            pl.scan_parquet(universal_ld_path)
            .filter(pl.col("R2") >= scaled_threshold)
            .join(target_lf, left_on="rsID1", right_on="Target_SNP", how="inner")
            .join(target_lf, left_on="rsID2", right_on="Target_SNP", how="inner")

            .select(["rsID1", "rsID2"])
            .collect(streaming=True)
        )

        if matched_df.height > 0:
            s1_arr = matched_df.get_column("rsID1").to_numpy()
            s2_arr = matched_df.get_column("rsID2").to_numpy()

            losers_int = fast_grim_reaper(s1_arr, s2_arr)

            return np.array(list(losers_int), dtype=np.int64)

        return set()

    except Exception as e:
        print(f" [SKIP] Chrom {chrom}: {e}")
        return set()

# Cauchy Combination Method
def cauchy_combine_gwas(df_merged):
    """
    Apply the Cauchy combination method (Pure Polars + NumPy Zero-Copy).
    """
    z_columns = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_columns:
        raise ValueError("No z-score columns found.")

    Z = df_merged.select(z_columns).to_numpy().astype(np.float32)
    n_snps = Z.shape[0]

    cauchy_p = np.full(n_snps, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(Z)
    finite_counts = finite_mask.sum(axis=1)

    valid_snps = finite_counts >= 2

    if not np.any(valid_snps):
        return df_merged.with_columns(pl.Series("Cauchy_p", cauchy_p))

    Z_valid = Z[valid_snps].astype(np.float64)
    finite_mask_valid = finite_mask[valid_snps]

    p_upper = ndtr(-Z_valid)
    p_lower = ndtr(Z_valid)

    epsilon = 1e-300
    p_upper = np.clip(p_upper, epsilon, 1.0 - epsilon)
    p_lower = np.clip(p_lower, epsilon, 1.0 - epsilon)

    T_upper = np.tan((0.5 - p_upper) * np.pi)
    T_lower = np.tan((0.5 - p_lower) * np.pi)

    T_upper[~finite_mask_valid] = 0.0
    T_lower[~finite_mask_valid] = 0.0

    counts_valid = finite_counts[valid_snps]
    t_upper = T_upper.sum(axis=1) / counts_valid
    t_lower = T_lower.sum(axis=1) / counts_valid

    combined_p_upper = 0.5 - (np.arctan(t_upper) / np.pi)
    combined_p_lower = 0.5 - (np.arctan(t_lower) / np.pi)

    combined_p_upper = np.clip(combined_p_upper, 0.0, 1.0)
    combined_p_lower = np.clip(combined_p_lower, 0.0, 1.0)

    cauchy_p_valid = 2.0 * np.minimum(combined_p_upper, combined_p_lower)
    cauchy_p_valid = np.clip(cauchy_p_valid, 0.0, 1.0)

    cauchy_p[valid_snps] = cauchy_p_valid

    return df_merged.with_columns(pl.Series("Cauchy_p", cauchy_p))


# Minimum P-value Method
def minp_gwas(df_merged):
    """
    Apply the Minimum P-value (minP) method to combine GWAS results.
    """
    z_cols = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_cols:
        raise ValueError("No z-score columns found.")

    Z = df_merged.select(z_cols).to_numpy().astype(np.float32)
    n_snps, k = Z.shape

    # Initialize a null output
    minp_p = np.full(n_snps, np.nan, dtype=np.float64)

    # Mask for valid z-scores
    finite_mask = np.isfinite(Z)
    k_valid = finite_mask.sum(axis=1)

    valid_snps = k_valid >= 1

    if not np.any(valid_snps):
        return df_merged.with_columns(pl.Series("MinP_p", minp_p))

    Z_valid = Z[valid_snps].astype(np.float64)
    finite_mask_valid = finite_mask[valid_snps]
    k_valid_vec = k_valid[valid_snps]

    p_upper = ndtr(-Z_valid)
    p_lower = ndtr(Z_valid)

    # Use Float64 tiny limit
    epsilon = 1e-300
    p_upper = np.clip(p_upper, epsilon, 1.0 - epsilon)
    p_lower = np.clip(p_lower, epsilon, 1.0 - epsilon)

    # Account for missing data, set missing entries to 1 so they won't affect the minimum
    p_upper[~finite_mask_valid] = 1.0
    p_lower[~finite_mask_valid] = 1.0

    # For each SNP, take the minimum p-value across studies
    min_p_upper = np.min(p_upper, axis=1)
    min_p_lower = np.min(p_lower, axis=1)

    # Calculate the combined p-value using minP formula
    combined_p_upper = 1.0 - np.power(1.0 - min_p_upper, k_valid_vec)
    combined_p_lower = 1.0 - np.power(1.0 - min_p_lower, k_valid_vec)

    # Two-sided final p-value
    minp_p_valid = 2 * np.minimum(combined_p_upper, combined_p_lower)
    minp_p_valid = np.clip(minp_p_valid, 0.0, 1.0)

    # Assign the calculated Minimum p-value to the valid SNPs
    minp_p[valid_snps] = minp_p_valid

    return df_merged.with_columns(pl.Series("MinP_p", minp_p))


# Harmonic Mean P-value Method
def hmp_gwas(df_merged):
    """
    Apply the Harmonic Mean P-value (HMP) method to combine GWAS z-scores.
    Goal: detects any meaningful association signal across multiple GWAS studies,
        without assuming a consistent direction or effect size. Is designed for
        heterogenous and sparse signals.
    """
    z_cols = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_cols:
        raise ValueError("No z-score columns found.")

    Z = df_merged.select(z_cols).to_numpy().astype(np.float32)
    n_snps, k = Z.shape

    hmp_p = np.full(n_snps, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(Z)
    k_valid = finite_mask.sum(axis=1)

    valid_snps = k_valid >= 2
    if not np.any(valid_snps):
        return df_merged.with_columns(pl.Series("HMP_p", hmp_p))

    # Create a subset data for only the valid SNPs
    Z_valid = Z[valid_snps].astype(np.float64)
    finite_mask_valid = finite_mask[valid_snps]
    k_valid_vec = k_valid[valid_snps]

    p_upper = ndtr(-Z_valid)
    p_lower = ndtr(Z_valid)

    # Allow precision all the way down to Float64's limit
    epsilon = 1e-300
    p_upper = np.clip(p_upper, epsilon, 1.0)
    p_lower = np.clip(p_lower, epsilon, 1.0)

    p_upper[~finite_mask_valid] = np.inf
    p_lower[~finite_mask_valid] = np.inf

    # Compute the Harmonic Mean P-value statistic for each direction:
    hmp_upper = k_valid_vec / np.sum(1.0 / p_upper, axis=1)
    hmp_lower = k_valid_vec / np.sum(1.0 / p_lower, axis=1)

    # Apply the heuristic mean p-value method for a two-sided combined p-value
    hmp_p_valid = 2 * np.minimum(hmp_upper, hmp_lower)
    hmp_p_valid = np.clip(hmp_p_valid, 0.0, 1.0)

    # Assign the calculated Harmonic p-value to the SNPs that have one
    hmp_p[valid_snps] = hmp_p_valid

    return df_merged.with_columns(pl.Series("HMP_p", hmp_p))


# Minimum Combination Method
def mcm_gwas(df_merged):
    """
    Apply the Minimum Combination Method (MCM) to combine GWAS z-scores.
    Goal: produces a single, robust p-value that is powerful under
        either consistent (using Cauchy) or either heterogenous effects (using minP)
        and automatically chooses the stronger signal for each SNP
    """
    # Compute Cauchy combined p-values for all SNPs
    df_cauchy = cauchy_combine_gwas(df_merged)

    # Compute minP p-values for all SNPs
    df_minp = minp_gwas(df_merged)

    # Use get_column() to extract the Series, then convert to a zero-copy NumPy array
    cauchy_p = df_cauchy.get_column('Cauchy_p').to_numpy()
    minp_p = df_minp.get_column('MinP_p').to_numpy()

    # Combine them using heuristic MCM method
    mcm_p = 2.0 * np.minimum(cauchy_p, minp_p)

    # Ensure that all p-values are in the valid range (0,1) and force float32
    mcm_p = np.clip(mcm_p, 0.0, 1.0)

    return df_merged.with_columns(pl.Series("MCM_p", mcm_p))


# Combined Method of Combination
def cmc_gwas(df_merged):
    """
    Apply the CMC (Combined Method of Combination) to combine GWAS z-scores.
    Goal: creates a robust, adaptive meta p-value by combining the results of
        two different combination methods (Cauchy and MinP) using the Cauchy framework
        and produces a strong combined signal, if both methods agree.
    """
    # Compute Cauchy combined p-values for all SNPs
    df_cauchy = cauchy_combine_gwas(df_merged)

    # Compute minP p-values for all SNPs
    df_minp = minp_gwas(df_merged)

    cauchy_p = df_cauchy.get_column('Cauchy_p').to_numpy()
    minp_p = df_minp.get_column('MinP_p').to_numpy()

    # Combine the two p-values using the Cauchy method
    combined_p_values = np.stack([cauchy_p, minp_p], axis=1)

    # Ensure all p-values are in range (0,1) safely for float64
    epsilon = 1e-300
    combined_p_values = np.clip(combined_p_values, epsilon, 1.0 - epsilon)

    n_snps = combined_p_values.shape[0]  # number of SNPs

    # Initialize a null output (Forced to float64)
    cmc_p = np.full(n_snps, np.nan, dtype=np.float64)

    # Keep the valid combined p-values
    finite_mask = np.isfinite(combined_p_values)
    k_valid = finite_mask.sum(axis=1)

    # Keep only SNPs with at least 1 valid z-score
    valid_snps = k_valid >= 1
    if not np.any(valid_snps):
        return df_merged.with_columns(pl.Series("CMC_p", cmc_p))

    # Create a subset data for the valid SNPs only
    Z_valid = combined_p_values[valid_snps].astype(np.float64)
    finite_mask_valid = finite_mask[valid_snps]
    k_valid_vec = k_valid[valid_snps]

    # Convert combined p-values to T-values for Cauchy
    T = np.tan(np.pi * (0.5 - Z_valid))

    T[~finite_mask_valid] = 0.0

    # Compute mean T over valid studies only
    t_stat = T.sum(axis=1) / k_valid_vec

    # Compute combined p-value
    cmc_p_valid = 0.5 - (np.arctan(t_stat) / np.pi)
    cmc_p_valid = np.clip(cmc_p_valid, 0.0, 1.0)

    # Assign the calculated Minimum p-value to the SNPs that have one
    cmc_p[valid_snps] = cmc_p_valid

    return df_merged.with_columns(pl.Series("CMC_p", cmc_p))


# Estimate correlation matrix R using the z-scores from the given dataframe
def estimate_correlation(df_merged, snp_col='SNP', method='global'):
    """
    Estimation of the correlation between GWAS studies using the SNP's z-scores.
    Optimized with Zero-Copy NumPy and strict positive-definite safeguards.
    """
    if method != 'global':
        raise ValueError("Only global method is supported.")

    # Identify z-score columns
    z_cols = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_cols:
        raise ValueError("No z-score columns found.")

    Z_matrix = df_merged.select(z_cols).to_numpy().astype(np.float32)

    finite_mask = np.isfinite(Z_matrix)
    valid_rows = np.all(finite_mask, axis=1)
    Z_valid = Z_matrix[valid_rows].astype(np.float64)

    k = len(z_cols)
    if len(Z_valid) < 2:
        return np.eye(k, dtype=np.float32)

    R = np.corrcoef(Z_valid, rowvar=False)

    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)

    print(f"\nCorrelation matrix R:\n {R}")
    return R.astype(np.float64)


@jit(nopython=True)
def yang_delta_kernel(R_matrix, N_corr_samples, k):
    """
    LLVM-compiled C-speed kernel for generating the Yang delta covariance matrix.
    """
    delta_matrix = np.zeros((k, k), dtype=np.float64)
    c1 = 3.9081

    for i in range(k):
        for j in range(i + 1, k):
            r = R_matrix[i, j]

            # Bias correction
            biased_corrected_r = r * (1.0 + ((1.0 - r ** 2) / (2.0 * (N_corr_samples - 3.0))))

            # Pre-calculate square to save CPU cycles
            r2 = biased_corrected_r ** 2

            # Polynomial (Identical to expanding powers up to 10)
            f_r = (3.9081 * r2 +
                   0.0313 * r2 ** 2 +
                   0.1022 * r2 ** 3 -
                   0.1378 * r2 ** 4 +
                   0.0941 * r2 ** 5)

            # Bias calculation
            bias = (c1 / k) * (1.0 - r2) ** 2

            # Apply to both sides of the symmetric matrix
            delta = f_r - bias
            delta_matrix[i, j] = delta
            delta_matrix[j, i] = delta

    return delta_matrix


# Yang Method
def yang_gwas(df_merged, snp_col='SNP', df_for_corr=None):
    """
    Combine GWAS z-scores across studies using Yang's improved Brown's method for dependent p-values.
    Goal: combines p-values from multiple hypothesis tests that are statistically dependent
        and accounts for genetically correlated traits, shared controls and sample overlap
    """
    z_cols = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_cols:
        raise ValueError("No z-scores found.")

    Z = df_merged.select(z_cols).to_numpy().astype(np.float32)
    n_snps, k = Z.shape

    yang_p = np.full(n_snps, np.nan, dtype=np.float64)

    # Find SNPs with at least 1 valid z-score
    finite_mask = np.isfinite(Z)
    valid_z = finite_mask.any(axis=1)
    if not np.any(valid_z):
        return df_merged.with_columns(pl.Series("Yang_p", yang_p))

    # Estimate global correlation matrix R
    R = estimate_correlation(df_for_corr, snp_col='SNP', method='global')

    N_corr_samples = len(df_for_corr) if df_for_corr is not None else n_snps

    delta_matrix = yang_delta_kernel(R, float(N_corr_samples), k)
    delta_sum = np.sum(delta_matrix)

    mean_T = 2.0 * k
    var_T = 4.0 * k + delta_sum
    if var_T <= 0:
        var_T = 4.0 * k

    # gamma distribution that responds to our correlated data
    v_gamma = (mean_T ** 2) / var_T
    gamma_value = var_T / mean_T

    # Convert to two-sided p-values with strict Float64 precision
    p_matrix = np.full(Z.shape, 1.0, dtype=np.float64)

    # Perfect precision: Area = 2 * (Area to the left of the negative absolute value)
    p_matrix[finite_mask] = 2.0 * ndtr(-np.abs(Z[finite_mask].astype(np.float64)))

    # Float64 safe clipping
    epsilon = np.finfo(np.float64).tiny
    p_matrix = np.clip(p_matrix, epsilon, 1.0)

    # Computation of T statistic
    log_p = np.log(p_matrix)
    T_vals = -2.0 * np.sum(log_p, axis=1)

    # Computation of final p-value
    yang_p_vals = gamma.sf(T_vals, a=v_gamma, scale=gamma_value)
    yang_p_vals = np.clip(yang_p_vals, 0.0, 1.0)

    # Assign the calculated Yang p-value to the SNPs that have one
    yang_p[valid_z] = yang_p_vals

    return df_merged.with_columns(pl.Series("Yang_p", yang_p))

def estimate_correlation_beta_ses(df_merged, snp_col='SNP', method='global'):
    """
    Estimation of the correlation between GWAS studies using Z-scores.
    Optimized: Batch Polars extraction and single-pass matrix division.
    """
    if method != 'global':
        raise ValueError("Only method='global' is supported.")

    beta_cols = sorted([col for col in df_merged.columns if col.startswith("BETA")])
    se_cols = sorted([col for col in df_merged.columns if col.startswith("SE")])

    if beta_cols and se_cols:
        beta_matrix = df_merged.select(beta_cols).to_numpy().astype(np.float32)
        se_matrix = df_merged.select(se_cols).to_numpy().astype(np.float32)

        # Masking SE > 0 to avoid DivisionByZero warnings
        z_matrix = np.full_like(beta_matrix, np.nan)
        valid_mask = (np.isfinite(beta_matrix) & np.isfinite(se_matrix) & (se_matrix > 0))
        z_matrix[valid_mask] = beta_matrix[valid_mask] / se_matrix[valid_mask]

    else:
        z_cols = sorted([col for col in df_merged.columns if col.startswith('Z')])
        if not z_cols:
            raise ValueError("No BETA/SE or Z columns found.")
        z_matrix = df_merged.select(z_cols).to_numpy().astype(np.float64)

    k = z_matrix.shape[1]

    valid_snps = np.all(np.isfinite(z_matrix), axis=1)
    z_complete = z_matrix[valid_snps]

    if len(z_complete) < 2:
        return np.eye(k, dtype=np.float32)

    R = np.corrcoef(z_complete, rowvar=False)

    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)

    return R.astype(np.float32)

def wald_worker(chunk_data):
    """
    Worker function for ProcessPoolExecutor.
    Uses Binary Bitmask Vectorization (Unix rwx style) to group SNPs instantly in C.
    """
    z_chunk, finite_mask_chunk, R = chunk_data
    n_chunk, k = z_chunk.shape

    w_out = np.full(n_chunk, np.nan, dtype=np.float32)
    p_out = np.full(n_chunk, np.nan, dtype=np.float64)

    valid_z = finite_mask_chunk.any(axis=1)
    if not np.any(valid_z):
        return w_out, p_out

    powers = 1 << np.arange(k)

    # Dot product converts each boolean row into a single unique integer pattern ID instantly
    pattern_ids = finite_mask_chunk.dot(powers)

    # Find all the unique patterns that actually exist in this chunk
    unique_patterns = np.unique(pattern_ids)

    # Process each group once based on its unique pattern integer
    for p_val in unique_patterns:
        if p_val == 0:
            continue

        # Get all SNP indices that share this exact missingness pattern
        indices = np.nonzero(pattern_ids == p_val)[0]

        # Determine the tuple of observed studies from the first SNP in this group
        obs_tuple = tuple(np.nonzero(finite_mask_chunk[indices[0]])[0])
        n_obs = len(obs_tuple)

        Z_group = z_chunk[np.ix_(indices, obs_tuple)].astype(np.float64)

        if n_obs == 1:
            w = Z_group[:, 0] ** 2
            p_val_array = chi2.sf(w, df=1)
        else:
            R_sub = R[np.ix_(obs_tuple, obs_tuple)]
            try:
                R_inv = np.linalg.inv(R_sub)
            except np.linalg.LinAlgError:
                R_inv = np.linalg.pinv(R_sub)

            Z_Rinv = Z_group @ R_inv
            w = np.einsum('ij,ij->i', Z_Rinv, Z_group)
            p_val_array = chi2.sf(w, df=n_obs)

        w_out[indices] = w.astype(np.float32)
        p_out[indices] = p_val_array

    return w_out, p_out


# Wald test using z-scores
def wald_test_gwas(df_merged, snp_col='SNP', df_for_corr=None):
    """
    Performs a Wald-test statistic meta-analysis using only z-scores.
    Parallelized across CPU cores using ProcessPoolExecutor.
    """
    import concurrent.futures
    import multiprocessing

    # Identify z-score columns
    z_cols = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_cols:
        raise ValueError("No z-scores found.")

    Z = df_merged.select(z_cols).to_numpy().astype(np.float32)
    n_snps, k = Z.shape

    # Estimate global correlation matrix R
    R = estimate_correlation(df_for_corr, snp_col=snp_col, method='global')
    R = R.astype(np.float32)

    finite_mask = np.isfinite(Z)
    valid_z = finite_mask.any(axis=1)

    if not np.any(valid_z):
        wald_stats = np.full(n_snps, np.nan, dtype=np.float32)
        wald_p = np.full(n_snps, np.nan, dtype=np.float64)
        return df_merged.with_columns([
            pl.Series("Wald_statistic", wald_stats),
            pl.Series("Wald_p", wald_p)
        ])

    num_cores = max(3, multiprocessing.cpu_count() - 8)
    print(f"\n[INFO] Launching ProcessPoolExecutor with {num_cores} workers...")

    # Split the arrays into equal chunks for each core
    z_chunks = np.array_split(Z, num_cores)
    mask_chunks = np.array_split(finite_mask, num_cores)

    # Package the chunks with the R matrix to send to the workers
    tasks = [(z_chunks[i], mask_chunks[i], R) for i in range(num_cores)]

    all_w_out = []
    all_p_out = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(wald_worker, tasks)

        for w_chunk, p_chunk in results:
            all_w_out.append(w_chunk)
            all_p_out.append(p_chunk)

    wald_stats = np.concatenate(all_w_out)
    wald_p = np.concatenate(all_p_out)

    return df_merged.with_columns([
        pl.Series("Wald_Statistic", wald_stats),
        pl.Series("Wald_p", wald_p)
    ])

# Wald test using betas and standard errors
def wald_test_gwas_from_beta_se(df_merged, snp_col='SNP', df_for_corr=None):
    """
    Performs a Wald meta-analysis using beta and SE columns.
    Optimized: Polars extraction, vectorized Z-calculation, and Multiprocessing.
    """
    import concurrent.futures
    import multiprocessing

    beta_cols = [col for col in df_merged.columns if col.startswith('BETA')]
    se_cols = [col for col in df_merged.columns if col.startswith('SE')]

    if not beta_cols or not se_cols:
        raise ValueError("Both BETA and SE columns are required.")

    # Ensure they are sorted so BETA1 matches SE1, BETA2 matches SE2, etc.
    beta_cols.sort()
    se_cols.sort()

    beta_matrix = df_merged.select(beta_cols).to_numpy().astype(np.float32)
    se_matrix = df_merged.select(se_cols).to_numpy().astype(np.float32)

    R = estimate_correlation_beta_ses(df_for_corr, snp_col=snp_col, method='global')
    R = R.astype(np.float32)

    # We use np.divide to handle division and then mask out invalid/zero SEs
    z_matrix = np.full_like(beta_matrix, np.nan, dtype=np.float32)

    # A value is valid only if BETA and SE are finite AND SE > 0
    valid_mask = (np.isfinite(beta_matrix) & np.isfinite(se_matrix) & (se_matrix > 0))

    # Apply calculation only where valid to avoid 0/0 or inf
    z_matrix[valid_mask] = beta_matrix[valid_mask] / se_matrix[valid_mask]
    n_snps, k = z_matrix.shape

    valid_z = valid_mask.any(axis=1)
    if not np.any(valid_z):
        return df_merged.with_columns([
            pl.Series("Wald_Statistic", np.full(n_snps, np.nan, dtype=np.float32)),
            pl.Series("Wald_p", np.full(n_snps, np.nan, dtype=np.float64))
        ])

    # Get the number of CPU cores available, leave some free for the OS
    num_cores = max(3, multiprocessing.cpu_count() - 8)
    print(f"\n[INFO] Launching ProcessPoolExecutor (Beta/SE) with {num_cores} workers...")

    # Split the arrays into equal chunks for each core
    z_chunks = np.array_split(z_matrix, num_cores)
    mask_chunks = np.array_split(valid_mask, num_cores)

    # Package the tasks for the Bitmask worker
    tasks = [(z_chunks[i], mask_chunks[i], R) for i in range(num_cores)]

    all_w_out = []
    all_p_out = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(wald_worker, tasks)

        for w_chunk, p_chunk in results:
            all_w_out.append(w_chunk)
            all_p_out.append(p_chunk)

    wald_stats = np.concatenate(all_w_out)
    wald_p = np.concatenate(all_p_out)

    return df_merged.with_columns([
        pl.Series("Wald_statistic", wald_stats),
        pl.Series("Wald_p", wald_p)
    ])

if __name__ == '__main__':
    main()
