import os
os.environ["POLARS_MAX_THREADS"] = "10"
import polars as pl
import numpy as np
from scipy.special import ndtr
import concurrent.futures
import time
import argparse
import gc
from numba import jit


# --- 1. THE NUMBA GRIM REAPER (High-Speed Logic) ---
@jit(nopython=True)
def fast_grim_reaper(s1_array, s2_array):
    losers = {np.int64(-1)}
    for i in range(len(s1_array)):
        s1 = s1_array[i]
        s2 = s2_array[i]

        if s1 in losers or s2 in losers:
            continue

        # Always arbitrarily drop the second SNP
        losers.add(s2)

    losers.remove(np.int64(-1))
    out_arr = np.empty(len(losers), dtype=np.int64)

    idx = 0
    for val in losers:
        out_arr[idx] = val
        idx += 1

    return out_arr  # Returns a pure NumPy array


# --- 2. THE WORKER PROCESS (Parallel Execution) ---
def TOP_LD_info(chrom_target_list, chrom, R2_threshold, base_dir):
    universal_ld_path = os.path.join(base_dir, f"UNIVERSAL_LD_chr{chrom}.parquet")

    if not os.path.exists(universal_ld_path) or len(chrom_target_list) == 0:
        return np.array([], dtype=np.int64)

    scaled_threshold = int(round(R2_threshold * 100))
    try:
        # We only need the SNP column now
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
            return losers_int

    except Exception as e:
        print(f" [SKIP] Chrom {chrom}: {e}")

    return np.array([], dtype=np.int64)


# --- 3. THE MAIN COORDINATOR ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input TSV GWAS file')
    parser.add_argument('-o', '--output', required=True, help='Output TSV file')
    parser.add_argument('--ld_dir', required=True, help='Directory containing UNIVERSAL_LD files')
    parser.add_argument('--r2', type=float, default=0.2, help='R2 threshold')
    parser.add_argument('--Z_prune', action='store_true', help='Flag to pre-filter highly significant SNPs before R2 pruning')
    parser.add_argument('--Z_threshold', type=float, default=2.0, help='Drops all SNPs with >= |Z-score| before R2 pruning')
    args = parser.parse_args()

    # Safety check: If they say --Z_prune, they MUST provide a threshold
    if args.Z_prune and args.Z_threshold is None:
        parser.error("--Z_prune requires --Z_threshold to be set.")

    start_time = time.time()

    print(f"Loading {args.input}...")
    df = pl.read_csv(args.input, separator='\t')
    starting = df.height
    print(f'Loaded {starting} SNPs')
    upper_cols = [col.upper() for col in df.columns]
    p_candidates = {"P", "PVAL", "PVALUE", "P_VALUE"}
    p_col_name = None

    # Hunt for a P-value column FIRST, but only if Z isn't there
    if "Z" not in upper_cols:
        for i, col in enumerate(upper_cols):
            if col in p_candidates:
                p_col_name = df.columns[i]
                break

    # Execute the filtering chain (Notice we NEVER sort!)
    if "Z" in upper_cols:
        print(" [INFO] Z-score column found. Working with Z-scores.")
        if args.Z_prune:
            df = df.filter(pl.col("Z").abs() < abs(args.Z_threshold))
            print(f" [INFO] Dropped SNPs with Z >= {abs(args.Z_threshold)}. Removed: {starting - df.height} SNPs.")

        df_valid = (
            df.select(["SNP", "CHR"])  # We don't even need Z anymore!
            .with_columns(
                pl.col("SNP").str.replace("rs", "").cast(pl.Int64, strict=False).alias("Target_SNP")
            )
            .drop_nulls(subset=["Target_SNP", "CHR"])
        )

    elif p_col_name:
        print(f" [INFO] Found P-value column: '{p_col_name}'. Working with P-values.")
        if args.Z_prune:
            p_thresh = abs(max(2.0 * (1.0 - ndtr(abs(args.Z_threshold))), 1e-300))
            df = df.filter(pl.col(p_col_name) > p_thresh)
            print(f" [INFO] Dropped SNPs with P-value <= {p_thresh:.2e}. Removed: {starting - df.height} SNPs.")

        df_valid = (
            df.select(["SNP", "CHR"])  # We don't even need P anymore!
            .with_columns(
                pl.col("SNP").str.replace("rs", "").cast(pl.Int64, strict=False).alias("Target_SNP")
            )
            .drop_nulls(subset=["Target_SNP", "CHR"])
        )

    else:
        print(" [INFO] No P-value column OR Z-score column is present. At least one of them is required.")
        os._exit(1)

    total_snps = df_valid.height
    print(f"Total SNPs entering R2 pruning phase: {total_snps}")

    chrom_dict = {}

    # Shred the dataframe into a dictionary in ONE single pass (O(N) speed)
    partitions = df_valid.partition_by("CHR", as_dict=True)
    unique_chrs = [k[0] for k in partitions.keys()]

    # Loop through the pre-separated chunks and just save the SNP arrays
    for (c,), chrom_df in partitions.items():
        chrom_dict[str(c)] = chrom_df.get_column("Target_SNP").to_numpy()

    del df_valid
    gc.collect()

    print(f"Starting parallel pruning on {len(unique_chrs)} chromosomes...")
    losers_arrays = []
    safe_workers = 3
    # if os.cpu_count() < 10:
    #     safe_workers = 2
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers)
    futures = {}

    try:
        # Notice we pass an empty list [] as the default fallback now
        futures = {executor.submit(TOP_LD_info, chrom_dict.get(str(c), []), c, args.r2, args.ld_dir): c
                   for c in unique_chrs}

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if len(res) > 0:
                losers_arrays.append(res)

        executor.shutdown(wait=True)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected: Terminating processes...")
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        print("[!] Pool closed: Bypassing Windows cleanup locks...")
        os._exit(1)

    if losers_arrays:
        print("Finalizing exclusion list...")
        flat_losers = np.concatenate(losers_arrays)
        loser_strings = pl.lit("rs") + pl.Series(flat_losers).cast(pl.Utf8)

        before = df.height
        df = df.filter(~pl.col("SNP").is_in(loser_strings))
        print(f"Done! Excluded {before - df.height} SNPs.")
    else:
        print("No SNPs met the criteria for exclusion.")

    print(f"Saving to {args.output}...")
    df.write_csv(args.output, separator='\t')
    print(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()