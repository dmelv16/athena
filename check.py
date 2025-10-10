#!/usr/bin/env python3
"""
Filters parquet to exactly 26 unique run_ids total, 
ensuring 4-5 runs per grouping combination.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import gc
from collections import defaultdict
from tqdm import tqdm

# ============ EDIT THESE ============
INPUT_FILE = "output_with_runid.parquet"  # Your parquet with run_ids
OUTPUT_FILE = "filtered_26_runs.parquet"
GROUPING_COLUMNS = ["ofp", "test_case"]  # Your grouping columns
TOTAL_RUN_IDS = 26  # Total number of unique run_ids to keep
RUNS_PER_GROUP = 5  # Target runs per grouping (will use 4 if needed to fit more groups)
CHUNK_SIZE = 10000  # Process this many rows at a time
# =====================================

def filter_parquet():
    print(f"📂 Reading: {INPUT_FILE}")
    print(f"🔑 Grouping by: {GROUPING_COLUMNS}")
    print(f"🎯 Target: {TOTAL_RUN_IDS} total unique run_ids")
    print(f"📊 Aiming for {RUNS_PER_GROUP} runs per group\n")
    
    # Open input file
    parquet_file = pq.ParquetFile(INPUT_FILE, memory_map=True)
    total_rows = parquet_file.metadata.num_rows
    schema = parquet_file.schema_arrow
    
    print(f"Total input rows: {total_rows:,}")
    
    # Check columns exist
    missing = [col for col in GROUPING_COLUMNS if col not in schema.names]
    if missing:
        print(f"❌ ERROR: These columns don't exist: {missing}")
        print(f"Available columns: {schema.names}")
        return
    
    if 'run_id' not in schema.names:
        print("❌ ERROR: No 'run_id' column found. Run the first script first!")
        return
    
    # First pass: collect all run_ids per grouping
    print("🔍 Collecting all run_ids per grouping...")
    all_run_ids_per_group = defaultdict(set)
    
    with tqdm(total=total_rows, desc="Scanning", unit="rows") as pbar:
        for i in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(i)
            
            for batch_start in range(0, row_group.num_rows, CHUNK_SIZE):
                batch_end = min(batch_start + CHUNK_SIZE, row_group.num_rows)
                batch = row_group.slice(batch_start, batch_end - batch_start)
                
                batch_dict = batch.to_pydict()
                num_rows = len(batch_dict['run_id'])
                
                for row_idx in range(num_rows):
                    group_key = tuple(batch_dict[col][row_idx] for col in GROUPING_COLUMNS)
                    run_id = batch_dict['run_id'][row_idx]
                    all_run_ids_per_group[group_key].add(run_id)
                
                pbar.update(num_rows)
                del batch_dict, batch
            
            del row_group
            gc.collect()
    
    print(f"\nFound {len(all_run_ids_per_group)} unique grouping combinations")
    
    # Now select which run_ids to keep
    kept_run_ids = set()
    kept_groups = []
    
    # Sort groups by number of available run_ids (descending) to prioritize groups with more data
    sorted_groups = sorted(all_run_ids_per_group.items(), key=lambda x: len(x[1]), reverse=True)
    
    # First try with 5 runs per group
    runs_per_group = RUNS_PER_GROUP
    max_groups = TOTAL_RUN_IDS // runs_per_group
    
    print(f"\n📈 Selecting run_ids...")
    print(f"   Can fit {max_groups} groups with {runs_per_group} runs each")
    
    for group_key, available_run_ids in sorted_groups:
        if len(kept_run_ids) + runs_per_group > TOTAL_RUN_IDS:
            # See if we can fit with 4 runs
            if len(kept_run_ids) + 4 <= TOTAL_RUN_IDS:
                selected = list(available_run_ids)[:4]
                kept_run_ids.update(selected)
                kept_groups.append((group_key, len(selected)))
            else:
                # Can't fit this group, we're full
                break
        else:
            # Take up to runs_per_group from this group
            selected = list(available_run_ids)[:runs_per_group]
            kept_run_ids.update(selected)
            kept_groups.append((group_key, len(selected)))
    
    print(f"\n📊 Selection summary:")
    print(f"   Groups included: {len(kept_groups)}")
    print(f"   Total unique run_ids: {len(kept_run_ids)}")
    
    # Show distribution
    distribution = defaultdict(int)
    for _, count in kept_groups:
        distribution[count] += 1
    
    print(f"\n   Distribution:")
    for runs, num_groups in sorted(distribution.items()):
        print(f"   - {num_groups} groups with {runs} runs")
    
    # Show which groups were kept
    print(f"\n   Kept groups:")
    for group_key, count in kept_groups[:10]:  # Show first 10
        group_str = ", ".join(f"{GROUPING_COLUMNS[i]}={group_key[i]}" for i in range(len(GROUPING_COLUMNS)))
        print(f"   - {group_str}: {count} runs")
    if len(kept_groups) > 10:
        print(f"   ... and {len(kept_groups) - 10} more groups")
    
    # Second pass: filter and write
    print("\n✍️ Filtering and writing...")
    writer = None
    output_rows = 0
    
    try:
        with tqdm(total=total_rows, desc="Filtering", unit="rows") as pbar:
            for i in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(i)
                
                for batch_start in range(0, row_group.num_rows, CHUNK_SIZE):
                    batch_end = min(batch_start + CHUNK_SIZE, row_group.num_rows)
                    batch = row_group.slice(batch_start, batch_end - batch_start)
                    
                    # Filter using PyArrow compute
                    run_id_array = batch.column('run_id')
                    mask = pc.is_in(run_id_array, value_set=pa.array(list(kept_run_ids)))
                    
                    filtered_batch = pc.filter(batch, mask)
                    
                    if filtered_batch.num_rows > 0:
                        if writer is None:
                            writer = pq.ParquetWriter(
                                OUTPUT_FILE,
                                schema,
                                compression='snappy',
                                use_dictionary=True
                            )
                        
                        writer.write_table(filtered_batch)
                        output_rows += filtered_batch.num_rows
                    
                    pbar.update(batch.num_rows)
                    del batch, filtered_batch, mask, run_id_array
                
                del row_group
                gc.collect()
    
    finally:
        if writer:
            writer.close()
    
    print(f"\n✅ Done!")
    print(f"📉 Rows: {total_rows:,} → {output_rows:,} ({output_rows/total_rows*100:.1f}% kept)")
    print(f"🎯 Unique run_ids in output: {len(kept_run_ids)}")
    print(f"💾 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        filter_parquet()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
