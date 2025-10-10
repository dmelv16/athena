#!/usr/bin/env python3
"""
Filters parquet to keep only 5 unique combinations of each grouping.
Memory efficient for large files.
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
OUTPUT_FILE = "filtered_5_each.parquet"
GROUPING_COLUMNS = ["ofp", "test_case"]  # Your grouping columns
MAX_GROUPS_PER_COMBINATION = 5  # How many unique run_ids to keep per group
CHUNK_SIZE = 10000  # Process this many rows at a time
# =====================================

def filter_parquet():
    print(f"📂 Reading: {INPUT_FILE}")
    print(f"🔑 Grouping by: {GROUPING_COLUMNS}")
    print(f"📊 Keeping {MAX_GROUPS_PER_COMBINATION} unique run_ids per group\n")
    
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
    
    # Track which run_ids we're keeping per grouping
    # Structure: {(ofp_val, test_case_val): set(run_ids)}
    kept_run_ids_per_group = defaultdict(set)
    
    # First pass: identify which run_ids to keep
    print("🔍 First pass: identifying run_ids to keep...")
    with tqdm(total=total_rows, desc="Scanning", unit="rows") as pbar:
        for i in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(i)
            
            # Process in chunks
            for batch_start in range(0, row_group.num_rows, CHUNK_SIZE):
                batch_end = min(batch_start + CHUNK_SIZE, row_group.num_rows)
                batch = row_group.slice(batch_start, batch_end - batch_start)
                
                # Convert to dict for easy access
                batch_dict = batch.to_pydict()
                num_rows = len(batch_dict['run_id'])
                
                for row_idx in range(num_rows):
                    # Get grouping values
                    group_key = tuple(batch_dict[col][row_idx] for col in GROUPING_COLUMNS)
                    run_id = batch_dict['run_id'][row_idx]
                    
                    # Add run_id if we haven't hit the limit for this group
                    if len(kept_run_ids_per_group[group_key]) < MAX_GROUPS_PER_COMBINATION:
                        kept_run_ids_per_group[group_key].add(run_id)
                
                pbar.update(num_rows)
                del batch_dict, batch
            
            del row_group
            gc.collect()
    
    # Flatten to a set of all run_ids we're keeping
    all_kept_run_ids = set()
    for run_ids in kept_run_ids_per_group.values():
        all_kept_run_ids.update(run_ids)
    
    print(f"\n📊 Summary:")
    print(f"   Unique grouping combinations: {len(kept_run_ids_per_group):,}")
    print(f"   Total unique run_ids to keep: {len(all_kept_run_ids):,}")
    
    # Second pass: filter and write
    print("\n✍️ Second pass: filtering and writing...")
    writer = None
    output_rows = 0
    
    try:
        with tqdm(total=total_rows, desc="Filtering", unit="rows") as pbar:
            for i in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(i)
                
                # Process in chunks
                for batch_start in range(0, row_group.num_rows, CHUNK_SIZE):
                    batch_end = min(batch_start + CHUNK_SIZE, row_group.num_rows)
                    batch = row_group.slice(batch_start, batch_end - batch_start)
                    
                    # Filter using PyArrow compute
                    run_id_array = batch.column('run_id')
                    
                    # Create mask for rows to keep
                    mask = pc.is_in(run_id_array, value_set=pa.array(list(all_kept_run_ids)))
                    
                    # Filter the batch
                    filtered_batch = pc.filter(batch, mask)
                    
                    if filtered_batch.num_rows > 0:
                        # Initialize writer on first non-empty batch
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
    
    # Print final summary
    print(f"\n✅ Done!")
    print(f"📉 Rows: {total_rows:,} → {output_rows:,} ({output_rows/total_rows*100:.1f}% kept)")
    print(f"💾 Output saved to: {OUTPUT_FILE}")
    
    # Show distribution
    print(f"\n📊 Distribution of run_ids per group:")
    distribution = defaultdict(int)
    for run_ids in kept_run_ids_per_group.values():
        distribution[len(run_ids)] += 1
    
    for count, num_groups in sorted(distribution.items()):
        print(f"   {num_groups:,} groups have {count} unique run_id(s)")

if __name__ == "__main__":
    try:
        filter_parquet()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
