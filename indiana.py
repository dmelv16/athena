#!/usr/bin/env python3
"""
Memory-efficient Parquet processor that adds run_id based on grouping columns.
Just edit the variables at the top and run it.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import hashlib
import gc
import os
from tqdm import tqdm

# ============ EDIT THESE ============
INPUT_FILE = "input.parquet"
OUTPUT_FILE = "output_with_runid.parquet"
GROUPING_COLUMNS = ["column1", "column2", "column3"]  # Change these to your columns
CHUNK_SIZE = 5000  # Lower this if you run out of memory
# =====================================

def create_run_id(row_values):
    """Create a deterministic run_id from concatenated values."""
    combined = '|'.join(str(v) if v is not None else 'NULL' for v in row_values)
    return hashlib.md5(combined.encode()).hexdigest()[:16]

def process_parquet():
    print(f"📂 Reading: {INPUT_FILE}")
    print(f"🔑 Grouping by: {GROUPING_COLUMNS}")
    print(f"📦 Chunk size: {CHUNK_SIZE:,} rows\n")
    
    # Open the parquet file
    parquet_file = pq.ParquetFile(INPUT_FILE, memory_map=True)
    total_rows = parquet_file.metadata.num_rows
    num_row_groups = parquet_file.num_row_groups
    
    print(f"Total rows: {total_rows:,}")
    print(f"Row groups: {num_row_groups}\n")
    
    # Get original schema and check columns
    original_schema = parquet_file.schema_arrow
    missing = [col for col in GROUPING_COLUMNS if col not in original_schema.names]
    if missing:
        print(f"❌ ERROR: These columns don't exist: {missing}")
        print(f"Available columns: {original_schema.names}")
        return
    
    # Create the new schema with run_id column added
    new_fields = list(original_schema)
    new_fields.append(pa.field('run_id', pa.string()))
    output_schema = pa.schema(new_fields)
    
    # Cache for run_ids
    run_id_cache = {}
    writer = None
    processed_rows = 0
    
    try:
        with tqdm(total=total_rows, desc="Processing", unit="rows") as pbar:
            # Process each row group
            for i in range(num_row_groups):
                row_group = parquet_file.read_row_group(i)
                
                # Process in chunks if row group is large
                for batch_start in range(0, row_group.num_rows, CHUNK_SIZE):
                    batch_end = min(batch_start + CHUNK_SIZE, row_group.num_rows)
                    batch = row_group.slice(batch_start, batch_end - batch_start)
                    
                    # Keep as Arrow table instead of converting to pandas
                    # This maintains schema consistency
                    batch_dict = batch.to_pydict()
                    
                    # Generate run_ids
                    run_ids = []
                    num_rows = len(batch_dict[next(iter(batch_dict))])  # Get row count
                    
                    for row_idx in range(num_rows):
                        # Get values for this row's grouping columns
                        row_values = []
                        for col in GROUPING_COLUMNS:
                            val = batch_dict[col][row_idx]
                            row_values.append(val)
                        
                        key = tuple(row_values)
                        if key not in run_id_cache:
                            run_id_cache[key] = create_run_id(row_values)
                        run_ids.append(run_id_cache[key])
                    
                    # Add run_id to the dictionary
                    batch_dict['run_id'] = run_ids
                    
                    # Create table with explicit schema
                    table = pa.Table.from_pydict(batch_dict, schema=output_schema)
                    
                    # Initialize writer on first batch
                    if writer is None:
                        writer = pq.ParquetWriter(
                            OUTPUT_FILE,
                            output_schema,  # Use consistent schema
                            compression='snappy',
                            use_dictionary=True,
                            data_page_size=1024*1024
                        )
                    
                    # Write batch
                    writer.write_table(table)
                    processed_rows += num_rows
                    pbar.update(num_rows)
                    
                    # Clean up memory
                    del batch_dict, table, batch
                    if processed_rows % (CHUNK_SIZE * 10) == 0:
                        gc.collect()
                
                del row_group
                gc.collect()
    
    finally:
        if writer:
            writer.close()
    
    # Print summary
    print(f"\n✅ Done! Processed {processed_rows:,} rows")
    print(f"📊 Unique groups: {len(run_id_cache):,}")
    
    # File sizes
    input_mb = os.path.getsize(INPUT_FILE) / (1024**2)
    output_mb = os.path.getsize(OUTPUT_FILE) / (1024**2)
    print(f"📁 Input: {input_mb:.1f} MB → Output: {output_mb:.1f} MB")
    print(f"💾 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        process_parquet()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
