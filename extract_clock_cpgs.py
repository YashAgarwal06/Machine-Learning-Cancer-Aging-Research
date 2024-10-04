import pandas as pd
import time
import threading

# Define file paths
tsv_file = 'kipan_clin_meth_20221210.tsv'
union_file = 'union_out.csv'
output_file = 'kipan_clin_meth_20221210_extract_union.tsv'

# Load the union set of CpGs
union_cpgs = pd.read_csv(union_file, header=None)[0].tolist()

# Define chunk size (number of rows to read at a time)
chunk_size = 100

# Initialize chunk counter
chunk_counter = 0

# Define a flag for when processing is complete
done = False

# Define a function to print a progress update every 60 seconds
def print_progress():
    while not done:
        print(f'Processed {chunk_counter} chunks so far')
        time.sleep(60)

# Start the progress update function in a separate thread
progress_thread = threading.Thread(target=print_progress)
progress_thread.start()

# Iterate over chunks of the TSV file
for chunk in pd.read_csv(tsv_file, sep='\t', chunksize=chunk_size):
    # Update the chunk counter
    chunk_counter += 1

    # Intersect the union set with the columns in the chunk
    valid_cpgs = list(set(union_cpgs) & set(chunk.columns))

    # Keep only the columns that are in the valid CpGs
    filtered_chunk = chunk[valid_cpgs]

    # Write the filtered chunk to the output file
    filtered_chunk.to_csv(output_file, index=False, sep='\t', mode='a')

# Set the done flag to True to terminate the progress update thread
done = True

# Wait for the progress update thread to terminate
progress_thread.join()

print(f'Filtered data has been written to {output_file}')