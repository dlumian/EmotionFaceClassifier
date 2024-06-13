import os
import pandas as pd

def generate_file_dataframe(root_dir):
    data = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                # Get the relative path and split into parts
                rel_path = os.path.relpath(root, root_dir)
                parts = rel_path.split(os.sep)
                
                # Collect data
                record = parts + [file, os.path.join(root, file)]
                data.append(record)
    
    # Define the columns based on the maximum directory depth + 1 for the file name + 1 for the full path
    max_depth = max(len(record) - 2 for record in data)
    columns = [f'Level {i}' for i in range(1, max_depth + 1)] + ['Filename', 'Full Path']
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    df.rename(columns={
            'Level 1': 'train_test_split', 
            'Level 2': 'emotion'
        },
        inplace=True
    )
    
    return df


