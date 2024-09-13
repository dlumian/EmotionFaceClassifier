import os
import shutil
import glob
from nbconvert.preprocessors import ClearOutputPreprocessor, ExecutePreprocessor

def check_directory_name(target_name) -> bool:
    """
    Check if the current directory name matches the target_name.
    If not, move up a directory and repeat the check.
    
    Args:
        target_name (str): The directory name to match.
        
    Returns:
        bool: True if the current directory name matches the target_name, False otherwise.
    """
    # Get the current directory path
    current_dir = os.getcwd()
    # Extract the directory name from the path
    current_dir_name = os.path.basename(current_dir)    
    # Check if the current directory name matches the target_name
    if current_dir_name == target_name:
        print(f'Directory set to {current_dir}, matches target dir sting {target_name}.')
        return True
    else:
        # Move up a directory
        os.chdir('..')
        # Check if we have reached the root directory
        if os.getcwd() == current_dir:
            return False
        # Recursively call the function to check the parent directory
        return check_directory_name(target_name)


def remove_files(file_patterns):
    for pattern in file_patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"Removed file: {file}")
            except OSError as e:
                print(f"Error removing file {file}: {e}")

def remove_directories(directories):
    for directory in directories:
        try:
            shutil.rmtree(directory)
            print(f"Removed directory: {directory}")
        except OSError as e:
            print(f"Error removing directory {directory}: {e}")

def reset_notebooks(notebook_dir):
    notebook_files = glob.glob(os.path.join(notebook_dir, "*.ipynb"))
    
    for notebook_file in notebook_files:
        try:
            with open(notebook_file, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Clear outputs
            clear_output_preprocessor = ClearOutputPreprocessor()
            nb, _ = clear_output_preprocessor.preprocess(nb, {})

            # Save the cleared notebook
            with open(notebook_file, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            
            print(f"Cleared output in notebook: {notebook_file}")

        except Exception as e:
            print(f"Failed to clear output in {notebook_file}: {e}")

def reset_project():
    # Define file patterns to remove
    file_patterns = [
        'data/intermediate/*.csv',
        'data/intermediate/*.jpg',
        'models/*',
        'metrics/*'
    ]
    
    # Define directories to remove
    directories = [
        'data/intermediate/',
        'models/',
        'metrics/',
        'imgs'
    ]
    
    # Remove files and directories
    remove_files(file_patterns)
    remove_directories(directories)
    
    # Reset notebooks
    reset_notebooks('notebooks') 
    
    print("Project reset completed.")

if __name__ == "__main__":
    confirm = input("Are you sure you want to reset the project? This will delete intermediate files, directories, and reset notebooks. (yes/no): ")
    if confirm.lower() == 'yes':
        main_dir = 'EmotionFaceClassifier'
        check_directory_name(main_dir)
        reset_project()
    else:
        print("Project reset aborted.")
