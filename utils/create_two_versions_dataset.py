import os 
import shutil

src_dir = "/Users/ammarahmed/Downloads/Images"
dest_dir = "/Users/ammarahmed/Downloads/Images2"


def get_directory_size(directory):
    """Returns the total size of the directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size

def move_files_until_size_reached(src_dir, dest_dir, size_limit_gb):
    """Moves files from src_dir to dest_dir until dest_dir size reaches size_limit_gb."""
    size_limit_bytes = size_limit_gb * 1024**3  # Convert GB to bytes

    # Get initial size of the destination directory
    dest_size = get_directory_size(dest_dir)

    # List all files in the source directory
    files = sorted(os.listdir(src_dir))  # Sort files to move them in a specific order (optional)

    for file_name in files:
        file_path = os.path.join(src_dir, file_name)

        # Skip if it's not a file
        if not os.path.isfile(file_path):
            continue

        # Get the size of the current file
        file_size = os.path.getsize(file_path)

        # Check if moving this file would exceed the limit
        if dest_size + file_size > size_limit_bytes:
            print(f"Stopping before moving {file_name}. Would exceed 5GB limit.")
            break

        # Move the file
        shutil.move(file_path, dest_dir)
        print(f"Moved: {file_name}")

        # Update the destination directory size
        dest_size += file_size

    print(f"Final size of destination directory: {dest_size / (1024**3):.2f} GB")


# Call the function to move files until the destination directory reaches 5GB
move_files_until_size_reached(src_dir, dest_dir, 5)

