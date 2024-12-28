import os
import nibabel as nib

def list_nii_shapes(folder_path):
    """
    Lists the shapes of all .nii and .nii.gz files in a folder.

    Args:
        folder_path (str): Path to the folder containing .nii files.

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    nii_files = [f for f in os.listdir(folder_path) if f.endswith(('.nii', '.nii.gz'))]
    
    if not nii_files:
        print(f"No .nii or .nii.gz files found in '{folder_path}'.")
        return

    print(f"Listing shapes of .nii files in folder: {folder_path}")
    for nii_file in nii_files:
        file_path = os.path.join(folder_path, nii_file)
        try:
            nii_data = nib.load(file_path)
            shape = nii_data.shape
            print(f"File: {nii_file}, Shape: {shape}")
        except Exception as e:
            print(f"Error loading file {nii_file}: {e}")

if __name__ == "__main__":
    # Update this path to the folder containing your .nii or .nii.gz files
    folder_path = r"C:\Users\claya\Task07_Pancreas\imagesTr"
    list_nii_shapes(folder_path)
