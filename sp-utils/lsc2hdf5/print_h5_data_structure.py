import h5py

def print_structure(name, obj):
    print(name)

def inspect_hdf5_structure(file_path):
    with h5py.File(file_path, 'r') as hdf:
        hdf.visititems(print_structure)

# Replace with your .h5 file path
inspect_hdf5_structure('diff2d.h5')
