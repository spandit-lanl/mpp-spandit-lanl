import h5py

def print_structure(name, obj):
    indent = '  ' * name.count('/')
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}- Dataset: {name} | shape: {obj.shape} | dtype: {obj.dtype}", flush=True)
    elif isinstance(obj, h5py.Group):
        print(f"{indent}- Group: {name}", flush=True)

def explore_hdf5_structure(file_path):
    with h5py.File(file_path, 'r') as hdf:
        print(f"File: {file_path}", flush=True)
        hdf.visititems(print_structure)

def print_attributes(file_path):
    with h5py.File(file_path, 'r') as hdf:
        for key in ['t-coordinate', 'x-coordinate', 'y-coordinate']:
            if key in hdf:
                dset = hdf[key]
                print(f"\nAttributes of '{key}':", flush=True)
                for attr_name, attr_value in dset.attrs.items():
                    print(f"  {attr_name}: {attr_value}", flush=True)
            else:
                print(f"{key} not found in the file.", flush=True)

def inspect_coordinates(file_path):
    with h5py.File(file_path, 'r') as hdf:
        for name in ['t-coordinate', 'x-coordinate', 'y-coordinate']:
            if name in hdf:
                data = hdf[name][:]
                print(f"{name}: shape = {data.shape}, dtype = {data.dtype}", flush=True)
                print(f"  min = {data.min()}, max = {data.max()}, mean spacing ≈ {(data[-1] - data[0]) / (len(data)-1) if len(data) > 1 else 'N/A'}\n", flush=True)

def inspect_first_dataset_with_hierarchy(file_path):
    with h5py.File(file_path, 'r') as hdf:
        found = [False]  # Use list to make mutable inside nested function

        def visit(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}[Group] {name}", flush=True)
            elif isinstance(obj, h5py.Dataset) and not found[0]:
                print(f"{indent}[Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}", flush=True)
                data = obj[()]
                flat = data.flat if hasattr(data, 'flat') else data.ravel()
                print(f"{indent}  First 1 values: {list(flat)[:1]}", flush=True)
                #found[0] = True
                #raise StopIteration  # Exit after printing the first dataset

        try:
            hdf.visititems(visit)
        except StopIteration:
            pass


if __name__ == "__main__":
    explore_hdf5_structure("input.h5")
    inspect_coordinates("input.h5")

    # Replace with your actual .h5 file path
    #inspect_first_dataset_with_hierarchy('cfd_rand.h5')
    inspect_first_dataset_with_hierarchy('input.h5')
    #inspect_first_dataset_with_hierarchy('diff2d.h5')
