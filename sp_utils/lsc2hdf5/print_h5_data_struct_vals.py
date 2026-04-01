import h5py

def inspect_first_dataset_with_hierarchy(file_path):
    with h5py.File(file_path, 'r') as hdf:
        found = [False]  # Use list to make mutable inside nested function

        def visit(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}[Group] {name}")
            elif isinstance(obj, h5py.Dataset) and not found[0]:
                print(f"{indent}[Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
                data = obj[()]
                flat = data.flat if hasattr(data, 'flat') else data.ravel()
                print(f"{indent}  First 1 values: {list(flat)[:1]}")
                #found[0] = True
                #raise StopIteration  # Exit after printing the first dataset

        try:
            hdf.visititems(visit)
        except StopIteration:
            pass

# Replace with your actual .h5 file path
#inspect_first_dataset_with_hierarchy('cfd_rand.h5')
inspect_first_dataset_with_hierarchy('nsincom.h5')
#inspect_first_dataset_with_hierarchy('diff2d.h5')

