import h5py
'''
def inspect_first_group_tree(file_path):
    with h5py.File(file_path, 'r') as hdf:
        top_keys = list(hdf.keys())
        if not top_keys:
            print("No groups or datasets found in the file.")
            return

        first_key = top_keys[0]
        first_obj = hdf[first_key]

        print("Tree Structure (first top-level group only):")
        print_branch(first_obj, prefix='', level=0, name=first_key)

def print_branch(obj, prefix='', level=0, name=''):
    indent = prefix + "└── "
    if isinstance(obj, h5py.Group):
        print(f"{indent}[Group] {name or '/'}")
        for key in obj:
            sub_obj = obj[key]
            new_prefix = prefix + "│   "
            print_branch(sub_obj, new_prefix, level+1, key)
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}[Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
        data = obj[()]
        flat = data.flat if hasattr(data, 'flat') else data.ravel()
        #print(f"{prefix}    First 10 values: {list(flat)[:10]}")
'''
import h5py

def inspect_first_group_tree_with_channel_names(file_path):
    with h5py.File(file_path, 'r') as hdf:
        top_keys = list(hdf.keys())
        if not top_keys:
            print("No groups or datasets found in the file.")
            return

        first_key = top_keys[0]
        first_obj = hdf[first_key]

        print("Tree Structure (first top-level group only):")
        print_branch(first_obj, prefix='', level=0, name=first_key)

def print_branch(obj, prefix='', level=0, name=''):
    indent = prefix + "└── "

    if isinstance(obj, h5py.Group):
        print(f"{indent}[Group] {name or '/'}")
        for key in obj:
            sub_obj = obj[key]
            new_prefix = prefix + "│   "
            print_branch(sub_obj, new_prefix, level + 1, key)

    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}[Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
        print_channel_names_if_any(obj, prefix)

def print_channel_names_if_any(dataset, prefix):
    found = False
    for key in dataset.attrs:
        if 'name' in key.lower() or 'field' in key.lower() or 'var' in key.lower():
            names = dataset.attrs[key]
            try:
                # Decode bytes if needed
                decoded = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in names]
                print(f"{prefix}    Channel names ({key}): {decoded}")
                found = True
            except Exception:
                pass
    if not found:
        print(f"{prefix}    No channel/variable names found in attributes.")

# 🔁 Replace with your file name
print("\n\nSWE")
inspect_first_group_tree_with_channel_names('swe.h5')

print("\n\nDIFF2d")
inspect_first_group_tree_with_channel_names('diff2d.h5')

print("\n\nNSIMCOM")
inspect_first_group_tree_with_channel_names('nsincom.h5')

print("\n\nCFD_RAND")
inspect_first_group_tree_with_channel_names('cfd_rand.h5')

'''
# Replace with your file path
print("\n\nDIFF2d")
inspect_first_group_tree('diff2d.h5')

print("\n\nNSIMCOM")
inspect_first_group_tree('nsincom.h5')

print("\n\nCFD_RAND")
inspect_first_group_tree('cfd_rand.h5')

print("\n\nDIFF2d")
inspect_first_group_tree('diff2d.h5')
'''
