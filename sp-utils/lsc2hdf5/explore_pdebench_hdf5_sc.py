import h5py
import os
import matplotlib.pyplot as plt

#file_path = "/lustre/scratch5/exempt/artimis/data/pdebench/2D/CFD/2D_Train_Rand/2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5"
file_path = "/lustre/scratch5/exempt/artimis/data/pdebench/2D/CFD/2D_Train_Rand/npz_2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train/sample_s00015_t010.npz"

def explore_hdf5(file_path):
    def print_attrs(name, obj):
        print(f"\n📂 Path: {name}")
        if isinstance(obj, h5py.Dataset):
            print(f"   └─ Dataset: shape = {obj.shape}, dtype = {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print("   └─ Group")

    with h5py.File(file_path, 'r') as f:
        print(f"✅ Opened file: {file_path}")
        print("🔍 Exploring contents...\n")
        f.visititems(print_attrs)

        # Optional: List top-level keys
        print("\n📁 Top-level keys:")
        for key in f.keys():
            print(f" - {key}")

        # Optional: Read a specific dataset if known
        # data = f["/some_dataset"][:]
        # print("\nSample data preview:", data[:5])

explore_hdf5(file_path)

"""
# Ensure 'results' directory exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Which sample and time step to plot
sample_idx = 200
time_idx = 18


# Output image path
output_fig = os.path.join(results_dir, f"cfd_slice_sample{sample_idx}_t{time_idx}.png")

# Open the file and extract the slice
with h5py.File(file_path, "r") as f:
    density = f["density"][sample_idx, time_idx, :, :]
    pressure = f["pressure"][sample_idx, time_idx, :, :]
    vx = f["Vx"][sample_idx, time_idx, :, :]
    vy = f["Vy"][sample_idx, time_idx, :, :]

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
cmap = 'viridis'

# Density
im1 = axs[0, 0].imshow(density, cmap=cmap)
axs[0, 0].set_title("Density")
plt.colorbar(im1, ax=axs[0, 0])

# Pressure
im2 = axs[0, 1].imshow(pressure, cmap=cmap)
axs[0, 1].set_title("Pressure")
plt.colorbar(im2, ax=axs[0, 1])

# Vx
im3 = axs[1, 0].imshow(vx, cmap=cmap)
axs[1, 0].set_title("Velocity X (Vx)")
plt.colorbar(im3, ax=axs[1, 0])

# Vy
im4 = axs[1, 1].imshow(vy, cmap=cmap)
axs[1, 1].set_title("Velocity Y (Vy)")
plt.colorbar(im4, ax=axs[1, 1])

fig.suptitle(f"Simulation {sample_idx}, Timestep {time_idx}: CFD Field Visualization\nDensity, Pressure, and Velocity Components", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Layout and save
plt.tight_layout()
plt.savefig(output_fig, dpi=300)
plt.close()

print(f"✅ Figure saved to: {os.path.abspath(output_fig)}")
"""
