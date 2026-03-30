import os
import h5py
import numpy as np

# === Configuration ===
input_file = "input.h5"
output_dir = "npz_2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train"
os.makedirs(output_dir, exist_ok=True)

# === Read the file ===
with h5py.File(input_file, 'r') as f:
    # Required datasets
    Vx = f['Vx']
    Vy = f['Vy']
    density = f['density']
    pressure = f['pressure']

    # Optional coordinates
    x_coord = f['x-coordinate'][:]
    y_coord = f['y-coordinate'][:]

    nt, ns, nx, ny = Vx.shape
    print(f"Exporting {nt * ns} npz files...")

    for t in range(nt):
        for s in range(ns):
            # Extract 2D slice
            vx = Vx[t, s]
            vy = Vy[t, s]
            rho = density[t, s]
            p = pressure[t, s]

            # File name
            filename = f"sample_s{s:05d}_t{t:05d}.npz"
            filepath = os.path.join(output_dir, filename)

            # Save safely to npz (no pickled objects)
            np.savez_compressed(
                filepath,
                Vx=vx,
                Vy=vy,
                density=rho,
                pressure=p,
                x_coord=x_coord,
                y_coord=y_coord,
                time_step=np.array(t),
                sample_index=np.array(s)
            )

print(f"✅ Done. Saved to folder: {output_dir}")

