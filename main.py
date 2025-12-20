import os
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px

# --- 1. KONFIGURASI GPU ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"🔥 Menghitung Fisika di GPU: {tf.config.experimental.get_device_details(gpus[0])['device_name']}")

# --- 2. PARAMETER ---
N_PARTICLES = 2500
N_FRAMES = 100

# Inisialisasi awal (Tensor biasa)
pos = tf.zeros([N_PARTICLES, 3], dtype=tf.float32)

# Kecepatan ledakan awal
angles = tf.random.uniform([N_PARTICLES], 0, 2 * np.pi)
phi = tf.random.uniform([N_PARTICLES], 0, np.pi)
speeds = tf.random.uniform([N_PARTICLES], 0.05, 0.2)

vx = speeds * tf.sin(phi) * tf.cos(angles)
vy = speeds * tf.sin(phi) * tf.sin(angles)
vz = speeds * tf.cos(phi)
vel = tf.stack([vx, vy, vz], axis=1)


# --- 3. FUNGSI FISIKA (Tanpa Assign/Loop Error) ---
@tf.function
def compute_next_step(p, v):
    p_new = p + v
    # Gaya tarik inti (gravitasi atom)
    dist_sq = tf.reduce_sum(p_new ** 2, axis=1, keepdims=True) + 1e-6
    attraction = -p_new * (0.0005 / dist_sq)
    v_new = (v + attraction) * 0.97  # Friksi 3%
    return p_new, v_new


# --- 4. PRE-COMPUTING (Menghitung semua frame dulu) ---
print("⚛️ Menghitung evolusi atom... Mohon tunggu.")
all_frames = []

for f in range(N_FRAMES):
    pos, vel = compute_next_step(pos, vel)

    df_f = pd.DataFrame(pos.numpy(), columns=['X', 'Y', 'Z'])
    df_f['Frame'] = f
    df_f['Size'] = 2  # Ukuran titik
    all_frames.append(df_f)

df_final = pd.concat(all_frames)

# --- 5. VISUALISASI DENGAN TOMBOL PAUSE ---
print("✨ Membuka Dashboard... Gunakan tombol Play/Pause di bawah grafik.")

fig = px.scatter_3d(
    df_final, x='X', y='Y', z='Z',
    animation_frame='Frame',
    range_x=[-5, 5], range_y=[-5, 5], range_z=[-5, 5],
    title="Oppenheimer Atomic Bloom (3D Interaktif + Pause)",
    template="plotly_dark",
    opacity=0.6
)

# Kustomisasi Tombol agar ada tombol Pause yang jelas
fig.update_layout(
    updatemenus=[dict(
        type="buttons",
        buttons=[
            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50}}]),
            dict(label="Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
        ]
    )]
)

fig.show()