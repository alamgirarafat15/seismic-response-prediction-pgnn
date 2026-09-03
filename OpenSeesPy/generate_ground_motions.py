import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

np.random.seed(42)

g = 9.81
dt = 0.01

# Output folder
os.makedirs("ground_motions", exist_ok=True)


# ============================================================
# 2. GROUND MOTION MATRIX
# ============================================================

# Target PGA levels (g)
pga_levels = [0.20, 0.35, 0.50, 0.70]

# Motion profiles:
# (dominant frequency Hz, duration seconds)
profiles = [
    (1.0, 40.0),
    (2.0, 30.0),
    (4.0, 25.0),
    (6.0, 20.0),
    (8.0, 15.0),
]


# ============================================================
# 3. MOTION GENERATOR
# ============================================================

def generate_ground_motion(target_pga_g, f_dom, duration, dt):

    time = np.arange(0, duration + dt, dt)

    # --------------------------------------------------------
    # Multi-frequency components around dominant frequency
    # --------------------------------------------------------

    frequencies = np.array([
        0.70 * f_dom,
        0.85 * f_dom,
        1.00 * f_dom,
        1.15 * f_dom,
        1.30 * f_dom
    ])

    # Dominant frequency gets highest weight
    weights = np.array([
        0.20,
        0.50,
        1.00,
        0.50,
        0.20
    ])

    # Random phases
    phases = np.random.uniform(
        0,
        2 * np.pi,
        len(frequencies)
    )

    signal = np.zeros_like(time)

    for f, w, phi in zip(frequencies, weights, phases):

        signal += (
            w
            * np.sin(2 * np.pi * f * time + phi)
        )

    # --------------------------------------------------------
    # Earthquake-like envelope
    # --------------------------------------------------------

    # Strong shaking roughly occurs around 30% of duration
    t_peak = 0.30 * duration

    envelope = np.zeros_like(time)

    for i, t in enumerate(time):

        if t <= t_peak:

            # Smooth build-up
            envelope[i] = (
                np.sin(
                    np.pi * t / (2 * t_peak)
                ) ** 2
            )

        else:

            # Exponential decay
            envelope[i] = np.exp(
                -2.5
                * (t - t_peak)
                / (duration - t_peak)
            )

    # Apply envelope
    ag = signal * envelope

    # --------------------------------------------------------
    # Remove mean
    # --------------------------------------------------------

    ag = ag - np.mean(ag)

    # --------------------------------------------------------
    # Scale to target PGA
    # --------------------------------------------------------

    current_pga = np.max(np.abs(ag))

    ag = ag / current_pga

    ag = ag * target_pga_g * g

    return time, ag


# ============================================================
# 4. FEATURE CALCULATION
# ============================================================

def calculate_features(time, ag, dt):

    # --------------------------------------------------------
    # PGA
    # --------------------------------------------------------

    pga = np.max(np.abs(ag))

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    velocity = np.cumsum(ag) * dt

    # Remove linear trend
    velocity = velocity - np.linspace(
        velocity[0],
        velocity[-1],
        len(velocity)
    )

    pgv = np.max(np.abs(velocity))

    # --------------------------------------------------------
    # Displacement
    # --------------------------------------------------------

    displacement = np.cumsum(velocity) * dt

    # Remove linear trend
    displacement = displacement - np.linspace(
        displacement[0],
        displacement[-1],
        len(displacement)
    )

    pgd = np.max(np.abs(displacement))

    # --------------------------------------------------------
    # Arias Intensity
    # AI = pi / (2g) * integral(a^2 dt)
    # --------------------------------------------------------

    arias = (
        np.pi / (2 * g)
        * np.trapezoid(ag ** 2, time)
    )

    # --------------------------------------------------------
    # Significant Duration D5-95
    # --------------------------------------------------------

    cumulative_ai = np.cumsum(ag ** 2) * dt

    cumulative_ai = (
        cumulative_ai
        / cumulative_ai[-1]
    )

    t5 = time[
        np.searchsorted(
            cumulative_ai,
            0.05
        )
    ]

    t95 = time[
        np.searchsorted(
            cumulative_ai,
            0.95
        )
    ]

    significant_duration = t95 - t5

    # --------------------------------------------------------
    # Dominant Frequency using FFT
    # --------------------------------------------------------

    n = len(ag)

    fft_values = np.fft.rfft(ag)

    fft_freq = np.fft.rfftfreq(
        n,
        dt
    )

    power = np.abs(fft_values) ** 2

    # Ignore zero frequency
    dominant_frequency = fft_freq[
        1 + np.argmax(power[1:])
    ]

    return {
        "PGA_m_per_s2": pga,
        "PGA_g": pga / g,
        "PGV_m_per_s": pgv,
        "PGV_cm_per_s": pgv * 100,
        "PGD_m": pgd,
        "PGD_cm": pgd * 100,
        "Arias_Intensity_m_per_s": arias,
        "Significant_Duration_s": significant_duration,
        "Dominant_Frequency_Hz": dominant_frequency
    }


# ============================================================
# 5. GENERATE 20 GROUND MOTIONS
# ============================================================

all_features = []

gm_id = 1

for target_pga in pga_levels:

    for f_dom, duration in profiles:

        print(
            f"Generating GM{gm_id:03d}..."
        )

        # Generate motion
        time, ag = generate_ground_motion(
            target_pga,
            f_dom,
            duration,
            dt
        )

        # Calculate features
        features = calculate_features(
            time,
            ag,
            dt
        )

        # Add metadata
        features["GM_ID"] = f"GM{gm_id:03d}"
        features["Target_PGA_g"] = target_pga
        features["Input_Dominant_Frequency_Hz"] = f_dom
        features["Total_Duration_s"] = duration

        # Put ID first
        features = {
            "GM_ID": features.pop("GM_ID"),
            **features
        }

        all_features.append(features)

        # Save time history
        motion_df = pd.DataFrame({
            "Time_s": time,
            "Ground_Acceleration_m_per_s2": ag
        })

        filename = (
            f"ground_motions/GM{gm_id:03d}.csv"
        )

        motion_df.to_csv(
            filename,
            index=False
        )

        gm_id += 1


# ============================================================
# 6. CREATE FEATURE DATASET
# ============================================================

df_features = pd.DataFrame(all_features)

df_features.to_csv(
    "ground_motion_features.csv",
    index=False
)


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("GROUND MOTION DATASET GENERATED")
print("=" * 80)

print(f"\nTotal Ground Motions: {len(df_features)}")

print("\nFEATURES:")
print(
    df_features.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(
    df_features[
        [
            "PGA_g",
            "PGV_cm_per_s",
            "PGD_cm",
            "Arias_Intensity_m_per_s",
            "Significant_Duration_s",
            "Dominant_Frequency_Hz"
        ]
    ].describe()
)


# ============================================================
# 8. PLOT SAMPLE GROUND MOTIONS
# ============================================================

sample_ids = ["GM001", "GM003", "GM007", "GM013", "GM020"]

plt.figure(figsize=(12, 8))

for i, gm in enumerate(sample_ids):

    motion = pd.read_csv(
        f"ground_motions/{gm}.csv"
    )

    plt.subplot(
        len(sample_ids),
        1,
        i + 1
    )

    plt.plot(
        motion["Time_s"],
        motion["Ground_Acceleration_m_per_s2"] / g
    )

    plt.ylabel("Accel (g)")
    plt.title(gm)
    plt.grid(True)

plt.xlabel("Time (s)")

plt.tight_layout()

plt.savefig(
    "sample_ground_motions.png",
    dpi=300
)

plt.show()


print("\n✓ Ground motion files saved in: ground_motions/")
print("✓ Feature dataset saved as: ground_motion_features.csv")
print("✓ Sample plot saved as: sample_ground_motions.png")