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

os.makedirs("ground_motions_v2", exist_ok=True)


# ============================================================
# 2. MOTION DESIGN MATRIX
# ============================================================

# We still maintain four broad intensity levels
pga_levels = [0.20, 0.35, 0.50, 0.70]

# Five broad frequency bands
frequency_bands = [
    (0.8, 1.5),   # Low frequency
    (1.5, 2.8),   # Low-medium
    (2.8, 4.5),   # Medium
    (4.5, 6.5),   # Medium-high
    (6.5, 8.5)    # High frequency
]

# Approximate duration ranges for each band
duration_ranges = [
    (25, 50),
    (20, 40),
    (15, 35),
    (12, 30),
    (10, 25)
]


# ============================================================
# 3. GROUND MOTION GENERATOR
# ============================================================

def generate_ground_motion(
        target_pga_g,
        f_min,
        f_max,
        duration_min,
        duration_max,
        dt
):

    # --------------------------------------------------------
    # Random dominant frequency and duration
    # --------------------------------------------------------

    f_dom = np.random.uniform(f_min, f_max)

    duration = np.random.uniform(
        duration_min,
        duration_max
    )

    time = np.arange(
        0,
        duration + dt,
        dt
    )

    # --------------------------------------------------------
    # Multiple frequency components
    # --------------------------------------------------------

    n_components = np.random.randint(6, 12)

    frequencies = np.random.normal(
        loc=f_dom,
        scale=0.30 * f_dom,
        size=n_components
    )

    # Remove invalid frequencies
    frequencies = np.clip(
        frequencies,
        0.2,
        15.0
    )

    # Random amplitudes
    weights = np.random.uniform(
        0.2,
        1.0,
        n_components
    )

    # Give components closer to f_dom higher weight
    gaussian_weight = np.exp(
        -0.5
        * (
            (frequencies - f_dom)
            / (0.35 * f_dom)
        ) ** 2
    )

    weights = weights * gaussian_weight

    # Random phases
    phases = np.random.uniform(
        0,
        2 * np.pi,
        n_components
    )

    signal = np.zeros_like(time)

    for f, w, phi in zip(
            frequencies,
            weights,
            phases
    ):

        signal += (
            w
            * np.sin(
                2 * np.pi * f * time + phi
            )
        )

    # --------------------------------------------------------
    # Add weak broadband component
    # --------------------------------------------------------

    broadband_freqs = np.random.uniform(
        0.5,
        12.0,
        5
    )

    for f in broadband_freqs:

        phi = np.random.uniform(
            0,
            2 * np.pi
        )

        signal += (
            0.10
            * np.sin(
                2 * np.pi * f * time + phi
            )
        )

    # --------------------------------------------------------
    # Earthquake-type envelope
    # --------------------------------------------------------

    t_peak = np.random.uniform(
        0.20 * duration,
        0.40 * duration
    )

    envelope = np.zeros_like(time)

    for i, t in enumerate(time):

        if t <= t_peak:

            # Smooth energy build-up
            envelope[i] = (
                np.sin(
                    np.pi * t / (2 * t_peak)
                ) ** 2
            )

        else:

            # Random decay rate
            decay_rate = np.random.uniform(
                1.5,
                3.5
            )

            envelope[i] = np.exp(
                -decay_rate
                * (t - t_peak)
                / (duration - t_peak)
            )

    ag = signal * envelope

    # --------------------------------------------------------
    # Remove mean
    # --------------------------------------------------------

    ag = ag - np.mean(ag)

    # --------------------------------------------------------
    # Scale to target PGA
    # --------------------------------------------------------

    current_pga = np.max(np.abs(ag))

    ag = (
        ag
        / current_pga
        * target_pga_g
        * g
    )

    return time, ag, f_dom, duration


# ============================================================
# 4. FEATURE CALCULATION
# ============================================================

def calculate_features(time, ag, dt):

    # PGA
    pga = np.max(np.abs(ag))

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    velocity = np.cumsum(ag) * dt

    velocity = velocity - np.linspace(
        velocity[0],
        velocity[-1],
        len(velocity)
    )

    pgv = np.max(
        np.abs(velocity)
    )

    # --------------------------------------------------------
    # Displacement
    # --------------------------------------------------------

    displacement = np.cumsum(
        velocity
    ) * dt

    displacement = displacement - np.linspace(
        displacement[0],
        displacement[-1],
        len(displacement)
    )

    pgd = np.max(
        np.abs(displacement)
    )

    # --------------------------------------------------------
    # Arias Intensity
    # --------------------------------------------------------

    arias = (
        np.pi
        / (2 * g)
        * np.trapezoid(
            ag ** 2,
            time
        )
    )

    # --------------------------------------------------------
    # Significant Duration D5-95
    # --------------------------------------------------------

    cumulative_energy = np.cumsum(
        ag ** 2
    ) * dt

    cumulative_energy = (
        cumulative_energy
        / cumulative_energy[-1]
    )

    t5 = time[
        np.searchsorted(
            cumulative_energy,
            0.05
        )
    ]

    t95 = time[
        np.searchsorted(
            cumulative_energy,
            0.95
        )
    ]

    significant_duration = (
        t95 - t5
    )

    # --------------------------------------------------------
    # Dominant Frequency
    # --------------------------------------------------------

    n = len(ag)

    fft_values = np.fft.rfft(ag)

    fft_freq = np.fft.rfftfreq(
        n,
        dt
    )

    power = (
        np.abs(fft_values)
        ** 2
    )

    dominant_frequency = fft_freq[
        1 + np.argmax(power[1:])
    ]

    return {
        "PGA_g": pga / g,
        "PGV_cm_per_s": pgv * 100,
        "PGD_cm": pgd * 100,
        "Arias_Intensity": arias,
        "Significant_Duration_s":
            significant_duration,
        "Dominant_Frequency_Hz":
            dominant_frequency
    }


# ============================================================
# 5. GENERATE 20 MOTIONS
# ============================================================

all_features = []

gm_id = 1


for pga in pga_levels:

    for i in range(len(frequency_bands)):

        f_min, f_max = frequency_bands[i]

        d_min, d_max = duration_ranges[i]

        print(
            f"Generating GM{gm_id:03d}..."
        )

        # Generate motion
        time, ag, input_fdom, duration = \
            generate_ground_motion(
                pga,
                f_min,
                f_max,
                d_min,
                d_max,
                dt
            )

        # Extract features
        features = calculate_features(
            time,
            ag,
            dt
        )

        # Metadata
        features["GM_ID"] = (
            f"GM{gm_id:03d}"
        )

        features["Target_PGA_g"] = pga

        features["Input_Frequency_Hz"] = (
            input_fdom
        )

        features["Total_Duration_s"] = (
            duration
        )

        # Put ID first
        features = {
            "GM_ID": features.pop("GM_ID"),
            **features
        }

        all_features.append(
            features
        )

        # Save raw motion
        motion_df = pd.DataFrame({
            "Time_s": time,
            "Ground_Acceleration_m_per_s2": ag
        })

        motion_df.to_csv(
            f"ground_motions_v2/"
            f"GM{gm_id:03d}.csv",
            index=False
        )

        gm_id += 1


# ============================================================
# 6. SAVE FEATURE DATASET
# ============================================================

df = pd.DataFrame(
    all_features
)

df.to_csv(
    "ground_motion_features_v2.csv",
    index=False
)


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print(
    "GROUND MOTION DATASET V2"
)
print("=" * 80)

print(
    f"\nTotal motions: {len(df)}"
)

print("\nFEATURES:\n")

print(
    df.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    df[
        [
            "PGA_g",
            "PGV_cm_per_s",
            "PGD_cm",
            "Arias_Intensity",
            "Significant_Duration_s",
            "Dominant_Frequency_Hz"
        ]
    ].describe()
)


# ============================================================
# 8. QUALITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("QUALITY CHECK")
print("=" * 80)

print(
    f"PGA Range : "
    f"{df['PGA_g'].min():.3f} "
    f"to "
    f"{df['PGA_g'].max():.3f} g"
)

print(
    f"PGV Range : "
    f"{df['PGV_cm_per_s'].min():.2f} "
    f"to "
    f"{df['PGV_cm_per_s'].max():.2f} cm/s"
)

print(
    f"PGD Range : "
    f"{df['PGD_cm'].min():.2f} "
    f"to "
    f"{df['PGD_cm'].max():.2f} cm"
)

print(
    f"Duration Range : "
    f"{df['Significant_Duration_s'].min():.2f} "
    f"to "
    f"{df['Significant_Duration_s'].max():.2f} s"
)


# ============================================================
# 9. SAMPLE PLOTS
# ============================================================

sample_ids = [
    "GM001",
    "GM004",
    "GM007",
    "GM013",
    "GM020"
]

plt.figure(figsize=(12, 10))

for i, gm in enumerate(sample_ids):

    motion = pd.read_csv(
        f"ground_motions_v2/{gm}.csv"
    )

    plt.subplot(
        len(sample_ids),
        1,
        i + 1
    )

    plt.plot(
        motion["Time_s"],
        motion[
            "Ground_Acceleration_m_per_s2"
        ] / g
    )

    plt.ylabel("Accel (g)")

    plt.title(gm)

    plt.grid(True)


plt.xlabel("Time (s)")

plt.tight_layout()

plt.savefig(
    "sample_ground_motions_v2.png",
    dpi=300
)

plt.show()


print(
    "\n✓ Ground motions saved "
    "in ground_motions_v2/"
)

print(
    "✓ Features saved as "
    "ground_motion_features_v2.csv"
)