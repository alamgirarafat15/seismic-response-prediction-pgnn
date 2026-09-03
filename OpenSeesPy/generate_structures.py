import numpy as np
import pandas as pd
from itertools import product


# ============================================================
# 1. STRUCTURAL PARAMETER CONFIGURATION
# ============================================================

# Natural period values (seconds)
T_values = [0.40, 0.80, 1.30, 2.00, 2.80]

# Strength ratio
# SR = Fy / (m * g)
SR_values = [0.10, 0.20, 0.30, 0.40, 0.50]

# Damping ratios
zeta_values = [0.02, 0.05]

# Effective mass
mass = 1.0e6  # kg

# Gravity
g = 9.81  # m/s²

# Post-yield stiffness ratio
alpha = 0.03

# Effective height for each natural period
height_map = {
    0.40: 6.0,
    0.80: 12.0,
    1.30: 21.0,
    2.00: 33.0,
    2.80: 45.0
}


# ============================================================
# 2. GENERATE STRUCTURAL CONFIGURATIONS
# ============================================================

structures = []

structure_id = 1

for T, SR, zeta in product(T_values, SR_values, zeta_values):

    # Natural circular frequency
    omega_n = 2 * np.pi / T

    # Initial stiffness
    k = mass * omega_n ** 2

    # Damping coefficient
    c = 2 * zeta * mass * omega_n

    # Yield strength
    Fy = SR * mass * g

    # Yield displacement
    uy = Fy / k

    # Post-yield stiffness
    k_post = alpha * k

    # Effective height
    H = height_map[T]

    structures.append({
        "Structure_ID": f"S{structure_id:03d}",
        "Natural_Period_s": T,
        "Strength_Ratio": SR,
        "Damping_Ratio": zeta,
        "Mass_kg": mass,
        "Height_m": H,
        "Initial_Stiffness_N_per_m": k,
        "Damping_Coefficient_Ns_per_m": c,
        "Yield_Strength_N": Fy,
        "Yield_Displacement_m": uy,
        "PostYield_Ratio": alpha,
        "PostYield_Stiffness_N_per_m": k_post
    })

    structure_id += 1


# ============================================================
# 3. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(structures)


# ============================================================
# 4. QUALITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("STRUCTURAL DATASET GENERATED")
print("=" * 70)

print(f"\nTotal structures: {len(df)}")

print("\nFirst 10 structures:")
print(df.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(df.describe())


# Physical consistency checks
assert (df["Initial_Stiffness_N_per_m"] > 0).all()
assert (df["Damping_Coefficient_Ns_per_m"] > 0).all()
assert (df["Yield_Strength_N"] > 0).all()
assert (df["Yield_Displacement_m"] > 0).all()

assert (
    df["PostYield_Stiffness_N_per_m"]
    < df["Initial_Stiffness_N_per_m"]
).all()

print("\n✓ All physical consistency checks passed!")


# ============================================================
# 5. SAVE CSV
# ============================================================

output_file = "structural_parameters.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print(f"✓ Dataset saved successfully as: {output_file}")
print("=" * 70)