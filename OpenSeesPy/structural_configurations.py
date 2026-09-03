import pandas as pd
import numpy as np
import random


# ============================================================
# STRUCTURAL CONFIGURATION GENERATOR
# ============================================================
# Generates controlled RC frame configurations for
# OpenSees-based nonlinear/time-history simulations.
#
# These are INPUT configurations.
# Actual dynamic properties such as:
#   - Fundamental Period
#   - Natural Frequency
#   - Modal Properties
# will be obtained later from OpenSees.
# ============================================================


# Reproducibility
random.seed(42)
np.random.seed(42)


# ============================================================
# USER SETTINGS
# ============================================================

NUM_STRUCTURES = 50

OUTPUT_FILE = "structural_configurations.csv"


# ============================================================
# DEFINE SCIENTIFICALLY CONTROLLED PARAMETER OPTIONS
# ============================================================

STORY_OPTIONS = [
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    15
]


STORY_HEIGHT_OPTIONS = [
    3.0,
    3.2,
    3.5,
    3.8
]


BAY_WIDTH_OPTIONS = [
    4.0,
    4.5,
    5.0,
    5.5,
    6.0
]


NUM_BAYS_OPTIONS = [
    2,
    3,
    4
]


# Column dimensions in metres
COLUMN_SECTION_OPTIONS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65
]


# Beam width options
BEAM_WIDTH_OPTIONS = [
    0.25,
    0.30,
    0.35,
    0.40
]


# Beam depth options
BEAM_DEPTH_OPTIONS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


# Seismic weight intensity
# kN/m²
SEISMIC_LOAD_OPTIONS = [
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5
]


# Damping ratios
DAMPING_OPTIONS = [
    0.02,
    0.03,
    0.05,
    0.07
]


# ============================================================
# GEOMETRIC CONSISTENCY CHECK
# ============================================================

def is_valid_configuration(
    column_size,
    beam_width,
    beam_depth,
    bay_width
):

    # Beam depth should be smaller than bay span / 8
    if beam_depth > bay_width / 8:
        return False

    # Column should generally be larger than beam width
    if column_size < beam_width:
        return False

    # Avoid extremely deep beam relative to column
    if beam_depth > column_size * 1.5:
        return False

    return True


# ============================================================
# GENERATE UNIQUE CONFIGURATIONS
# ============================================================

def generate_configurations():

    configurations = []

    generated_combinations = set()


    while len(configurations) < NUM_STRUCTURES:

        num_stories = random.choice(
            STORY_OPTIONS
        )

        story_height = random.choice(
            STORY_HEIGHT_OPTIONS
        )

        num_bays_x = random.choice(
            NUM_BAYS_OPTIONS
        )

        num_bays_y = random.choice(
            NUM_BAYS_OPTIONS
        )

        bay_width_x = random.choice(
            BAY_WIDTH_OPTIONS
        )

        bay_width_y = random.choice(
            BAY_WIDTH_OPTIONS
        )

        column_size = random.choice(
            COLUMN_SECTION_OPTIONS
        )

        beam_width = random.choice(
            BEAM_WIDTH_OPTIONS
        )

        beam_depth = random.choice(
            BEAM_DEPTH_OPTIONS
        )

        seismic_load = random.choice(
            SEISMIC_LOAD_OPTIONS
        )

        damping_ratio = random.choice(
            DAMPING_OPTIONS
        )


        # ----------------------------------------------------
        # PHYSICAL CONSISTENCY CHECK
        # ----------------------------------------------------

        valid = is_valid_configuration(

            column_size,
            beam_width,
            beam_depth,
            min(
                bay_width_x,
                bay_width_y
            )

        )


        if not valid:
            continue


        # ----------------------------------------------------
        # CHECK DUPLICATE CONFIGURATION
        # ----------------------------------------------------

        combination = (

            num_stories,
            story_height,

            num_bays_x,
            num_bays_y,

            bay_width_x,
            bay_width_y,

            column_size,

            beam_width,
            beam_depth,

            seismic_load,

            damping_ratio

        )


        if combination in generated_combinations:
            continue


        generated_combinations.add(
            combination
        )


        # ----------------------------------------------------
        # CALCULATED GEOMETRY
        # ----------------------------------------------------

        total_height = (

            num_stories
            *
            story_height

        )


        floor_length = (

            num_bays_x
            *
            bay_width_x

        )


        floor_width = (

            num_bays_y
            *
            bay_width_y

        )


        floor_area = (

            floor_length
            *
            floor_width

        )


        # ----------------------------------------------------
        # CREATE CONFIGURATION
        # ----------------------------------------------------

        structure_id = (

            f"S{len(configurations) + 1:03d}"

        )


        configuration = {

            "Structure_ID":
                structure_id,

            "Num_Stories":
                num_stories,

            "Story_Height_m":
                story_height,

            "Total_Height_m":
                total_height,

            "Num_Bays_X":
                num_bays_x,

            "Num_Bays_Y":
                num_bays_y,

            "Bay_Width_X_m":
                bay_width_x,

            "Bay_Width_Y_m":
                bay_width_y,

            "Floor_Area_m2":
                floor_area,

            "Column_Size_m":
                column_size,

            "Beam_Width_m":
                beam_width,

            "Beam_Depth_m":
                beam_depth,

            "Seismic_Load_kN_m2":
                seismic_load,

            "Damping_Ratio":
                damping_ratio

        }


        configurations.append(
            configuration
        )


    return pd.DataFrame(
        configurations
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 75)

    print(
        "STRUCTURAL CONFIGURATION GENERATOR"
    )

    print("=" * 75)


    # --------------------------------------------------------
    # GENERATE DATA
    # --------------------------------------------------------

    df = generate_configurations()


    # --------------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # --------------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    print(

        f"\nTotal Configurations Generated : "

        f"{len(df)}"

    )


    print("\n" + "-" * 75)

    print(
        "FIRST 10 CONFIGURATIONS"
    )

    print("-" * 75)

    print(
        df.head(10).to_string(
            index=False
        )
    )


    print("\n" + "-" * 75)

    print(
        "PARAMETER RANGES"
    )

    print("-" * 75)


    print(

        f"\nStories Range : "

        f"{df['Num_Stories'].min()}"

        f" to "

        f"{df['Num_Stories'].max()}"

    )


    print(

        f"Total Height Range : "

        f"{df['Total_Height_m'].min():.2f}"

        f" to "

        f"{df['Total_Height_m'].max():.2f} m"

    )


    print(

        f"Floor Area Range : "

        f"{df['Floor_Area_m2'].min():.2f}"

        f" to "

        f"{df['Floor_Area_m2'].max():.2f} m²"

    )


    print(

        f"Column Size Range : "

        f"{df['Column_Size_m'].min():.2f}"

        f" to "

        f"{df['Column_Size_m'].max():.2f} m"

    )


    print(

        f"Seismic Load Range : "

        f"{df['Seismic_Load_kN_m2'].min():.1f}"

        f" to "

        f"{df['Seismic_Load_kN_m2'].max():.1f} kN/m²"

    )


    print("\n" + "=" * 75)

    print(

        f"✓ CONFIGURATIONS SAVED AS: "

        f"{OUTPUT_FILE}"

    )

    print("=" * 75)