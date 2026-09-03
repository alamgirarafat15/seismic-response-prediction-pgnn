import os
import time
import numpy as np
import pandas as pd


from run_parameterized_nltha import (
    run_parameterized_nltha
)


from build_parameterized_rc_frame import (
    load_configuration
)


# ============================================================
# FULL BATCH NLTHA DATASET GENERATION
# ============================================================

NUM_STRUCTURES = 50

OUTPUT_FOLDER = "batch_nltha_results"

SUCCESS_FILE = os.path.join(
    OUTPUT_FOLDER,
    "nltha_dataset.csv"
)

FAILURE_FILE = os.path.join(
    OUTPUT_FOLDER,
    "failed_simulations.csv"
)

PROGRESS_FILE = os.path.join(
    OUTPUT_FOLDER,
    "batch_progress.csv"
)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# SAFE VALUE
# ============================================================

def safe_value(value):

    if value is None:

        return np.nan


    try:

        value = float(value)


        if not np.isfinite(value):

            return np.nan


        return value


    except Exception:

        return value


# ============================================================
# LOAD GROUND MOTION FEATURES
# ============================================================

def load_ground_motion_features():

    feature_file = (
        "ground_motion_features_v2.csv"
    )


    if not os.path.exists(feature_file):

        raise FileNotFoundError(

            "\nGround motion feature file not found:\n"
            f"{feature_file}"

        )


    df = pd.read_csv(
        feature_file
    )


    if "GM_ID" not in df.columns:

        raise ValueError(

            "\nColumn 'GM_ID' not found in "
            "ground motion feature file."
        )


    print(
        f"\n✓ Ground motion features loaded: "
        f"{len(df)} records"
    )


    return df


# ============================================================
# EXTRACT STRUCTURAL FEATURES
# ============================================================

def extract_structure_features(
    structure_id
):


    # --------------------------------------------------------
    # USE EXISTING FUNCTION FROM
    # build_parameterized_rc_frame.py
    # --------------------------------------------------------

    config = load_configuration(
        structure_id
    )


    num_stories = int(
        config["Num_Stories"]
    )


    story_height = float(
        config["Story_Height_m"]
    )


    num_bays_x = int(
        config["Num_Bays_X"]
    )


    num_bays_y = int(
        config["Num_Bays_Y"]
    )


    bay_width_x = float(
        config["Bay_Width_X_m"]
    )


    bay_width_y = float(
        config["Bay_Width_Y_m"]
    )


    column_size = float(
        config["Column_Size_m"]
    )


    beam_width = float(
        config["Beam_Width_m"]
    )


    beam_depth = float(
        config["Beam_Depth_m"]
    )


    # --------------------------------------------------------
    # DERIVED STRUCTURAL PROPERTIES
    # --------------------------------------------------------

    total_height = (

        num_stories
        *
        story_height

    )


    floor_length_x = (

        num_bays_x
        *
        bay_width_x

    )


    floor_length_y = (

        num_bays_y
        *
        bay_width_y

    )


    floor_area = (

        floor_length_x
        *
        floor_length_y

    )


    column_area = (

        column_size
        *
        column_size

    )


    beam_area = (

        beam_width
        *
        beam_depth

    )


    return {

        "Structure_ID":
            structure_id,


        # ----------------------------------------------------
        # BASIC STRUCTURAL PARAMETERS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # DERIVED PLAN PARAMETERS
        # ----------------------------------------------------

        "Floor_Length_X_m":
            floor_length_x,

        "Floor_Length_Y_m":
            floor_length_y,

        "Floor_Area_m2":
            floor_area,


        # ----------------------------------------------------
        # SECTION PARAMETERS
        # ----------------------------------------------------

        "Column_Size_m":
            column_size,

        "Column_Area_m2":
            column_area,

        "Beam_Width_m":
            beam_width,

        "Beam_Depth_m":
            beam_depth,

        "Beam_Area_m2":
            beam_area

    }


# ============================================================
# CREATE DATASET ROW
# ============================================================

def create_dataset_row(
    structure_features,
    gm_row,
    results
):


    row = {}


    # ========================================================
    # STRUCTURAL FEATURES
    # ========================================================

    row.update(
        structure_features
    )


    # ========================================================
    # GROUND MOTION FEATURES
    # ========================================================

    for column in gm_row.index:


        if column != "GM_ID":

            row[column] = safe_value(

                gm_row[column]

            )


    row["Ground_Motion_ID"] = (
        gm_row["GM_ID"]
    )


    # ========================================================
    # DYNAMIC PROPERTIES
    # ========================================================

    fundamental_period = safe_value(

        results[
            "fundamental_period_s"
        ]

    )


    row["Fundamental_Period_s"] = (
        fundamental_period
    )


    if (

        isinstance(
            fundamental_period,
            (int, float)
        )

        and fundamental_period > 0

    ):

        row["Fundamental_Frequency_Hz"] = (

            1.0
            /
            fundamental_period

        )


    else:

        row["Fundamental_Frequency_Hz"] = (
            np.nan
        )


    row["Damping_Ratio"] = 0.05


    # ========================================================
    # NLTHA RESPONSE TARGETS
    # ========================================================

    row["Max_Roof_Displacement_mm"] = (

        safe_value(

            results[
                "max_roof_displacement_mm"
            ]

        )

    )


    row["Max_Story_Drift"] = (

        safe_value(

            results[
                "max_story_drift"
            ]

        )

    )


    row["Max_Story_Drift_percent"] = (

        safe_value(

            results[
                "max_story_drift_percent"
            ]

        )

    )


    row["Critical_Story"] = (

        safe_value(

            results[
                "critical_story"
            ]

        )

    )


    row[
        "Peak_Roof_Acceleration_m_per_s2"
    ] = (

        safe_value(

            results[
                "max_roof_acceleration_m_per_s2"
            ]

        )

    )


    # ========================================================
    # SIMULATION INFORMATION
    # ========================================================

    row["Simulation_Success"] = bool(

        results[
            "success"
        ]

    )


    row["Simulation_Time_Steps"] = len(

        results["time"]

    )


    return row


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(

    successful_results,
    failed_results,
    completed_cases,
    total_cases

):


    # ========================================================
    # SAVE SUCCESSFUL DATA
    # ========================================================

    if len(successful_results) > 0:


        success_df = pd.DataFrame(

            successful_results

        )


        success_df.to_csv(

            SUCCESS_FILE,

            index=False

        )


    # ========================================================
    # SAVE FAILED DATA
    # ========================================================

    if len(failed_results) > 0:


        failed_df = pd.DataFrame(

            failed_results

        )


        failed_df.to_csv(

            FAILURE_FILE,

            index=False

        )


    # ========================================================
    # SAVE PROGRESS
    # ========================================================

    completion_percent = (

        completed_cases
        /
        total_cases
        *
        100.0

    )


    progress_df = pd.DataFrame([{

        "Completed_Cases":
            completed_cases,

        "Total_Cases":
            total_cases,

        "Successful_Cases":
            len(successful_results),

        "Failed_Cases":
            len(failed_results),

        "Completion_Percent":
            completion_percent

    }])


    progress_df.to_csv(

        PROGRESS_FILE,

        index=False

    )


# ============================================================
# MAIN DATASET GENERATION
# ============================================================

def generate_dataset():


    print("\n")

    print("=" * 80)

    print(
        "FULL BATCH NLTHA DATASET GENERATION"
    )

    print("=" * 80)


    # ========================================================
    # LOAD GROUND MOTION FEATURES
    # ========================================================

    gm_features_df = (

        load_ground_motion_features()

    )


    # ========================================================
    # CREATE STRUCTURE IDS
    # ========================================================

    structure_ids = [

        f"S{i:03d}"

        for i in range(

            1,
            NUM_STRUCTURES + 1

        )

    ]


    gm_ids = gm_features_df[
        "GM_ID"
    ].tolist()


    # ========================================================
    # TOTAL CASES
    # ========================================================

    total_cases = (

        len(structure_ids)
        *
        len(gm_ids)

    )


    print(
        f"\nNumber of Structures : "
        f"{len(structure_ids)}"
    )

    print(
        f"Ground Motions       : "
        f"{len(gm_ids)}"
    )

    print(
        f"Total Simulations    : "
        f"{total_cases}"
    )


    # ========================================================
    # RESULT STORAGE
    # ========================================================

    successful_results = []

    failed_results = []


    case_number = 0


    batch_start_time = time.time()


    # ========================================================
    # STRUCTURE LOOP
    # ========================================================

    for structure_id in structure_ids:


        print("\n")

        print("=" * 80)

        print(
            f"STRUCTURE: {structure_id}"
        )

        print("=" * 80)


        # ====================================================
        # LOAD STRUCTURE FEATURES
        # ====================================================

        try:


            structure_features = (

                extract_structure_features(

                    structure_id

                )

            )


        except Exception as e:


            print(

                f"\n✗ FAILED TO LOAD "
                f"{structure_id}"

            )


            print(
                f"Error: {e}"
            )


            # Mark all ground motions as failed

            for gm_id in gm_ids:


                case_number += 1


                failed_results.append({

                    "Case":
                        case_number,

                    "Structure_ID":
                        structure_id,

                    "Ground_Motion_ID":
                        gm_id,

                    "Error":
                        f"Structure configuration error: {e}"

                })


            continue


        # ====================================================
        # GROUND MOTION LOOP
        # ====================================================

        for _, gm_row in (

            gm_features_df.iterrows()

        ):


            gm_id = gm_row[
                "GM_ID"
            ]


            case_number += 1


            print(

                f"\n[{case_number}/{total_cases}] "

                f"Running "

                f"{structure_id} + {gm_id}"

            )


            case_start_time = time.time()


            try:


                # =============================================
                # RUN NLTHA
                # =============================================

                results = (

                    run_parameterized_nltha(

                        structure_id=structure_id,

                        gm_id=gm_id,

                        verbose=False

                    )

                )


                # =============================================
                # CHECK SUCCESS
                # =============================================

                if results["success"]:


                    row = (

                        create_dataset_row(

                            structure_features,

                            gm_row,

                            results

                        )

                    )


                    row["Case"] = (

                        case_number

                    )


                    row["Simulation_Runtime_s"] = (

                        time.time()

                        -

                        case_start_time

                    )


                    successful_results.append(

                        row

                    )


                    print(

                        f"✓ SUCCESS | "

                        f"Drift = "

                        f"{row['Max_Story_Drift_percent']:.4f}%"

                        f" | "

                        f"Roof Disp. = "

                        f"{row['Max_Roof_Displacement_mm']:.2f} mm"

                    )


                else:


                    failed_results.append({

                        "Case":
                            case_number,

                        "Structure_ID":
                            structure_id,

                        "Ground_Motion_ID":
                            gm_id,

                        "Error":
                            "NLTHA did not complete successfully"

                    })


                    print(
                        "✗ NLTHA FAILED"
                    )


            except Exception as e:


                failed_results.append({

                    "Case":
                        case_number,

                    "Structure_ID":
                        structure_id,

                    "Ground_Motion_ID":
                        gm_id,

                    "Error":
                        str(e)

                })


                print(
                    f"✗ ERROR: {e}"
                )


            # =================================================
            # SAVE CHECKPOINT
            # =================================================

            save_checkpoint(

                successful_results,

                failed_results,

                case_number,

                total_cases

            )


    # ========================================================
    # FINAL SAVE
    # ========================================================

    save_checkpoint(

        successful_results,

        failed_results,

        case_number,

        total_cases

    )


    # ========================================================
    # TOTAL RUNTIME
    # ========================================================

    total_runtime = (

        time.time()

        -

        batch_start_time

    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")

    print("=" * 80)

    print(
        "BATCH DATASET GENERATION COMPLETED"
    )

    print("=" * 80)


    print(
        f"\nTotal Cases       : "
        f"{total_cases}"
    )

    print(
        f"Successful Cases  : "
        f"{len(successful_results)}"
    )

    print(
        f"Failed Cases      : "
        f"{len(failed_results)}"
    )


    success_rate = (

        len(successful_results)

        /

        total_cases

        *

        100.0

    )


    print(
        f"Success Rate      : "
        f"{success_rate:.2f}%"
    )


    print(
        f"Total Runtime     : "
        f"{total_runtime / 60.0:.2f} minutes"
    )


    print("\n")

    print(
        f"Dataset File:\n"
        f"{SUCCESS_FILE}"
    )


    print(
        f"\nFailure Log:\n"
        f"{FAILURE_FILE}"
    )


    print(
        f"\nProgress File:\n"
        f"{PROGRESS_FILE}"
    )


    return (

        pd.DataFrame(successful_results),

        pd.DataFrame(failed_results)

    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":


    dataset_df, failed_df = (

        generate_dataset()

    )