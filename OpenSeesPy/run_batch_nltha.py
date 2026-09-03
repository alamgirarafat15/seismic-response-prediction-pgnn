import os
import pandas as pd
import traceback

from run_nltha import run_ground_motion_analysis


# ============================================================
# BATCH NLTHA SIMULATION
#
# Runs all ground motions on the current RC frame model
# and saves engineering response results.
# ============================================================


GROUND_MOTION_FOLDER = "ground_motions_v2"

OUTPUT_FILE = "pilot_nltha_results.csv"

DAMPING_RATIO = 0.05

DIRECTION = 1


# ============================================================
# FIND GROUND MOTION FILES
# ============================================================

def get_ground_motion_files():

    files = []

    for filename in os.listdir(GROUND_MOTION_FOLDER):

        if filename.endswith(".csv"):

            files.append(filename)

    files.sort()

    return files


# ============================================================
# RUN BATCH ANALYSIS
# ============================================================

def run_batch():

    gm_files = get_ground_motion_files()

    total_motions = len(gm_files)

    print("\n" + "=" * 75)

    print("BATCH NONLINEAR TIME-HISTORY ANALYSIS")

    print("=" * 75)

    print(f"\nGround Motion Folder : {GROUND_MOTION_FOLDER}")

    print(f"Total Ground Motions  : {total_motions}")

    print(f"Damping Ratio         : {DAMPING_RATIO * 100:.1f}%")

    print(f"Analysis Direction    : X-direction")

    print("\n" + "=" * 75)


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    all_results = []


    # --------------------------------------------------------
    # LOOP THROUGH ALL GROUND MOTIONS
    # --------------------------------------------------------

    for index, filename in enumerate(gm_files, start=1):

        gm_id = os.path.splitext(filename)[0]

        gm_path = os.path.join(
            GROUND_MOTION_FOLDER,
            filename
        )


        print("\n")

        print("=" * 75)

        print(
            f"RUNNING {gm_id} "
            f"({index}/{total_motions})"
        )

        print("=" * 75)


        try:

            # =================================================
            # RUN NLTHA
            # =================================================

            results = run_ground_motion_analysis(

                gm_file=gm_path,

                damping_ratio=DAMPING_RATIO,

                direction=DIRECTION

            )


            # =================================================
            # STORE ENGINEERING RESPONSE RESULTS
            # =================================================

            result_row = {

                "GM_ID":
                    gm_id,

                "Analysis_Success":
                    results["analysis_success"],

                "Fundamental_Period_s":
                    results["T1_s"],

                "Maximum_Roof_Displacement_m":
                    results["max_roof_displacement_m"],

                "Maximum_Roof_Displacement_mm":
                    results[
                        "max_roof_displacement_m"
                    ] * 1000,

                "Maximum_Story_Drift_Ratio":
                    results[
                        "max_story_drift_ratio"
                    ],

                "Maximum_Story_Drift_Percent":
                    results[
                        "max_story_drift_ratio"
                    ] * 100,

                "Critical_Story":
                    results[
                        "critical_story"
                    ],

                "Number_of_Time_Steps":
                    results[
                        "num_time_steps"
                    ],

                "Time_Step_s":
                    results[
                        "dt"
                    ]

            }


            all_results.append(
                result_row
            )


            print("\n✓ SIMULATION COMPLETED")

            print(
                f"Max Roof Displacement : "
                f"{result_row['Maximum_Roof_Displacement_mm']:.2f} mm"
            )

            print(
                f"Max Story Drift       : "
                f"{result_row['Maximum_Story_Drift_Percent']:.4f}%"
            )

            print(
                f"Critical Story        : "
                f"{result_row['Critical_Story']}"
            )


        # =====================================================
        # HANDLE ERRORS
        # =====================================================

        except Exception as error:


            print("\n⚠ SIMULATION FAILED")

            print(
                f"Ground Motion : {gm_id}"
            )

            print(
                f"Error : {error}"
            )


            traceback.print_exc()


            # Store failed simulation

            failed_row = {

                "GM_ID":
                    gm_id,

                "Analysis_Success":
                    False,

                "Fundamental_Period_s":
                    None,

                "Maximum_Roof_Displacement_m":
                    None,

                "Maximum_Roof_Displacement_mm":
                    None,

                "Maximum_Story_Drift_Ratio":
                    None,

                "Maximum_Story_Drift_Percent":
                    None,

                "Critical_Story":
                    None,

                "Number_of_Time_Steps":
                    None,

                "Time_Step_s":
                    None

            }


            all_results.append(
                failed_row
            )


    # ============================================================
    # CREATE DATAFRAME
    # ============================================================

    results_df = pd.DataFrame(
        all_results
    )


    # ============================================================
    # SAVE RESULTS
    # ============================================================

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    successful_runs = results_df[
        "Analysis_Success"
    ].sum()

    failed_runs = total_motions - successful_runs


    print("\n")

    print("=" * 75)

    print("BATCH SIMULATION SUMMARY")

    print("=" * 75)


    print(
        f"\nTotal Ground Motions : "
        f"{total_motions}"
    )

    print(
        f"Successful Runs      : "
        f"{successful_runs}"
    )

    print(
        f"Failed Runs          : "
        f"{failed_runs}"
    )


    # ------------------------------------------------------------
    # RESPONSE STATISTICS
    # ------------------------------------------------------------

    successful_df = results_df[
        results_df["Analysis_Success"] == True
    ]


    if len(successful_df) > 0:


        print("\n" + "-" * 75)

        print("RESPONSE RANGE")

        print("-" * 75)


        print(

            f"\nRoof Displacement Range : "

            f"{successful_df['Maximum_Roof_Displacement_mm'].min():.2f}"

            f" to "

            f"{successful_df['Maximum_Roof_Displacement_mm'].max():.2f}"

            f" mm"

        )


        print(

            f"Story Drift Range       : "

            f"{successful_df['Maximum_Story_Drift_Percent'].min():.4f}"

            f"% to "

            f"{successful_df['Maximum_Story_Drift_Percent'].max():.4f}"

            f"%"

        )


        print(

            f"Maximum Drift Observed  : "

            f"{successful_df['Maximum_Story_Drift_Percent'].max():.4f}"

            f"%"

        )


    print("\n" + "=" * 75)

    print(
        f"✓ RESULTS SAVED AS: {OUTPUT_FILE}"
    )

    print("=" * 75)


    return results_df


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    run_batch()