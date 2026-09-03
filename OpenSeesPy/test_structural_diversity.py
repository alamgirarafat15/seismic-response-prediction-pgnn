import pandas as pd

from run_parameterized_nltha import (
    run_parameterized_nltha
)


# ============================================================
# STRUCTURAL DIVERSITY VALIDATION
# ============================================================

STRUCTURES = [
    "S001",
    "S010",
    "S020"
]


GROUND_MOTIONS = [
    {
        "gm_id": "GM001",
        "description": "Low PGA / Low Frequency"
    },
    {
        "gm_id": "GM016",
        "description": "High PGA / Low Frequency"
    }
]


# ============================================================
# RUN VALIDATION
# ============================================================

def run_structural_diversity_validation():

    all_results = []

    total_cases = (
        len(STRUCTURES)
        *
        len(GROUND_MOTIONS)
    )

    case_number = 0


    print("\n" + "=" * 80)

    print(
        "STRUCTURAL DIVERSITY VALIDATION"
    )

    print("=" * 80)

    print(
        f"\nStructures Tested    : "
        f"{len(STRUCTURES)}"
    )

    print(
        f"Ground Motions Tested: "
        f"{len(GROUND_MOTIONS)}"
    )

    print(
        f"Total Cases          : "
        f"{total_cases}"
    )


    # ========================================================
    # LOOP THROUGH STRUCTURES
    # ========================================================

    for structure_id in STRUCTURES:

        for gm_info in GROUND_MOTIONS:

            case_number += 1

            gm_id = gm_info["gm_id"]

            description = gm_info[
                "description"
            ]


            print("\n")

            print("=" * 80)

            print(
                f"RUNNING CASE "
                f"{case_number}/{total_cases}"
            )

            print("=" * 80)

            print(
                f"\nStructure      : "
                f"{structure_id}"
            )

            print(
                f"Ground Motion  : "
                f"{gm_id}"
            )

            print(
                f"Description    : "
                f"{description}"
            )


            # =================================================
            # RUN NLTHA
            # =================================================

            try:

                results = run_parameterized_nltha(

                    structure_id=structure_id,

                    gm_id=gm_id,

                    verbose=False

                )


                model_info = results[
                    "model_info"
                ]


                # =============================================
                # STORE RESULTS
                # =============================================

                result_row = {

                    "Case":
                        case_number,

                    "Structure_ID":
                        structure_id,

                    "Ground_Motion":
                        gm_id,

                    "Description":
                        description,

                    "Success":
                        results["success"],

                    "Stories":
                        model_info["num_stories"],

                    "Story_Height_m":
                        model_info["story_height"],

                    "Total_Height_m":
                        model_info[
                            "num_stories"
                        ]
                        *
                        model_info[
                            "story_height"
                        ],

                    "T1_s":
                        results[
                            "fundamental_period_s"
                        ],

                    "Max_Roof_Displacement_mm":
                        results[
                            "max_roof_displacement_mm"
                        ],

                    "Max_Story_Drift_percent":
                        results[
                            "max_story_drift_percent"
                        ],

                    "Critical_Story":
                        results[
                            "critical_story"
                        ],

                    "Peak_Roof_Acceleration_m_per_s2":
                        results[
                            "max_roof_acceleration_m_per_s2"
                        ]

                }


                all_results.append(
                    result_row
                )


                print(
                    "\n✓ CASE COMPLETED"
                )

                print(
                    f"T1                 : "
                    f"{result_row['T1_s']:.4f} s"
                )

                print(
                    f"Roof Displacement  : "
                    f"{result_row['Max_Roof_Displacement_mm']:.2f} mm"
                )

                print(
                    f"Max Story Drift    : "
                    f"{result_row['Max_Story_Drift_percent']:.4f}%"
                )

                print(
                    f"Critical Story     : "
                    f"{result_row['Critical_Story']}"
                )


            except Exception as e:

                print(
                    "\n✗ CASE FAILED"
                )

                print(
                    f"Error: {e}"
                )


                all_results.append({

                    "Case":
                        case_number,

                    "Structure_ID":
                        structure_id,

                    "Ground_Motion":
                        gm_id,

                    "Description":
                        description,

                    "Success":
                        False

                })


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n")

    print("=" * 100)

    print(
        "STRUCTURAL DIVERSITY VALIDATION SUMMARY"
    )

    print("=" * 100)

    print(
        results_df.to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = (
        "structural_diversity_validation_results.csv"
    )

    results_df.to_csv(

        output_file,

        index=False

    )


    successful_cases = int(

        results_df[
            "Success"
        ].sum()

    )


    print("\n")

    print("=" * 80)

    print(
        f"Successful Cases : "
        f"{successful_cases}/{total_cases}"
    )

    print(
        f"Results Saved    : "
        f"{output_file}"
    )

    print("=" * 80)


    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = (
        run_structural_diversity_validation()
    )


    print("\n")

    print("=" * 80)

    print(
        "✓ STRUCTURAL DIVERSITY "
        "VALIDATION COMPLETED"
    )

    print("=" * 80)