import pandas as pd

from run_parameterized_nltha import (
    run_parameterized_nltha
)


# ============================================================
# PARAMETERIZED NLTHA VALIDATION CASES
# ============================================================


TEST_CASES = [

    {
        "structure_id": "S001",
        "gm_id": "GM001",
        "description": "Low PGA - Low Frequency"
    },

    {
        "structure_id": "S001",
        "gm_id": "GM005",
        "description": "Low PGA - High Frequency"
    },

    {
        "structure_id": "S001",
        "gm_id": "GM011",
        "description": "Medium PGA - Low Frequency"
    },

    {
        "structure_id": "S001",
        "gm_id": "GM016",
        "description": "High PGA - Low Frequency"
    }

]


# ============================================================
# RUN VALIDATION CASES
# ============================================================


def run_validation_cases():

    all_results = []

    total_cases = len(
        TEST_CASES
    )


    print("\n")

    print("=" * 80)

    print(
        "PARAMETERIZED NLTHA VALIDATION TEST"
    )

    print("=" * 80)

    print(
        f"\nTotal Test Cases : "
        f"{total_cases}"
    )


    # ========================================================
    # RUN EACH CASE
    # ========================================================

    for case_number, case in enumerate(

        TEST_CASES,

        start=1

    ):


        structure_id = case[
            "structure_id"
        ]


        gm_id = case[
            "gm_id"
        ]


        description = case[
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
            f"\nStructure : "
            f"{structure_id}"
        )


        print(
            f"Ground Motion : "
            f"{gm_id}"
        )


        print(
            f"Description : "
            f"{description}"
        )


        # ====================================================
        # RUN NLTHA
        # ====================================================

        try:

            results = run_parameterized_nltha(

                structure_id=structure_id,

                gm_id=gm_id,

                verbose=False

            )


            # =================================================
            # STORE RESULTS
            # =================================================

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
                    ],

                "Number_of_Time_Steps":
                    len(
                        results["time"]
                    )

            }


            all_results.append(
                result_row
            )


            print(
                "\n✓ CASE COMPLETED"
            )


            print(
                f"Max Roof Displacement : "
                f"{results['max_roof_displacement_mm']:.2f} mm"
            )


            print(
                f"Max Story Drift       : "
                f"{results['max_story_drift_percent']:.4f}%"
            )


            print(
                f"Critical Story        : "
                f"{results['critical_story']}"
            )


        except Exception as e:


            print(
                "\n✗ CASE FAILED"
            )


            print(
                f"Error: {e}"
            )


            # ================================================
            # STORE FAILURE
            # ================================================

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
                    False,

                "T1_s":
                    None,

                "Max_Roof_Displacement_mm":
                    None,

                "Max_Story_Drift_percent":
                    None,

                "Critical_Story":
                    None,

                "Peak_Roof_Acceleration_m_per_s2":
                    None,

                "Number_of_Time_Steps":
                    None

            })


    # ========================================================
    # CREATE RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n")

    print("=" * 80)

    print(
        "VALIDATION TEST SUMMARY"
    )

    print("=" * 80)


    print(
        results_df.to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = (
        "parameterized_nltha_validation_results.csv"
    )


    results_df.to_csv(

        output_file,

        index=False

    )


    print("\n")

    print("=" * 80)

    print(
        f"✓ RESULTS SAVED AS: "
        f"{output_file}"
    )

    print("=" * 80)


    # ========================================================
    # SUCCESS STATISTICS
    # ========================================================

    successful_cases = (

        results_df["Success"]
        .sum()

    )


    print(
        f"\nSuccessful Cases : "
        f"{successful_cases}/{total_cases}"
    )


    return results_df


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    validation_results = (
        run_validation_cases()
    )


    print("\n")

    print("=" * 80)

    print(
        "✓ PARAMETERIZED NLTHA "
        "VALIDATION COMPLETED"
    )

    print("=" * 80)