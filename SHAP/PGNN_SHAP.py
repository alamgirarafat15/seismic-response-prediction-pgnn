# ============================================================
# PGNN SHAP EXPLAINABLE AI ANALYSIS
#
# Model:
#     Physics-Guided Neural Network (PGNN)
#
# Targets:
#     1. Max_Story_Drift
#     2. Max_Roof_Displacement_mm
#
# Dataset:
#     nltha_dataset.csv
#
# Model:
#     PGNN_model.pth
#
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_FILE = os.path.join(
    SCRIPT_DIR,
    "nltha_dataset.csv"
)

MODEL_FILE = os.path.join(
    SCRIPT_DIR,
    "PGNN_model.pth"
)

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "PGNN_XAI"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. CHECK FILES
# ============================================================

print("=" * 70)
print("PGNN - EXPLAINABLE AI / SHAP ANALYSIS")
print("=" * 70)


if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}"
    )


if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"\nPGNN model not found:\n{MODEL_FILE}"
    )


print("\nDataset:")
print(DATA_FILE)

print("\nModel:")
print(MODEL_FILE)


# ============================================================
# 4. LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE
)


print("\nDataset shape:")
print(df.shape)


# ============================================================
# 5. TARGETS
# ============================================================

TARGETS = [

    "Max_Story_Drift",

    "Max_Roof_Displacement_mm"

]


# ============================================================
# 6. REMOVE NON-INPUT COLUMNS
# ============================================================

DROP_COLUMNS = [

    "Structure_ID",
    "Ground_Motion_ID",

    "Max_Story_Drift",
    "Max_Roof_Displacement_mm",

    "Max_Story_Drift_percent",

    "Critical_Story",
    "Peak_Roof_Acceleration_m_per_s2",

    "Simulation_Success",
    "Simulation_Time_Steps",
    "Simulation_Runtime_s",

    "Case"

]


# ============================================================
# 7. IDENTIFY FEATURES
# ============================================================

FEATURES = []


for column in df.columns:

    if column in DROP_COLUMNS:
        continue

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):

        FEATURES.append(column)


print("\n" + "=" * 70)
print("INPUT FEATURES")
print("=" * 70)

print(
    "Number of features:",
    len(FEATURES)
)


for i, feature in enumerate(
    FEATURES,
    1
):

    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# 8. CLEAN DATA
# ============================================================

data = df[
    FEATURES + TARGETS
].copy()


data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


data = data.dropna(
    axis=0
).reset_index(
    drop=True
)


X = data[
    FEATURES
].values.astype(
    np.float32
)


Y = data[
    TARGETS
].values.astype(
    np.float32
)


print("\nClean dataset:")
print(
    "Rows:",
    len(data)
)

print(
    "X shape:",
    X.shape
)

print(
    "Y shape:",
    Y.shape
)


# ============================================================
# 9. SAME TRAIN / VALIDATION / TEST SPLIT
# ============================================================

indices = np.arange(
    len(data)
)


train_idx, temp_idx = train_test_split(

    indices,

    test_size=0.30,

    random_state=SEED

)


val_idx, test_idx = train_test_split(

    temp_idx,

    test_size=0.50,

    random_state=SEED

)


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    "Training:",
    len(train_idx)
)

print(
    "Validation:",
    len(val_idx)
)

print(
    "Test:",
    len(test_idx)
)


# ============================================================
# 10. SAME INPUT SCALING AS PGNN
# ============================================================

X_scaler = StandardScaler()


X_train = X_scaler.fit_transform(
    X[train_idx]
).astype(
    np.float32
)


X_test = X_scaler.transform(
    X[test_idx]
).astype(
    np.float32
)


# ============================================================
# 11. DEVICE
# ============================================================

DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


print("\nDevice:")
print(DEVICE)


# ============================================================
# 12. PGNN ARCHITECTURE
# ============================================================

class PGNN(nn.Module):

    def __init__(
        self,
        input_features
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                input_features,
                128
            ),

            nn.Tanh(),


            nn.Linear(
                128,
                128
            ),

            nn.Tanh(),


            nn.Linear(
                128,
                64
            ),

            nn.Tanh(),


            nn.Linear(
                64,
                2
            )

        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# 13. LOAD TRAINED MODEL
# ============================================================

model = PGNN(
    len(FEATURES)
).to(
    DEVICE
)


state_dict = torch.load(

    MODEL_FILE,

    map_location=DEVICE

)


model.load_state_dict(
    state_dict
)


model.eval()


print("\n" + "=" * 70)
print("PGNN MODEL LOADED")
print("=" * 70)

print(model)


# ============================================================
# 14. BACKGROUND DATA
# ============================================================
#
# SHAP needs a background/reference dataset.
#
# We use a subset of TRAINING data only.
#
# Test data are used only for explanation.
# ============================================================

BACKGROUND_SIZE = min(
    100,
    len(X_train)
)


background_indices = np.linspace(

    0,

    len(X_train) - 1,

    BACKGROUND_SIZE

).astype(int)


background = X_train[
    background_indices
]


# ============================================================
# 15. TEST DATA FOR SHAP
# ============================================================
#
# To keep computation reasonable, use up to 150 test samples.
# Your current test set has 150 samples.
# ============================================================

MAX_SHAP_SAMPLES = min(

    150,

    len(X_test)

)


shap_indices = np.arange(
    MAX_SHAP_SAMPLES
)


X_explain = X_test[
    shap_indices
]


print("\n" + "=" * 70)
print("SHAP DATA")
print("=" * 70)

print(
    "Background samples:",
    len(background)
)

print(
    "Explained test samples:",
    len(X_explain)
)


# ============================================================
# 16. PREDICTION FUNCTION
# ============================================================

def model_predict(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )


    tensor = torch.tensor(
        x,
        dtype=torch.float32,
        device=DEVICE
    )


    with torch.no_grad():

        output = model(
            tensor
        )


    return output.cpu().numpy()


# ============================================================
# 17. CREATE SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 70)
print("CREATING SHAP EXPLAINER")
print("=" * 70)


# Use GradientExplainer for PyTorch neural network

background_tensor = torch.tensor(

    background,

    dtype=torch.float32,

    device=DEVICE

)


explainer = shap.GradientExplainer(

    model,

    background_tensor

)


print(
    "SHAP GradientExplainer created."
)


# ============================================================
# 18. CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

print(
    "This may take some time."
)


X_explain_tensor = torch.tensor(

    X_explain,

    dtype=torch.float32,

    device=DEVICE

)


shap_values = explainer.shap_values(

    X_explain_tensor

)


# ============================================================
# 19. HANDLE SHAP OUTPUT FORMAT
# ============================================================

#
# Depending on SHAP version, multi-output results may be:
#
#   list of arrays
#
# or:
#
#   numpy array with shape
#   (samples, features, outputs)
#
# We convert everything into:
#
#   drift_shap
#   roof_shap
#
# ============================================================


if isinstance(
    shap_values,
    list
):

    drift_shap = np.asarray(
        shap_values[0]
    )

    roof_shap = np.asarray(
        shap_values[1]
    )

else:

    shap_values = np.asarray(
        shap_values
    )


    if shap_values.ndim == 3:

        # samples x features x outputs

        if shap_values.shape[2] == 2:

            drift_shap = shap_values[
                :,
                :,
                0
            ]

            roof_shap = shap_values[
                :,
                :,
                1
            ]

        # outputs x samples x features

        elif shap_values.shape[0] == 2:

            drift_shap = shap_values[
                0
            ]

            roof_shap = shap_values[
                1
            ]

        else:

            raise ValueError(
                "Unexpected SHAP output shape."
            )

    else:

        raise ValueError(
            "Unexpected SHAP output format."
        )


# ============================================================
# 20. CHECK SHAP SHAPES
# ============================================================

print("\nSHAP shapes:")

print(
    "Story Drift:",
    drift_shap.shape
)

print(
    "Roof Displacement:",
    roof_shap.shape
)


# ============================================================
# 21. FEATURE IMPORTANCE
# ============================================================

drift_importance = np.mean(

    np.abs(
        drift_shap
    ),

    axis=0

)


roof_importance = np.mean(

    np.abs(
        roof_shap
    ),

    axis=0

)


# ============================================================
# 22. IMPORTANCE TABLES
# ============================================================

drift_importance_df = pd.DataFrame({

    "Feature":
        FEATURES,

    "Mean_Absolute_SHAP":
        drift_importance

})


drift_importance_df = (

    drift_importance_df

    .sort_values(

        "Mean_Absolute_SHAP",

        ascending=False

    )

    .reset_index(
        drop=True
    )

)


drift_importance_df[
    "Rank"
] = np.arange(

    1,

    len(drift_importance_df) + 1

)


roof_importance_df = pd.DataFrame({

    "Feature":
        FEATURES,

    "Mean_Absolute_SHAP":
        roof_importance

})


roof_importance_df = (

    roof_importance_df

    .sort_values(

        "Mean_Absolute_SHAP",

        ascending=False

    )

    .reset_index(
        drop=True
    )

)


roof_importance_df[
    "Rank"
] = np.arange(

    1,

    len(roof_importance_df) + 1

)


# ============================================================
# 23. SAVE IMPORTANCE CSV
# ============================================================

drift_importance_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "SHAP_Story_Drift_Importance.csv"

    ),

    index=False

)


roof_importance_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "SHAP_Roof_Displacement_Importance.csv"

    ),

    index=False

)


# ============================================================
# 24. PRINT TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP FEATURES - MAX STORY DRIFT")
print("=" * 70)


print(

    drift_importance_df.head(
        10
    ).to_string(
        index=False
    )

)


print("\n" + "=" * 70)
print("TOP FEATURES - MAX ROOF DISPLACEMENT")
print("=" * 70)


print(

    roof_importance_df.head(
        10
    ).to_string(
        index=False
    )

)


# ============================================================
# 25. SHAP BAR PLOT - STORY DRIFT
# ============================================================

print("\nGenerating Story Drift SHAP bar plot...")


plt.figure(
    figsize=(9, 7)
)


top_drift = (
    drift_importance_df
    .head(15)
    .sort_values(
        "Mean_Absolute_SHAP"
    )
)


plt.barh(

    top_drift["Feature"],

    top_drift[
        "Mean_Absolute_SHAP"
    ]

)


plt.xlabel(
    "Mean |SHAP value|"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "PGNN Feature Importance - Max Story Drift"
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "SHAP_Story_Drift_Bar.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 26. SHAP BAR PLOT - ROOF DISPLACEMENT
# ============================================================

print("\nGenerating Roof Displacement SHAP bar plot...")


plt.figure(
    figsize=(9, 7)
)


top_roof = (
    roof_importance_df
    .head(15)
    .sort_values(
        "Mean_Absolute_SHAP"
    )
)


plt.barh(

    top_roof["Feature"],

    top_roof[
        "Mean_Absolute_SHAP"
    ]

)


plt.xlabel(
    "Mean |SHAP value|"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "PGNN Feature Importance - Max Roof Displacement"
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "SHAP_Roof_Displacement_Bar.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 27. SHAP BEESWARM - STORY DRIFT
# ============================================================

print("\nGenerating Story Drift SHAP beeswarm...")


plt.figure(
    figsize=(10, 8)
)


shap.summary_plot(

    drift_shap,

    X_explain,

    feature_names=FEATURES,

    max_display=15,

    show=False

)


plt.title(
    "SHAP Summary - Max Story Drift"
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "SHAP_Story_Drift_Beeswarm.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 28. SHAP BEESWARM - ROOF DISPLACEMENT
# ============================================================

print("\nGenerating Roof Displacement SHAP beeswarm...")


plt.figure(
    figsize=(10, 8)
)


shap.summary_plot(

    roof_shap,

    X_explain,

    feature_names=FEATURES,

    max_display=15,

    show=False

)


plt.title(
    "SHAP Summary - Max Roof Displacement"
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "SHAP_Roof_Displacement_Beeswarm.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 29. DEPENDENCE PLOTS
# ============================================================
#
# Automatically generate dependence plots for top 5 features.
# ============================================================


TOP_N = 5


top_drift_features = (

    drift_importance_df
    .head(TOP_N)
    ["Feature"]
    .tolist()

)


top_roof_features = (

    roof_importance_df
    .head(TOP_N)
    ["Feature"]
    .tolist()

)


# ------------------------------------------------------------
# Story Drift dependence plots
# ------------------------------------------------------------

for feature in top_drift_features:

    print(
        "Generating Story Drift dependence:",
        feature
    )


    plt.figure(
        figsize=(8, 6)
    )


    shap.dependence_plot(

        feature,

        drift_shap,

        X_explain,

        feature_names=FEATURES,

        show=False

    )


    plt.title(

        f"PGNN SHAP Dependence - "
        f"Max Story Drift\n{feature}"

    )


    plt.tight_layout()


    safe_name = (

        feature
        .replace("/", "_")
        .replace(" ", "_")

    )


    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            f"SHAP_Dependence_Drift_{safe_name}.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )


    plt.show()

    plt.close()


# ------------------------------------------------------------
# Roof displacement dependence plots
# ------------------------------------------------------------

for feature in top_roof_features:

    print(
        "Generating Roof Displacement dependence:",
        feature
    )


    plt.figure(
        figsize=(8, 6)
    )


    shap.dependence_plot(

        feature,

        roof_shap,

        X_explain,

        feature_names=FEATURES,

        show=False

    )


    plt.title(

        f"PGNN SHAP Dependence - "
        f"Max Roof Displacement\n{feature}"

    )


    plt.tight_layout()


    safe_name = (

        feature
        .replace("/", "_")
        .replace(" ", "_")

    )


    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            f"SHAP_Dependence_Roof_{safe_name}.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )


    plt.show()

    plt.close()


# ============================================================
# 30. SAVE RAW SHAP VALUES
# ============================================================

np.save(

    os.path.join(

        OUTPUT_DIR,

        "SHAP_Story_Drift_values.npy"

    ),

    drift_shap

)


np.save(

    os.path.join(

        OUTPUT_DIR,

        "SHAP_Roof_Displacement_values.npy"

    ),

    roof_shap

)


# ============================================================
# 31. SAVE EXPLAINED TEST DATA
# ============================================================

explained_data = pd.DataFrame(

    X_explain,

    columns=FEATURES

)


explained_data.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "SHAP_Explained_Test_Features.csv"

    ),

    index=False

)


# ============================================================
# 32. FINAL
# ============================================================

print("\n" + "=" * 70)
print("PGNN SHAP ANALYSIS COMPLETED")
print("=" * 70)


print("\nOutput folder:")

print(OUTPUT_DIR)


print("\nGenerated files include:")

print(
    "1. SHAP_Story_Drift_Bar.png"
)

print(
    "2. SHAP_Story_Drift_Beeswarm.png"
)

print(
    "3. SHAP_Roof_Displacement_Bar.png"
)

print(
    "4. SHAP_Roof_Displacement_Beeswarm.png"
)

print(
    "5. SHAP_Story_Drift_Importance.csv"
)

print(
    "6. SHAP_Roof_Displacement_Importance.csv"
)

print(
    "7. Top-5 SHAP dependence plots"
)

print(
    "\nDone!"
)