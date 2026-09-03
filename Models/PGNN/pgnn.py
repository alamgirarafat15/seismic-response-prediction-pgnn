# ============================================================
# PHYSICS-GUIDED NEURAL NETWORK (PGNN)
# Seismic Response Prediction
#
# Targets:
#   1. Max Story Drift
#   2. Max Roof Displacement
# ============================================================

import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
# 2. FILE PATH
# ============================================================

# IMPORTANT:
# Keep nltha_dataset.csv in the SAME folder as pgnn.py

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FILE_PATH = os.path.join(
    SCRIPT_DIR,
    "nltha_dataset.csv"
)


# ============================================================
# 3. CHECK FILE
# ============================================================

print("=" * 70)
print("PGNN - SEISMIC RESPONSE PREDICTION")
print("=" * 70)

print("\nPython script location:")
print(SCRIPT_DIR)

print("\nLooking for dataset:")
print(FILE_PATH)


if not os.path.exists(FILE_PATH):

    print("\n" + "!" * 70)
    print("ERROR: DATASET NOT FOUND")
    print("!" * 70)

    print("\nPlease put this file:")
    print("nltha_dataset.csv")

    print("\ninside this folder:")
    print(SCRIPT_DIR)

    print("\nExpected structure:")
    print(
        SCRIPT_DIR
        + "\\pgnn.py"
    )

    print(
        SCRIPT_DIR
        + "\\nltha_dataset.csv"
    )

    raise FileNotFoundError(
        f"\nDataset not found:\n{FILE_PATH}"
    )


print("\nDataset found successfully!")


# ============================================================
# 4. LOAD DATASET
# ============================================================

df = pd.read_csv(FILE_PATH)


print("\nDataset shape:")
print(df.shape)


# ============================================================
# 5. TARGETS
# ============================================================

TARGETS = [
    "Max_Story_Drift",
    "Max_Roof_Displacement_mm"
]


print("\n" + "=" * 70)
print("TARGET VARIABLES")
print("=" * 70)

for target in TARGETS:
    print("-", target)


# Check targets exist

for target in TARGETS:

    if target not in df.columns:

        raise ValueError(
            f"Target column not found: {target}"
        )


# ============================================================
# 6. REMOVE COLUMNS
# ============================================================

DROP_COLUMNS = [

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    "Structure_ID",
    "Ground_Motion_ID",

    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    "Max_Story_Drift",
    "Max_Roof_Displacement_mm",

    # Derived target
    "Max_Story_Drift_percent",

    # --------------------------------------------------------
    # RESPONSE-DERIVED
    # --------------------------------------------------------

    "Critical_Story",
    "Peak_Roof_Acceleration_m_per_s2",

    # --------------------------------------------------------
    # SIMULATION INFORMATION
    # --------------------------------------------------------

    "Simulation_Success",
    "Simulation_Time_Steps",
    "Simulation_Runtime_s",

    # Case identifier
    "Case"
]


# ============================================================
# 7. SELECT NUMERICAL FEATURES
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
    "Number of input features:",
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
# 8. CREATE CLEAN DATASET
# ============================================================

data = df[
    FEATURES + TARGETS
].copy()


# Replace infinite values

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


# Remove missing values

data = data.dropna(
    axis=0
).reset_index(
    drop=True
)


print("\n" + "=" * 70)
print("CLEAN DATASET")
print("=" * 70)

print(
    "Rows after cleaning:",
    len(data)
)

print(
    "Columns:",
    data.shape[1]
)


# ============================================================
# 9. X AND Y
# ============================================================

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


print("\nX shape:", X.shape)

print("Y shape:", Y.shape)


# ============================================================
# 10. TRAIN / VALIDATION / TEST SPLIT
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
    "Training samples   :",
    len(train_idx)
)

print(
    "Validation samples :",
    len(val_idx)
)

print(
    "Test samples       :",
    len(test_idx)
)


# ============================================================
# 11. FEATURE SCALING
# ============================================================

X_SCALER = StandardScaler()

Y_SCALER = StandardScaler()


X_train = X_SCALER.fit_transform(
    X[train_idx]
)

X_val = X_SCALER.transform(
    X[val_idx]
)

X_test = X_SCALER.transform(
    X[test_idx]
)


Y_train = Y_SCALER.fit_transform(
    Y[train_idx]
)

Y_val = Y_SCALER.transform(
    Y[val_idx]
)

Y_test = Y_SCALER.transform(
    Y[test_idx]
)


# ============================================================
# 12. PHYSICS VARIABLES
# ============================================================

T1 = data[
    "Fundamental_Period_s"
].values.astype(
    np.float32
)


DAMPING = data[
    "Damping_Ratio"
].values.astype(
    np.float32
)


PGA = data[
    "PGA_g"
].values.astype(
    np.float32
)


GROUND_FREQ = data[
    "Dominant_Frequency_Hz"
].values.astype(
    np.float32
)


TOTAL_HEIGHT = data[
    "Total_Height_m"
].values.astype(
    np.float32
)


# ============================================================
# 13. NATURAL FREQUENCY
#
# f_n = 1 / T
# ============================================================

NATURAL_FREQ = (

    1.0
    /
    np.maximum(
        T1,
        1e-3
    )

)


# ============================================================
# 14. NATURAL CIRCULAR FREQUENCY
#
# omega_n = 2*pi/T
# ============================================================

OMEGA = (

    2.0
    *
    np.pi
    /
    np.maximum(
        T1,
        1e-3
    )

)


# ============================================================
# 15. FREQUENCY RATIO
#
# r = f_g / f_n
# ============================================================

FREQUENCY_RATIO = (

    GROUND_FREQ
    /
    np.maximum(
        NATURAL_FREQ,
        1e-3
    )

)


# ============================================================
# 16. RESONANCE FACTOR
# ============================================================

RESONANCE_FACTOR = np.exp(

    -0.5
    *
    (
        np.log(
            np.maximum(
                FREQUENCY_RATIO,
                1e-4
            )
        )
        /
        0.55
    )
    ** 2

)


# ============================================================
# 17. DYNAMIC AMPLIFICATION FACTOR
#
# DAF =
#
# 1 /
# sqrt[
#       (1-r²)²
#       +
#       (2*zeta*r)²
# ]
# ============================================================

DYNAMIC_FACTOR = (

    1.0
    /
    np.sqrt(

        np.maximum(

            1e-4,

            (
                1.0
                -
                FREQUENCY_RATIO ** 2
            )
            ** 2

            +

            (
                2.0
                *
                DAMPING
                *
                FREQUENCY_RATIO
            )
            ** 2

        )

    )

)


# ============================================================
# 18. APPROXIMATE SPECTRAL ACCELERATION
# ============================================================

SA_PROXY = (

    PGA
    *
    RESONANCE_FACTOR
    *
    DYNAMIC_FACTOR

)


# ============================================================
# 19. PHYSICS-BASED DISPLACEMENT
#
# u = a / omega²
#
# PGA is converted from g to m/s².
# ============================================================

D_PHYS_M = (

    SA_PROXY
    *
    9.80665
    /
    (
        OMEGA ** 2
    )

)


# Convert meter → millimeter

D_PHYS_MM = (

    D_PHYS_M
    *
    1000.0

)


# ============================================================
# 20. PHYSICS TARGET
# ============================================================

PHYSICS_TARGET = np.column_stack(

    [

        # Roof displacement proxy
        D_PHYS_MM,

        # Deformation / drift proxy
        D_PHYS_MM
        /
        np.maximum(
            TOTAL_HEIGHT,
            1e-3
        )

    ]

).astype(
    np.float32
)


# ============================================================
# 21. SCALE PHYSICS TARGET
# ============================================================

PHYSICS_SCALER = StandardScaler()


PHYSICS_TRAIN = (
    PHYSICS_SCALER
    .fit_transform(
        PHYSICS_TARGET[
            train_idx
        ]
    )
)


PHYSICS_VAL = (
    PHYSICS_SCALER
    .transform(
        PHYSICS_TARGET[
            val_idx
        ]
    )
)


# ============================================================
# 22. PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

Y_train_tensor = torch.tensor(
    Y_train,
    dtype=torch.float32
)

P_train_tensor = torch.tensor(
    PHYSICS_TRAIN,
    dtype=torch.float32
)


X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

Y_val_tensor = torch.tensor(
    Y_val,
    dtype=torch.float32
)

P_val_tensor = torch.tensor(
    PHYSICS_VAL,
    dtype=torch.float32
)


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)


# ============================================================
# 23. DEVICE
# ============================================================

DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


print("\n" + "=" * 70)
print("COMPUTING DEVICE")
print("=" * 70)

print(DEVICE)


# ============================================================
# 24. PGNN MODEL
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
# 25. CREATE MODEL
# ============================================================

model = PGNN(
    input_features=len(FEATURES)
).to(DEVICE)


print("\n" + "=" * 70)
print("MODEL ARCHITECTURE")
print("=" * 70)

print(model)


# ============================================================
# 26. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001,

    weight_decay=1e-5

)


# ============================================================
# 27. LOSS FUNCTION
# ============================================================

MSE = nn.MSELoss()


# Physics loss weight

LAMBDA_PHYSICS = 0.08


# ============================================================
# 28. DATALOADER
# ============================================================

train_dataset = TensorDataset(

    X_train_tensor,

    Y_train_tensor,

    P_train_tensor

)


train_loader = DataLoader(

    train_dataset,

    batch_size=64,

    shuffle=True

)


# ============================================================
# 29. TRAINING SETTINGS
# ============================================================

EPOCHS = 1500

PATIENCE = 80


best_val_loss = np.inf

best_model_state = None

early_stop_counter = 0


train_losses = []

val_losses = []


# ============================================================
# 30. TRAIN PGNN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING PGNN")
print("=" * 70)


for epoch in range(EPOCHS):


    # ========================================================
    # TRAINING
    # ========================================================

    model.train()


    running_loss = 0.0


    for X_batch, Y_batch, P_batch in train_loader:


        X_batch = X_batch.to(
            DEVICE
        )

        Y_batch = Y_batch.to(
            DEVICE
        )

        P_batch = P_batch.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Neural network prediction
        # ----------------------------------------------------

        prediction = model(
            X_batch
        )


        # ----------------------------------------------------
        # DATA LOSS
        # ----------------------------------------------------

        data_loss = MSE(

            prediction,

            Y_batch

        )


        # ----------------------------------------------------
        # PHYSICS LOSS
        # ----------------------------------------------------

        physics_loss = MSE(

            prediction,

            P_batch

        )


        # ----------------------------------------------------
        # TOTAL PGNN LOSS
        #
        # L = L_data + lambda * L_physics
        # ----------------------------------------------------

        total_loss = (

            data_loss

            +

            LAMBDA_PHYSICS
            *
            physics_loss

        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        optimizer.zero_grad()

        total_loss.backward()


        # Gradient clipping

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=5.0

        )


        optimizer.step()


        running_loss += (
            total_loss.item()
        )


    train_loss = (

        running_loss
        /
        len(train_loader)

    )


    train_losses.append(
        train_loss
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    with torch.no_grad():

        Xv = X_val_tensor.to(
            DEVICE
        )

        Yv = Y_val_tensor.to(
            DEVICE
        )

        Pv = P_val_tensor.to(
            DEVICE
        )


        val_prediction = model(
            Xv
        )


        val_data_loss = MSE(

            val_prediction,

            Yv

        )


        val_physics_loss = MSE(

            val_prediction,

            Pv

        )


        val_total_loss = (

            val_data_loss

            +

            LAMBDA_PHYSICS
            *
            val_physics_loss

        )


        val_loss = (
            val_total_loss.item()
        )


    val_losses.append(
        val_loss
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        best_model_state = {

            key:
            value.detach()
            .cpu()
            .clone()

            for key, value
            in model.state_dict().items()

        }


        early_stop_counter = 0


    else:

        early_stop_counter += 1


    # ========================================================
    # PRINT TRAINING PROGRESS
    # ========================================================

    if (epoch + 1) % 100 == 0:

        print(

            f"Epoch "
            f"{epoch + 1:4d}/{EPOCHS} | "

            f"Train Loss: "
            f"{train_loss:.6f} | "

            f"Val Loss: "
            f"{val_loss:.6f}"

        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        early_stop_counter
        >= PATIENCE
    ):

        print(

            f"\nEarly stopping at "
            f"epoch {epoch + 1}"

        )

        break


# ============================================================
# 31. LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_state
)

model.eval()


# ============================================================
# 32. TEST PREDICTION
# ============================================================

with torch.no_grad():

    prediction_scaled = model(

        X_test_tensor.to(
            DEVICE
        )

    ).cpu().numpy()


# Convert back to original units

prediction = (
    Y_SCALER
    .inverse_transform(
        prediction_scaled
    )
)


actual = (
    Y_SCALER
    .inverse_transform(
        Y_test
    )
)


# ============================================================
# 33. PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("PGNN TEST RESULTS")
print("=" * 70)


results = []


for i, target in enumerate(
    TARGETS
):


    y_true = actual[:, i]

    y_pred = prediction[:, i]


    r2 = r2_score(

        y_true,

        y_pred

    )


    mae = mean_absolute_error(

        y_true,

        y_pred

    )


    rmse = np.sqrt(

        mean_squared_error(

            y_true,

            y_pred

        )

    )


    results.append({

        "Target": target,

        "R2": r2,

        "MAE": mae,

        "RMSE": rmse

    })


    print("\n" + target)

    print(
        f"R²   = {r2:.4f}"
    )

    print(
        f"MAE  = {mae:.6f}"
    )

    print(
        f"RMSE = {rmse:.6f}"
    )


# ============================================================
# 34. SAVE METRICS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(

    os.path.join(
        SCRIPT_DIR,
        "PGNN_metrics.csv"
    ),

    index=False

)


# ============================================================
# 35. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual_Max_Story_Drift":
        actual[:, 0],

    "Predicted_Max_Story_Drift":
        prediction[:, 0],

    "Actual_Max_Roof_Displacement_mm":
        actual[:, 1],

    "Predicted_Max_Roof_Displacement_mm":
        prediction[:, 1]

})


prediction_df.to_csv(

    os.path.join(
        SCRIPT_DIR,
        "PGNN_test_predictions.csv"
    ),

    index=False

)


# ============================================================
# 36. SAVE MODEL
# ============================================================

torch.save(

    model.state_dict(),

    os.path.join(
        SCRIPT_DIR,
        "PGNN_model.pth"
    )

)


# ============================================================
# 37. LOSS CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    train_losses,

    label="Training Loss"

)


plt.plot(

    val_losses,

    label="Validation Loss"

)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Total PGNN Loss"
)

plt.title(
    "PGNN Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 38. ACTUAL VS PREDICTED
# ============================================================

for i, target in enumerate(
    TARGETS
):


    plt.figure(
        figsize=(6, 6)
    )


    plt.scatter(

        actual[:, i],

        prediction[:, i],

        alpha=0.7

    )


    minimum = min(

        actual[:, i].min(),

        prediction[:, i].min()

    )


    maximum = max(

        actual[:, i].max(),

        prediction[:, i].max()

    )


    plt.plot(

        [minimum, maximum],

        [minimum, maximum],

        "--"

    )


    plt.xlabel(
        f"Actual {target}"
    )

    plt.ylabel(
        f"Predicted {target}"
    )

    plt.title(

        f"PGNN: Actual vs Predicted\n"
        f"{target}"

    )


    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# 39. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("PGNN TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles saved in:")

print(SCRIPT_DIR)

print("\nGenerated files:")

print("1. PGNN_metrics.csv")

print("2. PGNN_test_predictions.csv")

print("3. PGNN_model.pth")

print("\nFinal information:")

print(
    "Features:",
    len(FEATURES)
)

print(
    "Train:",
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

print(
    "Physics weight:",
    LAMBDA_PHYSICS
)
# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

np.save(
    os.path.join(
        SCRIPT_DIR,
        "PGNN_train_losses.npy"
    ),
    np.array(train_losses)
)

np.save(
    os.path.join(
        SCRIPT_DIR,
        "PGNN_val_losses.npy"
    ),
    np.array(val_losses)
)

print("\nTraining history saved.")