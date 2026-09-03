# ============================================================
# ANN + RANDOM FOREST
# Seismic Response Prediction
#
# Same dataset and same targets as PGNN
#
# Dataset:
#     nltha_dataset.csv
#
# Targets:
#     1. Max_Story_Drift
#     2. Max_Roof_Displacement_mm
#
# Split:
#     70% Train
#     15% Validation
#     15% Test
#
# Random State:
#     42
# ============================================================


import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    TensorDataset
)


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

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FILE_PATH = os.path.join(
    SCRIPT_DIR,
    "nltha_dataset.csv"
)


# ============================================================
# 3. CHECK DATASET
# ============================================================

print("=" * 70)
print("ANN + RANDOM FOREST")
print("SEISMIC RESPONSE PREDICTION")
print("=" * 70)


print("\nLooking for dataset:")

print(FILE_PATH)


if not os.path.exists(FILE_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{FILE_PATH}"
    )


print("\nDataset found successfully!")


# ============================================================
# 4. LOAD DATA
# ============================================================

df = pd.read_csv(
    FILE_PATH
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


print("\n" + "=" * 70)

print("TARGET VARIABLES")

print("=" * 70)


for target in TARGETS:

    print("-", target)


# ============================================================
# 6. DROP COLUMNS
# ============================================================

DROP_COLUMNS = [

    # IDs
    "Structure_ID",
    "Ground_Motion_ID",

    # Targets
    "Max_Story_Drift",
    "Max_Roof_Displacement_mm",

    # Target-derived
    "Max_Story_Drift_percent",

    # Response-derived
    "Critical_Story",
    "Peak_Roof_Acceleration_m_per_s2",

    # Simulation information
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

        FEATURES.append(
            column
        )


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


print("\n" + "=" * 70)

print("CLEAN DATASET")

print("=" * 70)


print(
    "Rows:",
    len(data)
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
# 10. SAME TRAIN / VALIDATION / TEST SPLIT
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
# 11. SCALING FOR ANN
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
# ============================================================
# PART A: ANN / DNN
# ============================================================
# ============================================================


print("\n")

print("=" * 70)

print("PART A: ANN / DNN")

print("=" * 70)


# ============================================================
# 12. PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)


Y_train_tensor = torch.tensor(
    Y_train,
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


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)


# ============================================================
# 13. DEVICE
# ============================================================

DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


print("\nANN device:")

print(DEVICE)


# ============================================================
# 14. ANN MODEL
# ============================================================

class ANN(nn.Module):


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
# 15. CREATE ANN
# ============================================================

ann_model = ANN(
    input_features=len(FEATURES)
).to(DEVICE)


print("\nANN architecture:")

print(ann_model)


# ============================================================
# 16. ANN OPTIMIZER
# ============================================================

ann_optimizer = torch.optim.Adam(

    ann_model.parameters(),

    lr=0.001,

    weight_decay=1e-5

)


# ============================================================
# 17. ANN LOSS
# ============================================================

MSE = nn.MSELoss()


# ============================================================
# 18. ANN DATALOADER
# ============================================================

ann_dataset = TensorDataset(

    X_train_tensor,

    Y_train_tensor

)


ann_loader = DataLoader(

    ann_dataset,

    batch_size=64,

    shuffle=True

)


# ============================================================
# 19. ANN TRAINING
# ============================================================

ANN_EPOCHS = 1500

ANN_PATIENCE = 80


best_ann_val_loss = np.inf

best_ann_state = None

ann_counter = 0


ann_train_losses = []

ann_val_losses = []


print("\n" + "=" * 70)

print("ANN TRAINING")

print("=" * 70)


for epoch in range(
    ANN_EPOCHS
):


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    ann_model.train()


    running_loss = 0.0


    for X_batch, Y_batch in ann_loader:


        X_batch = X_batch.to(
            DEVICE
        )


        Y_batch = Y_batch.to(
            DEVICE
        )


        prediction = ann_model(
            X_batch
        )


        loss = MSE(

            prediction,

            Y_batch

        )


        ann_optimizer.zero_grad()


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            ann_model.parameters(),

            max_norm=5.0

        )


        ann_optimizer.step()


        running_loss += (
            loss.item()
        )


    train_loss = (

        running_loss
        /
        len(ann_loader)

    )


    ann_train_losses.append(
        train_loss
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    ann_model.eval()


    with torch.no_grad():

        validation_prediction = ann_model(

            X_val_tensor.to(
                DEVICE
            )

        )


        val_loss = MSE(

            validation_prediction,

            Y_val_tensor.to(
                DEVICE
            )

        ).item()


    ann_val_losses.append(
        val_loss
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_ann_val_loss:

        best_ann_val_loss = val_loss


        best_ann_state = {

            key:
            value.detach()
            .cpu()
            .clone()

            for key, value
            in ann_model.state_dict().items()

        }


        ann_counter = 0


    else:

        ann_counter += 1


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    if (
        epoch + 1
    ) % 100 == 0:

        print(

            f"Epoch "
            f"{epoch + 1:4d}/{ANN_EPOCHS} | "

            f"Train Loss: "
            f"{train_loss:.6f} | "

            f"Val Loss: "
            f"{val_loss:.6f}"

        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if ann_counter >= ANN_PATIENCE:

        print(
            f"\nANN early stopping at "
            f"epoch {epoch + 1}"
        )

        break


# ============================================================
# 20. LOAD BEST ANN
# ============================================================

ann_model.load_state_dict(
    best_ann_state
)

ann_model.eval()


# ============================================================
# 21. ANN TEST PREDICTION
# ============================================================

with torch.no_grad():

    ann_prediction_scaled = ann_model(

        X_test_tensor.to(
            DEVICE
        )

    ).cpu().numpy()


ann_prediction = (
    Y_SCALER
    .inverse_transform(
        ann_prediction_scaled
    )
)


actual = (
    Y_SCALER
    .inverse_transform(
        Y_test
    )
)


# ============================================================
# 22. ANN METRICS
# ============================================================

ann_results = []


print("\n" + "=" * 70)

print("ANN TEST RESULTS")

print("=" * 70)


for i, target in enumerate(
    TARGETS
):


    y_true = actual[:, i]

    y_pred = ann_prediction[:, i]


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


    ann_results.append({

        "Model": "ANN",

        "Target": target,

        "R2": r2,

        "MAE": mae,

        "RMSE": rmse

    })


    print("\n", target)

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
# 23. SAVE ANN MODEL
# ============================================================

torch.save(

    ann_model.state_dict(),

    os.path.join(

        SCRIPT_DIR,

        "ANN_model.pth"

    )

)


# ============================================================
# 24. SAVE ANN LOSS HISTORY
# ============================================================

np.save(

    os.path.join(

        SCRIPT_DIR,

        "ANN_train_losses.npy"

    ),

    np.array(
        ann_train_losses
    )

)


np.save(

    os.path.join(

        SCRIPT_DIR,

        "ANN_val_losses.npy"

    ),

    np.array(
        ann_val_losses
    )

)


# ============================================================
# ============================================================
# PART B: RANDOM FOREST
# ============================================================
# ============================================================


print("\n")

print("=" * 70)

print("PART B: RANDOM FOREST")

print("=" * 70)


# ============================================================
# 25. RANDOM FOREST
# ============================================================

rf_model = RandomForestRegressor(

    n_estimators=500,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    max_features="sqrt",

    random_state=SEED,

    n_jobs=-1

)


print("\nTraining Random Forest...")


rf_model.fit(

    X[train_idx],

    Y[train_idx]

)


print(
    "Random Forest training completed."
)


# ============================================================
# 26. RF PREDICTION
# ============================================================

rf_prediction = rf_model.predict(

    X[test_idx]

)


# ============================================================
# 27. RF METRICS
# ============================================================

rf_results = []


print("\n" + "=" * 70)

print("RANDOM FOREST TEST RESULTS")

print("=" * 70)


for i, target in enumerate(
    TARGETS
):


    y_true = Y[
        test_idx,
        i
    ]


    y_pred = rf_prediction[
        :,
        i
    ]


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


    rf_results.append({

        "Model": "Random Forest",

        "Target": target,

        "R2": r2,

        "MAE": mae,

        "RMSE": rmse

    })


    print("\n", target)

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
# 28. SAVE RF MODEL
# ============================================================

import joblib


joblib.dump(

    rf_model,

    os.path.join(

        SCRIPT_DIR,

        "RandomForest_model.pkl"

    )

)


# ============================================================
# ============================================================
# PART C: SAVE RESULTS
# ============================================================
# ============================================================


all_results = (

    ann_results

    +

    rf_results

)


results_df = pd.DataFrame(
    all_results
)


results_path = os.path.join(

    SCRIPT_DIR,

    "ANN_RF_metrics.csv"

)


results_df.to_csv(

    results_path,

    index=False

)


# ============================================================
# 29. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual_Max_Story_Drift":
        actual[:, 0],

    "ANN_Max_Story_Drift":
        ann_prediction[:, 0],

    "RF_Max_Story_Drift":
        rf_prediction[:, 0],


    "Actual_Max_Roof_Displacement_mm":
        actual[:, 1],

    "ANN_Max_Roof_Displacement_mm":
        ann_prediction[:, 1],

    "RF_Max_Roof_Displacement_mm":
        rf_prediction[:, 1]

})


prediction_path = os.path.join(

    SCRIPT_DIR,

    "ANN_RF_test_predictions.csv"

)


prediction_df.to_csv(

    prediction_path,

    index=False

)


# ============================================================
# 30. ANN LOSS CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    ann_train_losses,

    label="Training Loss"

)


plt.plot(

    ann_val_losses,

    label="Validation Loss"

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "MSE Loss"
)


plt.title(
    "ANN Training and Validation Loss"
)


plt.legend()

plt.grid(True)

plt.tight_layout()


ann_loss_path = os.path.join(

    SCRIPT_DIR,

    "ANN_Loss_Curve.png"

)


plt.savefig(

    ann_loss_path,

    dpi=300,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 31. ANN STORY DRIFT
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.scatter(

    actual[:, 0],

    ann_prediction[:, 0],

    alpha=0.7

)


minimum = min(

    actual[:, 0].min(),

    ann_prediction[:, 0].min()

)


maximum = max(

    actual[:, 0].max(),

    ann_prediction[:, 0].max()

)


plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    "--",

    linewidth=2,

    label="Ideal Prediction"

)


plt.xlabel(
    "Actual Max Story Drift"
)


plt.ylabel(
    "Predicted Max Story Drift"
)


plt.title(
    "ANN: Actual vs Predicted Max Story Drift"
)


plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    os.path.join(

        SCRIPT_DIR,

        "ANN_Story_Drift_Actual_vs_Predicted.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 32. ANN ROOF DISPLACEMENT
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.scatter(

    actual[:, 1],

    ann_prediction[:, 1],

    alpha=0.7

)


minimum = min(

    actual[:, 1].min(),

    ann_prediction[:, 1].min()

)


maximum = max(

    actual[:, 1].max(),

    ann_prediction[:, 1].max()

)


plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    "--",

    linewidth=2,

    label="Ideal Prediction"

)


plt.xlabel(
    "Actual Max Roof Displacement (mm)"
)


plt.ylabel(
    "Predicted Max Roof Displacement (mm)"
)


plt.title(
    "ANN: Actual vs Predicted Max Roof Displacement"
)


plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    os.path.join(

        SCRIPT_DIR,

        "ANN_Roof_Displacement_Actual_vs_Predicted.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 33. RF STORY DRIFT
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.scatter(

    Y[test_idx, 0],

    rf_prediction[:, 0],

    alpha=0.7

)


minimum = min(

    Y[test_idx, 0].min(),

    rf_prediction[:, 0].min()

)


maximum = max(

    Y[test_idx, 0].max(),

    rf_prediction[:, 0].max()

)


plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    "--",

    linewidth=2,

    label="Ideal Prediction"

)


plt.xlabel(
    "Actual Max Story Drift"
)


plt.ylabel(
    "Predicted Max Story Drift"
)


plt.title(
    "Random Forest: Actual vs Predicted Max Story Drift"
)


plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    os.path.join(

        SCRIPT_DIR,

        "RF_Story_Drift_Actual_vs_Predicted.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 34. RF ROOF DISPLACEMENT
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.scatter(

    Y[test_idx, 1],

    rf_prediction[:, 1],

    alpha=0.7

)


minimum = min(

    Y[test_idx, 1].min(),

    rf_prediction[:, 1].min()

)


maximum = max(

    Y[test_idx, 1].max(),

    rf_prediction[:, 1].max()

)


plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    "--",

    linewidth=2,

    label="Ideal Prediction"

)


plt.xlabel(
    "Actual Max Roof Displacement (mm)"
)


plt.ylabel(
    "Predicted Max Roof Displacement (mm)"
)


plt.title(
    "Random Forest: Actual vs Predicted Max Roof Displacement"
)


plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    os.path.join(

        SCRIPT_DIR,

        "RF_Roof_Displacement_Actual_vs_Predicted.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.show()

plt.close()


# ============================================================
# 35. FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)

print("ANN + RANDOM FOREST COMPLETED")

print("=" * 70)


print("\nResults saved:")

print(
    "1.",
    results_path
)


print(
    "2.",
    prediction_path
)


print(
    "3. ANN_model.pth"
)


print(
    "4. RandomForest_model.pkl"
)


print(
    "5. ANN_Loss_Curve.png"
)


print("\nAll done!")