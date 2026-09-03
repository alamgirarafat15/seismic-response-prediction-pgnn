from pathlib import Path
from PIL import Image

folder = Path("PGNN_XAI")

# Max Roof Displacement
img1 = Image.open(folder / "SHAP_Roof_Displacement_Beeswarm.png")
img1.convert("RGB").save(
    folder / "SHAP_Roof_Displacement_Beeswarm.pdf",
    "PDF",
    resolution=300.0
)

# Max Story Drift
img2 = Image.open(folder / "SHAP_Story_Drift_Beeswarm.png")
img2.convert("RGB").save(
    folder / "SHAP_Story_Drift_Beeswarm.pdf",
    "PDF",
    resolution=300.0
)

print("Both SHAP PDF files created successfully!")