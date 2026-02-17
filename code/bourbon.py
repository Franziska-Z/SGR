import bourbon
import torch

# Load Bourbon (pretrained on Rwanda)
model = bourbon.load_model(pretrained=True)
if torch.cuda.is_available(): model.cuda() elif torch.backends.mps.is_available(): model.to('mps')

# Predict for Kigali (returns dict with maps and counts)
# Note: Requires MPC dependencies installed
result = model.predict_coords(lat=-1.9441, lon=30.0619, size_meters=5000, ensemble=10)

print(f"Population: {result['pop_count']:.2f}")

# Access results
pop_map = result['pop_map'] # (H, W) array
if 'std_map' in result:
    uncert = result['std_map'] # Uncertainty
