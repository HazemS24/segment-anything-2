import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

# Mask function
def show_mask(mask, ax, obj_id=None):
    h, w = mask.shape
    # Create a colormap with bright colors
    colormap = plt.get_cmap('tab20')  # Using 'tab20' for distinct colors
    color = colormap(obj_id % 20)  # Ensure a unique color for each mask
    color = np.array([color[0], color[1], color[2], 0.5])  # Add transparency

    mask_image = np.zeros((h, w, 4))
    mask_image[..., :3] = color[:3]
    mask_image[..., 3] = mask  # Set alpha channel to mask

    ax.imshow(mask_image, alpha=0.5)

# Load the image and convert to RGB format
image_path = "./white_bag_pics/4_Color.png"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)  # Convert image to numpy array

# Set the image for the predictor
predictor.set_image(image_np)

height, width = image_np.shape[0], image_np.shape[1]
num_points = 10  # Number of points to generate

# Create a grid of points in the range [0, 1] to represent normalized coordinates
grid_size = int(np.ceil(np.sqrt(num_points)))
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
xx, yy = np.meshgrid(x, y)
grid_points = np.vstack([xx.ravel(), yy.ravel()]).T[:num_points]  # Select the first `num_points` points

# Use a Gaussian distribution to create a density map
x_center = width / 2
y_center = height / 2
sigma = min(width, height) / 4  # Standard deviation for Gaussian

# Transform grid points to be more dense in the center
points = np.array([
    [
        x_center + (x - x_center) * np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2)),
        y_center + (y - y_center) * np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2))
    ]
    for x, y in grid_points
])

# Make sure points are within image bounds
points = np.clip(points, [0, 0], [width, height])

# Labels
labels = np.ones(num_points, dtype=np.int32)  # Label all points as foreground

# Segment objects in the image
try:
    # Ensure that point_coords and point_labels are both numpy arrays
    points_np = np.array(points)
    labels_np = np.array(labels)

    # Ensure both arrays are of the correct shape and dtype
    assert points_np.shape[1] == 2  # Check if point_coords has the right shape
    assert labels_np.shape[0] == points_np.shape[0]  # Check if point_labels matches the number of points

    masks, obj_ids, _ = predictor.predict(
        point_coords= points_np,  # Convert to list if required by the method
        point_labels= labels_np   # Convert to list if required by the method
    )

    print(f"Number of masks: {len(masks)}")
    for i, mask in enumerate(masks):
        print(f"Mask {i+1} shape: {mask.shape}")
        mask_np = mask
        print(f"Mask {i+1} unique values: {np.unique(mask_np)}")
    print(f"Object IDs: {obj_ids}")

    # Show results
    plt.figure(figsize=(12, 8))
    plt.title("Segmented Image")

    plt.imshow(image_np)  # Show the original image

    # Convert the masks to a numpy array and show
    for i, mask in enumerate(masks):
        mask = (mask > 0.0)
        show_mask(mask, plt.gca(), obj_id=obj_ids[i])

    plt.show()

    # Save the output with masks
    output_path = "output_with_masks.png"
    plt.savefig(output_path)
except Exception as e:
    print(f"An error occurred: {e}")
