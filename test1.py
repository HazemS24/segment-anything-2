import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

# Use SAM2AutomaticMaskGenerator
mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

# Load the image and convert to RGB format
image_path = "./white_bag_pics/3_Color.png"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)  # Convert image to numpy array

# Automatically generate masks for the image
masks = mask_generator.generate(image_np)

# Function to calculate the centroid of a mask
def find_centroid(mask):
    y_coords, x_coords = np.where(mask > 0)
    
    if len(y_coords) == 0:  # If there are no pixels in the mask
        return None
    
    # Calculate the centroid
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    
    return (centroid_x, centroid_y)

# Array to store centroids
centroids = []

# Mask function
def show_mask(mask, ax, obj_id=None, centroid=None):
    h, w = mask.shape
    # Create a colormap with bright colors
    colormap = plt.get_cmap('tab20')  # Using 'tab20' for distinct colors
    color = colormap(obj_id % 20)  # Ensure a unique color for each mask
    color = np.array([color[0], color[1], color[2], 0.5])  # Add transparency

    mask_image = np.zeros((h, w, 4))
    mask_image[..., :3] = color[:3]
    mask_image[..., 3] = mask  # Set alpha channel to mask

    ax.imshow(mask_image, alpha=0.5)

    # Mark the centroid with a red dot
    if centroid is not None:
        ax.plot(centroid[0], centroid[1], 'ro', markersize=5)  # 'ro' means red dot

# Show results
plt.figure(figsize=(12, 8))
plt.title("Segmented Image")

plt.imshow(image_np) # Show the original image

# Display the automatically generated masks and calculate centroids
for i, mask_dict in enumerate(masks):
    mask = mask_dict['segmentation']  # Extract the 'segmentation' mask
    if isinstance(mask, np.ndarray):  # Check if mask is a numpy array
        mask = (mask > 0.0)  # Convert to binary mask
        
        # Calculate the centroid
        centroid = find_centroid(mask)
        centroids.append(centroid)  # Store centroid
        
        show_mask(mask, plt.gca(), obj_id=i, centroid=centroid)  # Display each mask with centroid
    else:
        print(f"Mask {i} is not a numpy array: {mask}")

plt.axis('off')
plt.show()

# Print the array of centroids
print("Centroids of masks:", centroids)

# Save the output with masks
output_path = "output_with_masks.png"
plt.savefig(output_path)
