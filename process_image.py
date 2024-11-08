import zmq
import json
import base64
import torch
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Initialize ZeroMQ context and REQ socket
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Ignore annoying warnings that clutter terminal logs
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

# Use SAM2AutomaticMaskGenerator
mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

# Function to calculate the centroid of a mask
def find_centroid(mask):
    y_coords, x_coords = np.where(mask > 0)
    
    if len(y_coords) == 0:  # If there are no pixels in the mask
        return None
    
    # Calculate the centroid
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    
    return (centroid_x, centroid_y)

# Function to calculate the area of a mask
def calculate_area(mask):
    return np.sum(mask)

# Function to calculate the average height and minimum height from depth image for a given mask
def calculate_heights(mask, depth_image):
    # Extract depth values within the mask
    masked_depth_values = depth_image[mask > 0]

    # Filter out zero values (assumed to be invalid depth values)
    valid_depth_values = masked_depth_values[masked_depth_values > 0]

    # no valid depth values
    if valid_depth_values.size == 0:
        print("Warning: No valid depth values for this mask.")
        return float('nan'), float('nan')

    if len(masked_depth_values) > 0:
        avg_height = np.mean(valid_depth_values) # Average depth value (height)
        min_height = np.min(valid_depth_values) # Minimum depth value (closest point)
        return avg_height, min_height
    else:
        return None, None

# Mask function
def show_mask(mask, ax, obj_id=None, centroid=None, avg_height=None, min_height=None):
    h, w = mask.shape
    
    # Create a colormap with bright colors
    colormap = plt.get_cmap('tab20')
    color = colormap(obj_id % 20) # Ensure a unique color for each mask
    color = np.array([color[0], color[1], color[2], 0.5])  # Add transparency

    mask_image = np.zeros((h, w, 4))
    mask_image[..., :3] = color[:3]
    mask_image[..., 3] = mask  # Set alpha channel to mask

    ax.imshow(mask_image, alpha=0.5)

    # Mark the centroid with a red dot
    if centroid is not None:
        ax.plot(centroid[0], centroid[1], 'ro', markersize=5)
        
        # Get the area for annotation
        area = calculate_area(mask)

        # Check for None before formatting height values
        avg_height_str = f"{avg_height:.2f} mm" if avg_height is not None else "N/A"
        min_height_str = f"{min_height:.2f} mm" if min_height is not None else "N/A"

        # Create a text label with area, avg height, and min height information
        label = f"Area: {area}, Avg Height: {avg_height_str}, Min Height: {min_height_str}"
        ax.text(centroid[0], centroid[1], label, fontsize=5, ha='center', color='white', backgroundcolor='black')

try:
    while (True):
        # Request image from server
        socket.send(b"capture")
        print("Getting image from server...")
        message = socket.recv()
        message_dict = json.loads(message.decode('utf-8'))

        # Deserialize the received images
        color_image = np.frombuffer(base64.b64decode(message_dict["color_image"]), dtype=np.uint8).reshape((480, 848, 3))
        depth_image = np.frombuffer(base64.b64decode(message_dict["depth_image"]), dtype=np.uint16).reshape((480, 848))

        # Automatically generate masks for the color image
        masks = mask_generator.generate(color_image)

        mask_info = [] # Store information about each mask

        # Show results
        plt.figure(figsize=(12, 8))
        plt.title("Segmented Image")
        plt.imshow(color_image)

        # Display the automatically generated masks and calculate centroids
        for i, mask_dict in enumerate(masks):
            mask = mask_dict['segmentation']

            if isinstance(mask, np.ndarray):
                if np.any(mask > 0):
                    mask = (mask > 0.0)

                    centroid = find_centroid(mask)
                    avg_height, min_height = calculate_heights(mask, depth_image)

                    mask_info.append({
                        'obj_id': i,
                        'avg_height': avg_height,
                        'min_height': min_height,
                        'color': plt.get_cmap('tab20')(i % 20), # Use colormap
                        'centroid': centroid
                    })

                    show_mask(mask, plt.gca(), obj_id=i, centroid=centroid, avg_height=avg_height, min_height=min_height)
                    
                else:
                    print(f"Mask {i} is empty, skipping display.")
            
            else:
                print(f"Mask {i} is not a numpy array: {mask}")

        plt.axis('off')
        plt.show()

        # Sort masks by average height
        sorted_by_avg = sorted(mask_info, key=lambda x: x['avg_height'])
        print("\nSorted masks by average height:")
        for info in sorted_by_avg:
            color_str = f"RGB({info['color'][0]*255:.0f}, {info['color'][1]*255:.0f}, {info['color'][2]*255:.0f})"
            print(f"Mask {info['obj_id']+1} - Avg Height: {info['avg_height']:.2f} mm, Color: {color_str}")

        # Sort masks by minimum height
        sorted_by_min = sorted(mask_info, key=lambda x: x['min_height'])
        print("\nSorted masks by minimum height:")
        for info in sorted_by_min:
            color_str = f"RGB({info['color'][0]*255:.0f}, {info['color'][1]*255:.0f}, {info['color'][2]*255:.0f})"
            print(f"Mask {info['obj_id']+1} - Min Height: {info['min_height']:.2f} mm, Color: {color_str}")

except KeyboardInterrupt:
    print("Stopping by user interrupt...")

finally:
    # Stop the pipeline when done
    print("Client stopped")
