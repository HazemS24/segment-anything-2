import pyrealsense2 as rs
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream both color and depth images
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Function to align depth and color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Helper function to convert RealSense frame to a numpy array
def rs_frame_to_np(frame):
    return np.asanyarray(frame.get_data())

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
    # Skip the first few frames to allow the pipeline to stabilize
    skip_frames = 25
    while skip_frames > 0:
        pipeline.wait_for_frames()
        skip_frames -= 1

    while True:
        # Wait for a coherent set of frames
        frames = pipeline.wait_for_frames()

        # Align depth frame to color frame
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        color_image = rs_frame_to_np(color_frame)
        depth_image = rs_frame_to_np(depth_frame)

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
    print("Stopped by user")

finally:
    # Stop the pipeline when done
    pipeline.stop()
