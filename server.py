import pyrealsense2 as rs
import numpy as np
import zmq
import json
import base64

# Initialize ZeroMQ context and REP socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# Set a timeout for recv to avoid indefinite blocking
socket.RCVTIMEO = 1000

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream both color and depth images
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Function to convert RealSense frame to a numpy array
def rs_frame_to_np(frame):
    return np.asanyarray(frame.get_data())

try:
    # Skip the first few frames to allow the pipeline to stabilize
    skip_frames = 25
    while skip_frames > 0:
        pipeline.wait_for_frames()
        skip_frames -= 1

    while True:
        try:
            # Wait for a request from the client, but with timeout
            message = socket.recv() # Receive request from client
            print("Received message: ", message)

            # Capture the next set of frames
            frames = pipeline.wait_for_frames()

            # Align depth frame to color frame
            align_to = rs.stream.color
            align = rs.align(align_to)
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("No valid frames received.")
                continue

            # Convert frames to numpy arrays
            color_image = rs_frame_to_np(color_frame)
            depth_image = rs_frame_to_np(depth_frame)

            # Encode the images to base64 strings
            color_encoded = base64.b64encode(color_image).decode('utf-8')
            depth_encoded = base64.b64encode(depth_image).decode('utf-8')

            # Serialize images and send them to  client
            data = {
                "color_image": color_encoded,
                "depth_image": depth_encoded
            }

            # Send the JSON data to the client
            socket.send_json(data)
            print("Sent image to client as JSON.")
        
        except zmq.Again:
            pass

except KeyboardInterrupt:
    print("\nServer stopped by user")

finally:
    # Stop the pipeline when done
    pipeline.stop()
    socket.close()
    context.term()
