"""
train_artefact_detector.py - Train YOLOv8 object detection model
"""

import os
import numpy as np
import cv2
from sample_generation import sample, width_um, height_um, artifact_layer
from afm_utils import create_fov_image

def generate_training_data(num_samples=2000, output_dir="artefact_data"):
    """Generate training data"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    # Known artefact positions
    known_artefacts = [
        {'id': 0, 'type': 'cross', 'x': 100, 'y': 100, 'size': 60},
        {'id': 1, 'type': 'fiducial', 'x': 500, 'y': 500, 'size': 50},
        {'id': 2, 'type': 'fiducial', 'x': 2000, 'y': 1500, 'size': 50},
        {'id': 3, 'type': 'circle', 'x': 800, 'y': 800, 'size': 50},
        {'id': 4, 'type': 'square', 'x': 1200, 'y': 1200, 'size': 40},
        {'id': 5, 'type': 'cross', 'x': 1800, 'y': 600, 'size': 55},
    ]
    
    type_to_class = {'cross': 0, 'fiducial': 1, 'circle': 2, 'square': 3}
    
    for i in range(num_samples):
        # Random field of view position
        x = np.random.uniform(0, width_um - 842)
        y = np.random.uniform(0, height_um - 631)
        
        # Get field of view image
        fov, _, _ = create_fov_image(sample, artifact_layer, True, x, y, 842, 631)
        
        # Save image
        cv2.imwrite(f"{output_dir}/images/sample_{i:04d}.png", fov)
        
        # Generate YOLO format labels
        labels = []
        h, w = fov.shape
        
        for artefact in known_artefacts:
            ax, ay = artefact['x'], artefact['y']
            size = artefact['size']
            
            # Check if within field of view
            if x <= ax <= x + w and y <= ay <= y + h:
                center_x = (ax - x) / w
                center_y = (ay - y) / h
                width_norm = size / w
                height_norm = size / h
                class_id = type_to_class[artefact['type']]
                labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width_norm:.6f} {height_norm:.6f}")
        
        if labels:
            with open(f"{output_dir}/labels/sample_{i:04d}.txt", "w") as f:
                f.write("\n".join(labels))
        
        if (i + 1) % 200 == 0:
            print(f"Generated {i+1}/{num_samples} samples")
    
    print(f"Data generation complete: {output_dir}")

def train_model():
    """Train YOLOv8 model"""
    from ultralytics import YOLO
    
    # Create dataset configuration file
    with open("artefact_dataset.yaml", "w") as f:
        f.write("""
path: ./artefact_data
train: images
val: images

nc: 4
names: ['cross', 'fiducial', 'circle', 'square']
""")
    
    # Train
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='artefact_dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='cpu',
        workers=0,
        project='artefact_detector',
        name='exp1',
        verbose=True
    )
    
    print("Training complete! Model saved to artefact_detector/exp1/weights/best.pt")

if __name__ == "__main__":
    # Generate training data
    generate_training_data(2000)
    # Train model (requires ultralytics)
    # train_model()