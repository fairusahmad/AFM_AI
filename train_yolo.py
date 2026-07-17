from ultralytics import YOLO
import os

# Create dataset configuration file
with open('artefact_dataset.yaml', 'w') as f:
    f.write("""path: ./artefact_data
train: images
val: images

nc: 4
names: ['cross', 'fiducial', 'circle', 'square']
""")

print("Dataset configuration file created: artefact_dataset.yaml")

# Check if training data exists
if not os.path.exists("artefact_data/images"):
    print("Error: Training data not found. Please run train_artefact_detector.py first to generate data.")
    exit(1)

# Train model
print("Starting YOLO model training...")
model = YOLO('yolov8n.pt')

results = model.train(
    data='artefact_dataset.yaml',
    epochs=10,
    imgsz=640,
    batch=4,
    device='cpu',
    workers=0,
    project='artefact_detector',
    name='exp1',
    verbose=True
)

print("Training complete!")
print("Model saved to: artefact_detector/exp1/weights/best.pt")