from ultralytics import YOLO

# Ablation: UNet-YOLO26 (standard UNet enhance + YOLOv26, no deformable conv)
# D-UNet entry: main.py  -> dunet-yolo26x.yaml
# UNet entry:   unet-yolo.py -> unet-yolo26x.yaml
model = YOLO("ultralytics/cfg/models/26/unet-yolo26x.yaml")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="ultralytics/cfg/datasets/coco8.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    batch=6,
)

# # Evaluate the model's performance on the validation set
# metrics = model.val()

# # Perform object detection on an image
# results = model("path/to/image.jpg")  # Predict on an image
# results[0].show()  # Display results

# # Export the model to ONNX format for deployment
# path = model.export(format="onnx")  # Returns the path to the exported model
