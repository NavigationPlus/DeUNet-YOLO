from ultralytics import YOLO

# D-UNet-YOLO26: Deformable UNet enhancement + YOLOv26 detector
# Use scale suffix n/s/m/l/x, e.g. dunet-yolo26n.yaml / dunet-yolo26x.yaml
model = YOLO("ultralytics/cfg/models/26/dunet-yolo26x.yaml")

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
