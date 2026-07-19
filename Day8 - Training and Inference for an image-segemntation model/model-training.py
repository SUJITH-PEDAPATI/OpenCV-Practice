from ultralytics import YOLO

if __name__ == '__main__':
    # You were loading the 'best.pt' from Day 6, which was likely an Object Detection model (bounding boxes only).
    # To do segmentation, you need to load a Segmentation model (e.g., yolov8n-seg.pt) and train it on your segmentation dataset.
    
    # Load a pre-trained segmentation model
    model = YOLO("yolov8n-seg.pt") # You can also use the yolo26n-seg.pt in your root folder

    # Step 1: Train the segmentation model on Day 8's data
    # (Uncomment the training part and run this once to generate a new best.pt for segmentation)
    model.train(
        data=r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day8 - Training and Inference for an image-segemntation model\data.yaml",
        batch=16,
        workers=1,
        epochs=100,
    )

    # Step 2: After training is done, you can load your new segmentation 'best.pt' for inference
    # model = YOLO(r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day8 - Training and Inference for an image-segemntation model\runs\segment\train\weights\best.pt")
    # model.predict(
    #     source=r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day6 - Model Training on Custom Data\video.mov",
    #     show=True,
    #     line_width=2,
    #     show_boxes=False
    # )