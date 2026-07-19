from ultralytics import YOLO

if __name__ == "__main__":
    # Move model initialization here to prevent multiprocessing errors on Windows
    model = YOLO(r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day10 - Validation\best-detect.pt")

    metrics = model.val(
        data=r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Guide\Ex_Files_Hands_on_AI-Ultralytics\Exercise Files\03-02\data.yaml"
    )
    print("Model Validation")
    print("mAP50-95:", metrics.box.map)   # Prints the mean Average Precision (mAP@50-95)
    print("AP Matrix:", metrics.box.mp)    # Prints the Average Precision Matrix
    print("Mean Recall:", metrics.box.mr)  # Prints the Mean Recall
    print("mAP50:", metrics.box.map50)     # Prints the Mean Average Precision at 50% (mAP@50)