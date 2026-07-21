from ultralytics import YOLO

model = YOLO(r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day13 - How to benchMark different Models\best-detect.pt")

## We can export the Benchmark models in different formats:
## 1. ONNX
## 2. TensorRT
## 3. OpneVINO
## CoreML,Tensorflow and more

if __name__ == '__main__':
    model.benchmark(
        data = r"F:\D Drive\Linkedin Learning\Open CV, Hands on AI\Basics of Open CV\Day13 - How to benchMark different Models\data.yaml",
        # format = "tensorflow" ## By default it exports in all the formats available
    )
    print("Model BenchMark...")