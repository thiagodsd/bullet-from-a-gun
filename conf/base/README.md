# templates

## catalog

```yaml
yolo.yolov8_conf5_v1.model:
    type: MemoryDataset

yolo.yolov8_conf5_v1.fine_tuning_results:
    <<: *template_fine_tuning_json

yolo.yolov8_conf5_v1.evaluation_results:
    <<: *template_evaluation_json

yolo.yolov8_conf5_v1.evaluation_plots:
    <<: *template_evaluation_matplotlib
```

## parameters

```yaml
    yolov8_conf5_v1:
        dataprep_params:
            experiment_id: "yolov8_conf5_v1"
            yolo_data:
                path:
                    - data
                    - 05_model_input
                    - gunshots
                    - yolo
                    - v1
                datasets:
                    - train
                    - valid
                    - test
        fine_tuning_params:
            path:
                - data
                - 06_models
                - output
            model_name : yolov8s.pt
            experiment_name: "yolov8_conf5_v1"
            model_config:
                epochs: 128
                batch: 16
                img_size: 512
                optimizer: "auto" # Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration.
                lr: 0.0001
                workers: 8
                rect: True
                iou: 0.5
```

## pipeline

```python
    # gunshot :: yolov8 :: yolov8_conf5_v1
    yolo_rccn_101_conf2_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="yolo.yolov8_conf5_v1",
    )
    # kedro run -n yolo.yolov8_conf5_v1.fine_tune_yolo
    # kedro run -n yolo.yolov8_conf5_v1.evaluate_yolo
    # kedro run -n yolo.yolov8_conf5_v1.compress_results_yolo
```