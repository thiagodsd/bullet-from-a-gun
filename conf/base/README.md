# templates

## yolo

### catalog

```yaml
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
yolo.yolov8_conf8_v1.hyperparameter_tuning_results:
    <<: *template_hyperparameter_tuning_json

yolo.yolov8_conf8_v1.model:
    type: MemoryDataset

yolo.yolov8_conf8_v1.fine_tuning_results:
    <<: *template_fine_tuning_json

yolo.yolov8_conf8_v1.evaluation_results:
    <<: *template_evaluation_json

yolo.yolov8_conf8_v1.evaluation_plots:
    <<: *template_evaluation_matplotlib
```

### parameters

```yaml
    yolov8_conf8_v1: # yolov8n, epoch 256, batch 16, img_size 640, workers 16
        dataprep_params:
            experiment_id: "yolov8_conf8_v1"
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
            model_name : yolov8n.pt
            experiment_name: "yolov8_conf8_v1"
            model_config:
                epochs: 256
                batch: 16
                img_size: 640
                optimizer: "auto"
                lr0: 0.001
                lrf: 0.0001
                workers: 16
                iou: 0.5
```

### pipeline

```python
    # gunshot :: yolov8 :: yolov8_conf8_v1
    yolov8_conf8_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="yolo.yolov8_conf8_v1",
    )
    # kedro run -n yolo.yolov8_conf8_v1.fine_tune_yolo
    # kedro run -n yolo.yolov8_conf8_v1.evaluate_yolo
    # kedro run -n yolo.yolov8_conf8_v1.compress_results_yolo
    # kedro run -n yolo.yolov8_conf8_v1.fine_tune_yolo,yolo.yolov8_conf8_v1.evaluate_yolo
```

## detectron2

### catalog

```yaml
# rccn_101_conf5_v1
detectron2.rccn_101_conf5_v1.fine_tuning_results:
    <<: *template_fine_tuning_json
detectron2.rccn_101_conf5_v1.evaluation_results:
    <<: *template_evaluation_json
detectron2.rccn_101_conf5_v1.evaluation_plots:
    <<: *template_evaluation_matplotlib
```

### parameters

```yaml
    rccn_101_conf5_v1: # rcnn, epoch 256, batch 2, img_size 512, workers 2
        dataprep_params:
            experiment_id: "detectron2_rccn_101_conf5_v1"
            coco_data:
                path:
                    - data
                    - 05_model_input
                    - gunshots
                    - coco
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
            pretrained_model_config: COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
            num_workers: 8
            pretrained_model_weights: COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
            ims_per_batch: 8
            base_lr: 0.00125
            max_iter: 1024
            steps: []
            batch_size_per_image: 640
            num_classes: 2
            score_thresh_test: 0.5
```

### pipeline

```python
    # gunshot :: detectron2 :: rccn_101_conf5_v1
    detectron2_rccn_101_conf5_v1 = pipeline(
        pipe=template_fine_tuning,
        namespace="detectron2.rccn_101_conf5_v1",
    )
    # kedro run -n detectron2.rccn_101_conf5_v1.fine_tune_detectron2
    # kedro run -n detectron2.rccn_101_conf5_v1.evaluate_detectron2
    # kedro run -n detectron2.rccn_101_conf5_v1.compress_results_detectron2
```