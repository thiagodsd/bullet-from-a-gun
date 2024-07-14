"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 0.19.5
"""
import logging

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# coloredlogs.install(
#     level='DEBUG',
#     logger=logger,
#     fmt="""%(asctime)s :: [%(levelname)s] %(filename)s :: %(funcName)s :: L%(lineno)s\n%(message)s""",
#     field_styles={
#         'levelname': {'bold': True, 'color': 'magenta'}
#     },
#     level_styles={
#         'debug': {'italic': True,},
#         'info': {'italic': True, 'color': 'green'},
#         'warning': {'color': 'yellow'},
#     }
# )

def evaluate_model(
        evaluation_params: dict,
        images: dict,
    ):
    """
    `todo` documentation.
    """
    results_meta = dict()
    results_img = dict()

    logger.debug(evaluation_params)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(evaluation_params["pretrained_model_config"]))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = evaluation_params["score_thresh_test"]
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(evaluation_params["pretrained_model_config"])

    predictor = DefaultPredictor(cfg)

    for _key_ in images.keys():
        logger.debug(f"""{_key_} :: {images[_key_]}""")
        result_key = _key_.replace(".jpg", "").replace(".png", "")
        # loading image
        im = cv2.cvtColor(np.array(images[_key_]()), cv2.COLOR_RGB2BGR)
        outputs = predictor(im)
        # draw the predictions on the image
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # save the image
        results_meta[result_key] = {
            "num_instances": len(outputs["instances"].to("cpu")),
            "pred_boxes": outputs["instances"].to("cpu").pred_boxes.tensor.numpy().tolist(),
            "scores": outputs["instances"].to("cpu").scores.numpy().tolist(),
            "pred_classes": outputs["instances"].to("cpu").pred_classes.numpy().tolist(),
        }
        results_img[result_key] = Image.fromarray(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        logger.debug(f"""OUTPUTS :: {outputs}""")

    return results_meta, results_img
