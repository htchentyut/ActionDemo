# import detectron2
# from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
import argparse
from demo.predictor import VisualizationDemo
from detectron2.data.detection_utils import read_image
import tqdm
import glob
import csv
import numpy as np
import os


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    cfg = get_cfg()

    parser = argparse.ArgumentParser(description="Detectron2")

    parser.add_argument(
        "--config-file",
        default="./configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def person_box(img_path, output_path):
    args = get_parser().parse_args()
    # print(args.opts)
    args.config_file = './configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'

    args.input = glob.glob(img_path + "/*.jpg")

    args.output = output_path

    args.NUM_GPUS = 2

    args.opts = ["MODEL.WEIGHTS", "/home/hchen/model_zoo/detectron2/model_final_5ad38f.pkl", "MODEL.DEVICE", "cuda"]

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    # if not os.path.exists(args.output):
    #     os.makedirs(args.output)
    csvfile = open(args.output + "detectrion.csv", "w+", encoding="gbk")

    CSVwriter = csv.writer(csvfile)

    CSVwriter.writerow(["image_name", "x", "y", "width", "height", "score", "class",
                        "kp01_x", "kp02_x", "kp03_x", "kp04_x", "kp05_x", "kp06_x", "kp07_x", "kp08_x", "kp09_x", "kp10_x",
                        "kp11_x", "kp12_x", "kp13_x", "kp14_x", "kp15_x", "kp16_x", "kp17_x",
                        "kp01_y", "kp02_y", "kp03_y", "kp04_y", "kp05_y", "kp06_y", "kp07_y", "kp08_y", "kp09_y", "kp10_y",
                        "kp11_y", "kp12_y", "kp13_y", "kp14_y", "kp15_y", "kp16_y", "kp17_y",
                        "kp01_s", "kp02_s", "kp03_s", "kp04_s", "kp05_s", "kp06_s", "kp07_s", "kp08_s", "kp09_s", "kp10_s",
                        "kp11_s", "kp12_s", "kp13_s", "kp14_s", "kp15_s", "kp16_s", "kp17_s"
    ])

    for img_file in tqdm.tqdm(args.input, disable=not args.output):
        # print(img_file)
        img = read_image(img_file, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        # print(predictions['instances'][0])

        pred_boxes = predictions['instances'].pred_boxes
        pred_scores = predictions['instances'].scores
        pred_classes = predictions['instances'].pred_classes
        pred_keypoints = predictions['instances'].pred_keypoints

        boxes = []
        scores = []
        classes = []
        keypoints = []
        for ibox in pred_boxes:
            boxes.append(ibox.cpu().numpy().tolist())
        for iscores in pred_scores:
            scores.append(iscores.cpu().numpy().tolist())
        for iclasses in pred_classes:
            classes.append(iclasses.cpu().numpy().tolist())
        for ikeypoints in pred_keypoints:
            keypoints.append(ikeypoints.cpu().numpy())

        for iinstance in range(len(boxes)):
            ins_info = [img_file] + boxes[iinstance] + [scores[iinstance]] + [classes[iinstance]] + np.array(keypoints[iinstance]).T.ravel().tolist()
            CSVwriter.writerow(ins_info)
    csvfile.close()
