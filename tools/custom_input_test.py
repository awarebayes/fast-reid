import sys
import unittest

import torch
import torchvision.datasets
import torchvision.transforms as T

sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.modeling.backbones import build_backbone
from fastreid.data.transforms.build import build_transforms
from fastreid.evaluation import inference_on_dataset, print_csv_format, ReidEvaluator
from fastreid.utils.checkpoint import Checkpointer
from fastreid.modeling import build_model
from fastreid.data import build_reid_test_loader, build_reid_train_loader


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def try_compare(cfg):
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = 413

    model = build_model(cfg)
    model.eval()

    Checkpointer(model).load("/home/mscherbina/Documents/work/vas-mlops/logs/market1501/sbs_S50/model_best.pth")
    dataset = torchvision.datasets.ImageFolder(
        "/home/mscherbina/Documents/work/vas-cpp/logs/one_frame",

        transform=T.Compose(
            [
                T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
                T.PILToTensor(),
                T.Lambda(lambda x: x.float()),
            ]
    ))
    model.cuda()


    # loader, num_query = build_reid_test_loader(dataset, 8, 1)
    loader = torch.utils.data.DataLoader(dataset, 8)
    evaluator = ReidEvaluator(cfg, 1, None)
    results_i = inference_on_dataset(model, loader, evaluator, flip_test=True)
    print(results_i)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    try_compare(cfg)

