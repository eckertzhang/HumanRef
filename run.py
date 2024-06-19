import argparse
import logging
import os, sys
os.environ['TORCH_HOME'] = '/apdcephfs_cq10/share_1290939/vg_share/eckertzhang/Weights/TorchHub'
os.environ['WEIGHT_PATH'] = '/apdcephfs_cq10/share_1290939/vg_share/eckertzhang/Weights'
os.environ['ECON_PATH'] = './data/Results_ECON'  # Path saving Results_ECON
os.environ['Human3D_PATH'] = '/apdcephfs_cq10/share_1290939/eckertzhang'  # Path saving 3D human datasets: CAPE/Thuman2

class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/humanref.yaml', help="path to config file")
    parser.add_argument("--gpu", default="0", help="GPU(s) to be used")
    group = parser.add_mutually_exclusive_group(required=False)  #required=True
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="if true, set logging level to DEBUG")
    parser.add_argument("--typecheck", action="store_true", help="whether to enable dynamic type checking",)
    args, extras = parser.parse_known_args()

    # args.train = True

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    n_gpus = len(args.gpu.split(","))

    import pytorch_lightning as pl
    import torch
    from PIL import Image
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    if args.typecheck:
        from jaxtyping import install_import_hook
        install_import_hook("threestudio", "typeguard.typechecked")

    import threestudio
    from threestudio.systems.base import BaseSystem
    from threestudio.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
    )
    from threestudio.utils.config import ExperimentConfig, load_yaml, resolve_config
    from threestudio.utils.typing import Optional
    import warnings
    warnings.filterwarnings('ignore')

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            handler.addFilter(ColoredFilter())

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_yaml(args.config, cli_args=extras, n_gpus=n_gpus)
    try:
        if cfg.resume is not None:
            cfg.system.run_initial = False
    except:
        print('Training from scratch!')
    
    # caption generation (require large memery, you can input a given prompt to skip this step!)
    if cfg.prompt is None or cfg.prompt == '':
        print("load blip2 for image caption...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processor = Blip2Processor.from_pretrained(os.path.join(os.getenv('WEIGHT_PATH'),"blip2-opt-2.7b"))
        blip_model = Blip2ForConditionalGeneration.from_pretrained(os.path.join(os.getenv('WEIGHT_PATH'),"blip2-opt-2.7b"), torch_dtype=torch.float16).to(device)
        image_pil = Image.open(cfg.image_path).convert("RGB")
        inputs = processor(image_pil, return_tensors="pt").to(device, torch.float16)
        out = blip_model.generate(**inputs)
        caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        caption = caption.replace("there is ", "")
        caption = caption.replace("close up", "photo")
        for d in ["black background", "white background"]:
            if d in caption:
                caption = caption.replace(d, "ground")
        print("*** Predicted Caption: ", caption)
        cfg.prompt = caption + ', high quality'
        del blip_model, processor, inputs, out
        torch.cuda.empty_cache()
    
    # Resolves all interpolations in the given config object in-place.
    cfg = resolve_config(cfg)
    cfg.data.workspace = cfg.trial_dir
    pl.seed_everything(cfg.seed)

    dm = threestudio.find(cfg.data_type)(cfg.data)
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))
    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            CustomProgressBar(refresh_rate=1),
            # CodeSnapshotCallback(
            #     os.path.join(cfg.trial_dir, "code"), use_version=False
            # ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # make tensorboard logging dir to suppress warning
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
            CSVLogger(cfg.trial_dir, name="csv_logs"),
        ] + system.get_loggers()
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "log.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()

    trainer = Trainer(
        callbacks=callbacks, logger=loggers, inference_mode=False, **cfg.trainer
    )

    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

    if args.train:
        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        trainer.test(system, datamodule=dm)
    elif args.validate:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.test:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.export:
        set_system_status(system, cfg.resume)
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)


if __name__ == "__main__":
    main()
