from detectron2.config import get_cfg

def setup_cfg(config_file, overrides=None):
    cfg = get_cfg()

    cfg.MODEL.LINE_LOSS_WEIGHT = 0.0

    cfg.merge_from_file(config_file)

    if overrides:
        cfg.merge_from_list(overrides)

    cfg.freeze()
    return cfg
