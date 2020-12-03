import torch
from cvi import CVI
from trainer import Trainer
import hydra
from omegaconf import DictConfig
# from ml_logger import logbook as ml_logbook
import logging
import numpy as np
from utils import set_seed_everywhere


@hydra.main(config_name="conf.yml")
def main(cfg: DictConfig):
    set_seed_everywhere(cfg.seed)

    # override device
    if not torch.cuda.is_available():
        cfg.device = "cpu"

    # logbook_config = ml_logbook.make_config(logger_dir="logs")
    # logger = ml_logbook.LogBook(config=logbook_config)
    logger = logging.getLogger(__name__)

    agent = CVI(cfg.agent).to(cfg.device)
    train_env = hydra.utils.instantiate(cfg.env)
    test_env = hydra.utils.instantiate(cfg.env)
    buffer = hydra.utils.instantiate(cfg.buffer)
    trainer = Trainer(train_env, test_env, agent, buffer, cfg)

    for iter in range(cfg.train_cycles):
        metrics = dict(global_step=iter)

        trainer.collect()
        m_ = trainer.train_model()
        metrics.update(m_)
        trainer.augment_buffer()
        m_ = trainer.train_value()
        metrics.update(m_)
        m_, *_ = trainer.eval()
        metrics.update(m_)

        trainer.plot_value_grid(f"value/{iter:03d}.png")
        trainer.animate(f"anim/{iter:03d}.gif")
        trainer.plot_vi_convergence(f"vi_convergence/{iter:03d}.png")

        # logger.write_metric(metrics)
        logger.info(metrics)


if __name__ == "__main__":
    main()
