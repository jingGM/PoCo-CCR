from src.trainer import Trainer
from src.rerank_data_generation import RerankDataGenerator
from utils.arguments import get_configuration
from utils.config import ModelType, DatasetType

if __name__ == "__main__":
    cfg = get_configuration()
    if cfg.model.model_type == ModelType.fine_matching and cfg.generate_data:
        cfg.model.model_type = ModelType.coarse_matching
        generator = RerankDataGenerator(device=cfg.gpus.device, model_cfg=cfg.model, snapshot=cfg.snapshot,
                                        distance_type=cfg.loss.distance_type, sanity_check=cfg.sanity_check,
                                        data_loader_cfg=cfg.data_loader, rerank_num=cfg.rerank_num)
        types = [DatasetType.training]
        if cfg.evaluation_freq > 0:
            types.append(DatasetType.evaluation)
        if cfg.recall_frequency > 0:
            types.append(DatasetType.test)
        generator.run(types)
        # cfg.model.model_type = ModelType.fine_matching
    else:
        trainner = Trainer(cfgs=cfg)
        trainner.run()
