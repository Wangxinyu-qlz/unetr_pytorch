from easydict import EasyDict

cfg = EasyDict()
cfg.num_workers = 8
cfg.batch_size = 1
cfg.epoch = 500
cfg.learning_rate = 1e-4
cfg.weight_decay = 1e-5
cfg.patience = 25
cfg.inference_threshold = 0.75

cfg.unetr = EasyDict()
cfg.unetr.img_dim = (128, 128, 128)
cfg.unetr.in_channels = 1  # CT只有一个通道
cfg.unetr.base_filter = 32
cfg.unetr.class_num = 26  # 26分类（算上background）
cfg.unetr.patch_size = 16
cfg.unetr.embedding_dim = 768
cfg.unetr.block_num = 12
cfg.unetr.head_num = 12
cfg.unetr.mlp_dim = 3072
cfg.unetr.z_idx_list = [3, 6, 9, 12]
