train=dict(
    type='CustomDataset',
    pipeline=[  dict(type='LoadImageFromFile')],
    data_root='data/custom_dataset',
    test_mode=False,
    min_depth=1e-3,
    max_depth=10,
    depth_scale=1)