data_config = {
    # 3D pose dataset
    'DATASET': 'h36m',  #  h36m, humaneva, aibee
    'WORLD_3D_GT_EVAL': True,

    # 2D pose detection
    'KEYPOINTS': 'universal',  # gt, detectron_ft_h36m, detectron_pt_coco, cpn_ft_h36m_dbb

    # dataset for training and evaluation
    'TRAIN_SUBJECTS': 'S1,S5,S6,S7,S8',
    'TEST_SUBJECTS': 'S9,S11',

    'GT_3D': '/ssd/yzhan/data/benchmark/3D/h36m/VideoPose3D/data/data_3d_h36m.npz',
    'GT_2D': '/ssd/yzhan/data/benchmark/3D/h36m/VideoPose3D/data/data_2d_h36m_gt.npz',
    'CAMERA_PARAM': '',

    # reduce dataset size by fraction
    'SUBSET': 1,

    # chunk size to use during training
    'STRIDE': 1,

    # downsample frame rate by factor
    'DOWNSAMPLE': 1,

    # actions for training and evaluation
    'ACTIONS': '*',

    'REMOVE_IRRELEVANT_KPTS': False,

    'FRAME_PATH': '/ssd/yzhan/data/benchmark/3D/showroom/20210702/frame/'
}

model_config = {
    # method to calculate 3D pose
    'MODEL': 'PoseFormer',

    # initialize trajectory model
    'TRAJECTORY_MODEL': True,

    # how many frames used as input
    'NUM_FRAMES': 9,

    # number of key-points
    'NUM_KPTS': 14,

    'INPUT_DIM': 2,

    'PRETRAIN': ''
}

train_config = {
    # number of training epochs
    'EPOCHS': 128,

    # batch size in terms of predicted frames
    'BATCH_SIZE': 512,

    # initial learning rate
    'LEARNING_RATE': 4e-5,
    # learning rate decay per epoch
    'LR_DECAY': 0.98,

    # optimizer
    'OPTIMIZER': 'AdamW',

    # train-time flipping
    'TRAIN_TIME_AUGMENTATION': True,

    # terst-time flipping
    'TEST_TIME_AUGMENTATION': True,  # always set it as False

    # DEVICE ID
    'DEVICE': '0',

    # disable optimized model for single-frame predictions
    'DISABLE_OPTIMIZATIONS': False,

    'CATEGORY': 'POSEFORMER',
    # checkpoint directory
    'CHECKPOINT': 'checkpoint',
    #  create a checkpoint every N epochs
    'CHECKPOINT_FREQUENCY': 1,

    # break down error by subject (on evaluation)
    'BY_SUBJECT': False,

    # save training curves as .png images
    'EXPORT_TRAINING_CURVES': False,
}

plot_config = {
    'EXP_PLOTTING': True,
    'SAVE_PLOTS': True,

    # subject to render
    'VIZ_SUBJECT': 'S9',

    # action to render
    'VIZ_ACTION': 'Walking',

    # camera to render
    'VIZ_CAMERA': 0,

    # path to input video
    'VIZ_VIDEO': '/ssd/yzhan/data/benchmark/3D/h36m/meta/S9/Videos/Walking.54138969.mp4',

    # skip first N frames of input video
    'VIZ_SKIP': 0,

    # output file name (.gif or .mp4)
    'VIZ_OUTPUT': 'output.gif',

    # only render first N frames
    'VIZ_LIMIT': -1,

    # downsample FPS by a factor N
    'VIZ_DOWNSAMPLE': 1,

    # image size
    'VIZ_SIZE': 6
}
