data_config = {
    # 3D pose dataset
    'DATASET': 'humaneva',  #  h36m, humaneva, aibee
    'WORLD_3D_GT_EVAL': True,

    # 2D pose detection
    'KEYPOINTS': 'universal',  # gt -> dataset specific, universal -> cross dataset

    # dataset for training and evaluation
    'TRAIN_SUBJECTS': 'Train/S1,Train/S2,Train/S3',
    'TEST_SUBJECTS': 'Validate/S1,Validate/S2,Validate/S3',

    'GT_3D': '/ssd/yzhan/data/benchmark/3D/HumanEva_I/gt/data_3d_humaneva15.npz',
    'GT_2D': '/ssd/yzhan/data/benchmark/3D/HumanEva_I/gt/data_2d_humaneva15_gt.npz',
    'CAMERA_PARAM': '',

    # reduce dataset size by fraction
    'SUBSET': 1,

    # chunk size to use during training
    'STRIDE': 1,

    # downsample frame rate by factor
    'DOWNSAMPLE': 1,

    # actions for training and evaluation
    'ACTIONS': '*',

    'REMOVE_IRRELEVANT_KPTS': True,

    'FRAME_PATH': '/ssd/yzhan/data/benchmark/3D/showroom/20210702/frame/',

    'INTRINSIC_ENCODING': False,
    'RAY_ENCODING': False,
    'ADD_HEIGHT': False
}

model_config = {
    # method to calculate 3D pose
    'MODEL': 'VideoPose3D',  # VideoPose3D, PoseFormer

    # initialize trajectory model
    'TRAJECTORY_MODEL': False,
    'BONE_COMPARISON': False,

    # number for layers
    'ARCHITECTURE': '3,3',

    # dropout probability
    'DROPOUT': 0.25,

    # how many frames used as input
    'NUM_FRAMES': 9,

    # use causal convolutions for real-time processing
    'CAUSAL': False,

    # number of channels in convolution layers
    'CHANNELS': 1024,

    # use dense convolutions instead of dilated convolutions
    'DENSE': False,

    # number of key-points
    'NUM_KPTS': 14,

    'INPUT_DIM': 2,

    # disable optimized model for single-frame predictions
    'DISABLE_OPTIMIZATIONS': False,

    'PRETRAIN': ''
}

train_config = {
    # number of training epochs
    'EPOCHS': 1024,

    # batch size in terms of predicted frames
    'BATCH_SIZE': 128,

    # initial learning rate
    'LEARNING_RATE': 1e-3,
    # learning rate decay per epoch
    'LR_DECAY': 0.996,

    # optimizer
    'OPTIMIZER': 'Adam',

    'INITIAL_MOMENTUM': 0.01,
    'FINAL_MOMENTUM': 0.001,

    # train-time flipping
    'TRAIN_TIME_AUGMENTATION': True,

    # terst-time flipping
    'TEST_TIME_AUGMENTATION': False,  # always set it as False

    # DEVICE ID
    'DEVICE': '3',

    # disable optimized model for single-frame predictions
    'DISABLE_OPTIMIZATIONS': False,

    'CATEGORY': 'HumanEva_RAY3D',
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
    'VIZ_SUBJECT': 'Validate/S2',

    # action to render
    'VIZ_ACTION': 'ThrowCatch 1 chunk10',

    # camera to render
    'VIZ_CAMERA': 0,

    # path to input video
    'VIZ_VIDEO': '/ssd/yzhan/data/human3.6m/extracted/S9/Videos/Walking.54138969.mp4',

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
