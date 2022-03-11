data_config = {
    # 3D pose dataset
    'DATASET': 'h36m',  #  h36m, humaneva, aibee
    'WORLD_3D_GT_EVAL': True,

    # 2D pose detection
    'KEYPOINTS': 'gt',  # gt -> dataset specific, universal -> cross dataset

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

    'FRAME_PATH': '/ssd/yzhan/data/benchmark/3D/showroom/20210702/frame/',

    'INTRINSIC_ENCODING': False,
    'RAY_ENCODING': False
}

model_config = {
    # method to calculate 3D pose
    'MODEL': 'RIE',  # VideoPose3D, PoseFormer, RIE

    # initialize trajectory model
    'TRAJECTORY_MODEL': True,
    'BONE_COMPARISON': False,

    # number for layers
    'ARCHITECTURE': '3,3',

    # dropout probability
    'DROPOUT': 0.2,

    # how many frames used as input
    'NUM_FRAMES': 9,

    # use causal convolutions for real-time processing
    'CAUSAL': False,

    # number of channels in convolution layers
    'CHANNELS': 256,

    # use dense convolutions instead of dilated convolutions
    'DENSE': False,

    # number of key-points
    'NUM_KPTS': 17,

    'INPUT_DIM': 2,

    'CAMERA_EMBDDING': False,
    'EXTRINSIC_DIM': 2,
    'EMBEDD_DIM': 64,

    # disable optimized model for single-frame predictions
    'DISABLE_OPTIMIZATIONS': False,

    'PRETRAIN': '',
    'FINETUNE': '/model/from/stage2',
    'STAGE': 3,
    'LATENT_FEATURES_DIM': 256
}

train_config = {
    # number of training epochs
    'EPOCHS': 20,

    # batch size in terms of predicted frames
    'BATCH_SIZE': 1024,

    # initial learning rate
    'LEARNING_RATE': 5e-4,
    'LEARNING_RATE_TRAJECTORY': 2e-6,
    # learning rate decay per epoch
    'LR_DECAY': 0.95,

    # optimizer
    'OPTIMIZER': 'Adam',

    'INITIAL_MOMENTUM': 0.01,
    'FINAL_MOMENTUM': 0.001,

    # train-time flipping
    'TRAIN_TIME_AUGMENTATION': True,

    # terst-time flipping
    'TEST_TIME_AUGMENTATION': False,  # always set it as False

    # DEVICE ID
    'DEVICE': '0',

    # disable optimized model for single-frame predictions
    'DISABLE_OPTIMIZATIONS': False,

    'CATEGORY': 'RIE',
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
