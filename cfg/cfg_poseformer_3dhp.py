data_config = {
    # 3D pose dataset
    'DATASET': '3dhp',  #  h36m, humaneva, aibee
    'WORLD_3D_GT_EVAL': True,

    # 2D pose detection
    'KEYPOINTS': 'gt',  # gt -> dataset specific, universal -> cross dataset

    # dataset for training and evaluation
    'TRAIN_SUBJECTS': 'S1_Seq1_0,S1_Seq1_1,S1_Seq1_2,S1_Seq1_4,S1_Seq1_5,S1_Seq1_6,S1_Seq1_7,'
                      'S1_Seq1_8,'
                      'S1_Seq2_0,S1_Seq2_1,S1_Seq2_2,S1_Seq2_4,S1_Seq2_5,S1_Seq2_6,S1_Seq2_7,'
                      'S1_Seq2_8,'
                      'S2_Seq1_0,S2_Seq1_1,S2_Seq1_2,S2_Seq1_4,S2_Seq1_5,S2_Seq1_6,S2_Seq1_7,'
                      'S2_Seq1_8,'
                      'S2_Seq2_0,S2_Seq2_1,S2_Seq2_2,S2_Seq2_4,S2_Seq2_5,S2_Seq2_6,S2_Seq2_7,'
                      'S2_Seq2_8,'
                      'S3_Seq1_0,S3_Seq1_1,S3_Seq1_2,S3_Seq1_4,S3_Seq1_5,S3_Seq1_6,S3_Seq1_7,'
                      'S3_Seq1_8,'
                      'S3_Seq2_0,S3_Seq2_1,S3_Seq2_2,S3_Seq2_4,S3_Seq2_5,S3_Seq2_6,S3_Seq2_7,'
                      'S3_Seq2_8,'
                      'S4_Seq1_0,S4_Seq1_1,S4_Seq1_2,S4_Seq1_4,S4_Seq1_5,S4_Seq1_6,S4_Seq1_7,'
                      'S4_Seq1_8,'
                      'S4_Seq2_0,S4_Seq2_1,S4_Seq2_2,S4_Seq2_4,S4_Seq2_5,S4_Seq2_6,S4_Seq2_7,'
                      'S4_Seq2_8,'
                      'S5_Seq1_0,S5_Seq1_1,S5_Seq1_2,S5_Seq1_4,S5_Seq1_5,S5_Seq1_6,S5_Seq1_7,'
                      'S5_Seq1_8,'
                      'S5_Seq2_0,S5_Seq2_1,S5_Seq2_2,S5_Seq2_4,S5_Seq2_5,S5_Seq2_6,S5_Seq2_7,'
                      'S5_Seq2_8,'
                      'S6_Seq1_0,S6_Seq1_1,S6_Seq1_2,S6_Seq1_4,S6_Seq1_5,S6_Seq1_6,S6_Seq1_7,'
                      'S6_Seq1_8,'
                      'S6_Seq2_0,S6_Seq2_1,S6_Seq2_2,S6_Seq2_4,S6_Seq2_5,S6_Seq2_6,S6_Seq2_7,'
                      'S6_Seq2_8,'
                      'S7_Seq1_0,S7_Seq1_1,S7_Seq1_2,S7_Seq1_4,S7_Seq1_5,S7_Seq1_6,S7_Seq1_7,'
                      'S7_Seq1_8,'
                      'S7_Seq2_0,S7_Seq2_1,S7_Seq2_2,S7_Seq2_4,S7_Seq2_5,S7_Seq2_6,S7_Seq2_7,'
                      'S7_Seq2_8,'
                      'S8_Seq1_0,S8_Seq1_1,S8_Seq1_2,S8_Seq1_4,S8_Seq1_5,S8_Seq1_6,S8_Seq1_7,'
                      'S8_Seq1_8,'
                      'S8_Seq2_0,S8_Seq2_1,S8_Seq2_2,S8_Seq2_4,S8_Seq2_5,S8_Seq2_6,S8_Seq2_7,'
                      'S8_Seq2_8',
    'TEST_SUBJECTS': 'TS1,TS3,TS4',

    'GT_3D': '/ssd/yzhan/data/benchmark/3D/mpi_inf_3dhp/data_3d_3dhp.npz',
    'GT_2D': '/ssd/yzhan/data/benchmark/3D/mpi_inf_3dhp/data_2d_3dhp_gt.npz',
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

    'FRAME_PATH': '/ssd/yzhan/data/benchmark/3D/mpi_inf_3dhp'
}

model_config = {
    # method to calculate 3D pose
    'MODEL': 'PoseFormer',

    # initialize trajectory model
    'TRAJECTORY_MODEL': True,

    # how many frames used as input
    'NUM_FRAMES': 9,

    # number of key-points
    'NUM_KPTS': 17,

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
    'VIZ_SUBJECT': 'TS1',

    # action to render
    'VIZ_ACTION': 'Action',

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
