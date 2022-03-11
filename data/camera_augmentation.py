import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.camera.camera import CameraInfoPacket, catesian2homogenous
from lib.skeleton.bone import get_bone_length_from_3d_pose, get_bone_unit_vector_from_3d_pose, get_3d_pose_from_bone_vector

import ipdb
import copy
import glob
import json
import torch
import numpy as np
import scipy.linalg as linalg

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

H36M_KPT_IDX = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

RES_W, RES_H = 1000, 1000

AXIS_X, AXIS_Y, AXIS_Z = [1, 0, 0], [0, 1, 0], [0, 0, 1]

TRAIN_SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8']

TEST_SUBJECTS = ['S9', 'S11']

ALL_SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

IDX_TO_CAMERA = {
    0: '54138969',
    1: '55011271',
    2: '58860488',
    3: '60457274'
}

h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.541504956548, 515.4514869776],
        'focal_length': [1145.04940458804, 1143.78109572365],
        'radial_distortion': [-0.207098910824901, 0.247775183068982, -0.00307515035078854],
        'tangential_distortion': [-0.00142447157470321, -0.000975698859470499],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70,  # Only used for visualization
    },
    {
        'id': '55011271',
        'center': [508.848621645943, 508.064917088557],
        'focal_length': [1149.67569986785, 1147.59161666764],
        'radial_distortion': [-0.194213629607385, 0.240408539138292, 0.00681997559022603],
        'tangential_distortion': [-0.0027408943961907, -0.001619026613787],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': -70,  # Only used for visualization
    },
    {
        'id': '58860488',
        'center': [519.815837182153, 501.402658888552],
        'focal_length': [1149.14071676148, 1148.7989685676],
        'radial_distortion': [-0.208338188251856, 0.255488007488945, -0.00246049749891915],
        'tangential_distortion': [-0.000759999321030303, 0.00148438698385668],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': 110,  # Only used for visualization
    },
    {
        'id': '60457274',
        'center': [514.968197319863, 501.882018537695],
        'focal_length': [1145.51133842318, 1144.77392807652],
        'radial_distortion': [-0.198384093827848, 0.218323676298049, -0.00894780704152122],
        'tangential_distortion': [-0.00181336200488089, -0.000587205583421232],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': -110,  # Only used for visualization
    }
]

h36m_cameras_extrinsic_params = {
    'S1': [
        {
            'translation': [-346.05078140028075, 546.9807793144001, 5474.481087434061],
            'R':
                [
                    [-0.9153617321513369, 0.40180836633680234, 0.02574754463350265],
                    [0.051548117060134555, 0.1803735689384521, -0.9822464900705729],
                    [-0.399319034032262, -0.8977836111057917, -0.185819527201491]
                ]
        },
        {
            'translation': [251.42516271750836, 420.9422103702068, 5588.195881837821],
            'R':
                [
                    [0.9281683400814921, 0.3721538354721445, 0.002248380248018696],
                    [0.08166409428175585, -0.1977722953267526, -0.976840363061605],
                    [-0.3630902204349604, 0.9068559102440475, -0.21395758897485287]
                ]
        },
        {
            'translation': [480.482559565337, 253.83237471361554, 5704.2076793704555],
            'R':
                [
                    [-0.9141549520542256, -0.4027780222811878, -0.045722952682337906],
                    [-0.04562341383935875, 0.21430849526487267, -0.9756999400261069],
                    [0.40278930937200774, -0.889854894701693, -0.214287280609606]
                ]
        },
        {
            'translation': [51.88347637559197, 378.4208425426766, 4406.149140878431],
            'R':
                [
                    [0.9141562410494211, -0.40060705854636447, 0.061905989962380774],
                    [-0.05641000739510571, -0.2769531972942539, -0.9592261660183036],
                    [0.40141783470104664, 0.8733904688919611, -0.2757767409202658]
                ]
        },
    ],
    'S5': [
        {
            'translation': [-219.3059666108619, 544.4787497640639, 5518.740477016156],
            'R':
                [
                    [-0.9042074184788829, 0.42657831374650107, 0.020973473936051274],
                    [0.06390493744399675, 0.18368565260974637, -0.9809055713959477],
                    [-0.4222855708380685, -0.8856017859436166, -0.1933503902128034]
                ]
        },
        {
            'translation': [103.90282067751986, 395.67169468951965, 5767.97265758172],
            'R':
                [
                    [0.9222116004775194, 0.38649075753002626, 0.012274293810989732],
                    [0.09333184463870337, -0.19167233853095322, -0.9770111982052265],
                    [-0.3752531555110883, 0.902156643264318, -0.21283434941998647]
                ]
        },
        {
            'translation': [520.3272318446208, 283.3690958234795, 5591.123958858676],
            'R':
                [
                    [-0.9258288614330635, -0.3728674116124112, -0.06173178026768599],
                    [-0.023578112500148365, 0.220000562347259, -0.9752147584905696],
                    [0.3772068291381898, -0.9014264506460582, -0.21247437993123308]
                ]
        },
        {
            'translation': [-79.116431351199, 425.59047114848386, 4454.481629705836],
            'R':
                [
                    [0.9222815489764817, -0.3772688722588351, 0.0840532119677073],
                    [-0.021177649402562934, -0.26645871124348197, -0.9636136478735888],
                    [0.3859381447632816, 0.88694303832152, -0.25373962085111357]
                ]
        },
    ],
    'S6': [
        {
            'translation': [-239.5182864132218, 545.8141831785044, 5523.931578633363],
            'R':
                [
                    [-0.9149503344107554, 0.4034864343564006, 0.008036345687245266],
                    [0.07174776353922047, 0.1822275975157708, -0.9806351824867137],
                    [-0.3971374371533952, -0.896655898321083, -0.19567845056940925]
                ]
        },
        {
            'translation': [169.02510061389722, 409.6671223380997, 5714.338002825065],
            'R':
                [
                    [0.9197364689900042, 0.39209901596964664, 0.018525368698999664],
                    [0.101478073351267, -0.19191459963948, -0.9761511087296542],
                    [-0.37919260045353465, 0.899681692667386, -0.21630030892357308]
                ]
        },
        {
            'translation': [521.9864793089763, 286.28272817103516, 5643.2724406159],
            'R':
                [
                    [-0.916577698818659, -0.39393483656788014, -0.06856140726771254],
                    [-0.01984531630322392, 0.21607069980297702, -0.9761760169700323],
                    [0.3993638509543854, -0.8933805444629346, -0.20586334624209834]
                ]
        },
        {
            'translation': [-56.29675276801464, 420.29579722027506, 4499.322693551688],
            'R':
                [
                    [0.9182950552949388, -0.3850769011116475, 0.09192372735651859],
                    [-0.015534985886560007, -0.26706146429979655, -0.9635542737695438],
                    [0.3955917790277871, 0.8833990913037544, -0.25122338635033875]
                ]
        },
    ],
    'S7': [
        {
            'translation': [-323.9118424584857, 541.7715234126381, 5506.569132699328],
            'R':
                [
                    [-0.9055764231419416, 0.42392653746206904, 0.014752378956221508],
                    [0.06862812683752326, 0.18074371881263407, -0.9811329615890764],
                    [-0.41859469903024304, -0.8874784498483331, -0.19277053457045695]
                ]
        },
        {
            'translation': [178.6238708832376, 403.59193467821774, 5694.8801003668095],
            'R':
                [
                    [0.9212640765077017, 0.3886011826562522, 0.01617473877914905],
                    [0.09922277503271489, -0.1946115441987536, -0.9758489574618522],
                    [-0.3760682680727248, 0.9006194910741931, -0.21784671226815075]
                ]
        },
        {
            'translation': [441.1064712697594, 271.91614362573955, 5660.120611352617],
            'R':
                [
                    [-0.9245069728829368, -0.37555597339631824, -0.06515034871105972],
                    [-0.018955014220249332, 0.21601110989507338, -0.9762068980691586],
                    [0.38069353097569036, -0.9012751584550871, -0.20682244613440448]
                ]
        },
        {
            'translation': [25.768533743836343, 431.05581759025813, 4461.872981411145],
            'R':
                [
                    [0.9228353966173104, -0.3744001545228767, 0.09055029013436408],
                    [-0.014982084363704698, -0.269786590656035, -0.9628035794752281],
                    [0.3849030629889691, 0.8871525910436372, -0.25457791009093983]
                ]
        },
    ],
    'S8': [
        {
            'translation': [-82.70216069652597, 552.1896311377282, 5557.353609418419],
            'R':
                [
                    [-0.9115694669712032, 0.4106494283805017, 0.020202818036194434],
                    [0.060907749548984036, 0.1834736632003901, -0.9811359034082424],
                    [-0.40660958293025334, -0.8931430243150293, -0.19226072190306673]
                ]
        },
        {
            'translation': [-209.06289992510443, 375.0691429434037, 5818.276676972416],
            'R':
                [
                    [0.931016282525616, 0.3647626932499711, 0.01252434769597448],
                    [0.08939715221301257, -0.19463753190599434, -0.9767929055586687],
                    [-0.35385990285476776, 0.9105297407479727, -0.2138194574051759]
                ]
        },
        {
            'translation': [623.0985110132146, 290.9053651845054, 5534.379001592981],
            'R':
                [
                    [-0.9209075762929309, -0.3847355178017309, -0.0625125368875214],
                    [-0.02568138180824641, 0.21992027027623712, -0.9751797482259595],
                    [0.38893405939143305, -0.8964450100611084, -0.21240678280563546]
                ]
        },
        {
            'translation': [-178.36705625795474, 423.4669232560848, 4421.6448791590965],
            'R':
                [
                    [0.927667052235436, -0.3636062759574404, 0.08499597802942535],
                    [-0.01666268768012713, -0.26770413351564454, -0.9633570738505596],
                    [0.37303645269074087, 0.8922583555131325, -0.2543989622245125]
                ]
        },
    ],
    'S9': [
        {
            'translation': [-321.2078335720134, 467.13452033013084, 5514.330338522134],
            'R':
                [
                    [-0.9033486204435297, 0.4269119782787646, 0.04132109321984796],
                    [0.04153061098352977, 0.182951140059007, -0.9822444139329296],
                    [-0.4268916470184284, -0.8855930460167476, -0.18299857527497945]
                ]
        },
        {
            'translation': [19.193095609487138, 404.22842728571936, 5702.169280033924],
            'R':
                [
                    [0.9315720471487059, 0.36348288012373176, -0.007329176497134756],
                    [0.06810069482701912, -0.19426747906725159, -0.9785818524481906],
                    [-0.35712157080642226, 0.911120377575769, -0.20572758986325015]
                ]
        },
        {
            'translation': [455.40107288876885, 273.3589338272866, 5657.814488280711],
            'R':
                [
                    [-0.9269344193869241, -0.3732303525241731, -0.03862235247246717],
                    [-0.04725991098820678, 0.218240494552814, -0.9747500127472326],
                    [0.37223525218497616, -0.901704048173249, -0.21993345934341726]
                ]
        },
        {
            'translation': [-69.271255294384, 422.1843366088847, 4457.893374979773],
            'R':
                [
                    [0.915460708083783, -0.39734606500700814, 0.06362229623477154],
                    [-0.04940628468469528, -0.26789167566119776, -0.9621814117644814],
                    [0.39936288133525055, 0.8776959352388969, -0.26487569589663096]
                ]
        },
    ],
    'S11': [
        {
            'translation': [-234.7208032216618, 464.34018262882194, 5536.652631113797],
            'R':
                [
                    [-0.9059013006181885, 0.4217144115102914, 0.038727105014486805],
                    [0.044493184429779696, 0.1857199061874203, -0.9815948619389944],
                    [-0.4211450938543295, -0.8875049698848251, -0.1870073216538954]
                ]
        },
        {
            'translation': [-11.934348472090557, 449.4165893644565, 5541.113551868937],
            'R':
                [
                    [0.9216646531492915, 0.3879848687925067, -0.0014172943441045224],
                    [0.07721054863099915, -0.18699239961454955, -0.979322405373477],
                    [-0.3802272982247548, 0.9024974149959955, -0.20230080971229314]
                ]
        },
        {
            'translation': [781.127357651581, 235.3131620173424, 5576.37044019807],
            'R':
                [
                    [-0.9063540572469627, -0.42053101768163204, -0.04093880896680188],
                    [-0.0603212197838846, 0.22468715090881142, -0.9725620980997899],
                    [0.4181909532208387, -0.8790161246439863, -0.2290130547809762]
                ]
        },
        {
            'translation': [-155.13650339749012, 422.16256306729633, 4435.416222660868],
            'R':
                [
                    [0.91754082476548, -0.39226322025776267, 0.06517975852741943],
                    [-0.04531905395586976, -0.26600517028098103, -0.9629057236990188],
                    [0.395050652748768, 0.8805514269006645, -0.2618476013752581]
                ]
        },
    ],
}

def mkdirs(path):
    """

    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        return

def init_camera_h36m():
    """

    :return:
    """
    _cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
    for cameras in _cameras.values():
        for i, cam in enumerate(cameras):
            cam.update(h36m_cameras_intrinsic_params[i])
            for k, v in cam.items():
                if k not in ['id', 'res_w', 'res_h']:
                    cam[k] = np.array(v, dtype='float32')

            # Normalize camera frame
            if 'translation' in cam:
                cam['translation'] = cam['translation'] / 1000  # mm to meters

    camera_info = dict()
    for subject in _cameras:
        camera_info.setdefault(subject, list())
        for cam in _cameras[subject]:
            if 'translation' not in cam:
                continue
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = cam['focal_length'][0]
            K[1, 1] = cam['focal_length'][1]
            K[0, 2] = cam['center'][0]
            K[1, 2] = cam['center'][1]

            R = cam['R']
            dist_coeff = np.concatenate(
                (cam['radial_distortion'][:2], cam['tangential_distortion'], cam['radial_distortion'][2:])
            ).astype(np.float32).reshape((5,))

            t = np.array(cam['translation'], dtype=np.float64).reshape(3, 1)
            camera_info[subject].append(CameraInfoPacket(P=None, K=K, R=R, t=t,
                                                         res_w=cam['res_w'], res_h=cam['res_h'],
                                                         azimuth=cam['azimuth'], dist_coeff=dist_coeff))

    return camera_info

def get_camera_pose(camera_info):
    """

    :param camera_info:
    :return:
    """
    camera_pose = list()
    for sbj in camera_info.keys():
        for i, cam in enumerate(camera_info[sbj]):
            camera_pose.append(cam.cam_orig_world.reshape(3))

    return np.array(camera_pose)

def camera_translation(T, t, distance_ratio=1):
    """

    :param T:
    :param t:
    :param distance_ratio:
    :return:
    """
    return (T - t) * distance_ratio + t

def convertdegree2euler(degree):
    """

    :param degree:
    :return:
    """
    return degree / 180.0 * np.pi

def rotate_mat(axis, radian):
    """

    :param axis:
    :param radian:
    :return:
    """
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix

def rotate_camera(R, T, center, axis, radian):
    """

    :param R:
    :param T:
    :param center:
    :param axis:
    :param radian:
    :return:
    """
    Rw2c = R
    Tw2c = T

    Rc2w = Rw2c.T
    Tc2w = -np.dot(Rc2w, Tw2c)

    matrix = rotate_mat(axis, radian)
    new_Rc2w = np.dot(matrix, Rc2w)
    new_Tc2w = np.dot(matrix, Tc2w - center) + center

    new_Rw2c = new_Rc2w.T
    new_Tw2c = -np.dot(new_Rw2c, new_Tc2w)
    return new_Rw2c, new_Tw2c

def get_intrinsic(cam_idx, fx_bais=0, fy_bais=0, cx_bias=0, cy_bias=0):
    """

    :param cam_idx:
    :return:
    """
    intrinsic = h36m_cameras_intrinsic_params[cam_idx]
    dist_coeff = np.array(
        intrinsic['radial_distortion'][:2] + intrinsic['tangential_distortion'] + intrinsic['radial_distortion'][2:],
        dtype=np.float64
    ).reshape((5,))
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = intrinsic['focal_length'][0] + fx_bais
    K[1, 1] = intrinsic['focal_length'][1] + fy_bais
    K[0, 2] = intrinsic['center'][0] + cx_bias
    K[1, 2] = intrinsic['center'][1] + cy_bias
    return K, dist_coeff

def check_in_frame(kpt_2d, res_w=1000, res_h=1000):
    """

    :param kpt_2d:
    :param res_w:
    :param res_h:
    :return:
    """
    if np.sum(kpt_2d[..., 0] < 0) < 1 \
            and np.sum(kpt_2d[..., 0] > res_w) < 1 \
            and np.sum(kpt_2d[..., 1] < 0) < 1 \
            and np.sum(kpt_2d[..., 1] > res_h) < 1:
            return True
    else:
        return False

def distortPoint(kpts2d_reprj, K, dist_coeff):
    """

    :param kpts2d_reprj: numpy array with shape of (2, N)
    :param K: intrinsic matrix with shape of (3, 3)
    :param dist_coeff: distortion coefficient with shape of (1, 5)
    :return:
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    k1 = dist_coeff[0, 0]
    k2 = dist_coeff[0, 1]
    p1 = dist_coeff[0, 2]
    p2 = dist_coeff[0, 3]
    k3 = dist_coeff[0, 4]

    kpts2d_reprj_copy = kpts2d_reprj.copy()
    batch_size = kpts2d_reprj_copy.shape[1]
    for batch_idx in range(batch_size):

        # To relative coordinates
        x = (kpts2d_reprj_copy[0, batch_idx] - cx) / fx
        y = (kpts2d_reprj_copy[1, batch_idx] - cy) / fy

        r2 = x*x + y*y

        # Radial distortion
        xDistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        yDistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

        # Tangential distortion
        xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
        yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

        # Back to absolute coordinates
        kpts2d_reprj[0, batch_idx] = xDistort * fx + cx
        kpts2d_reprj[1, batch_idx] = yDistort * fy + cy

    return kpts2d_reprj

if __name__ == '__main__':
    # CAMERA AUGMENTATION FOR THE FOLLOWING CASES:
    # (1) ROTATION -> VIEWPOINT OF THE CAMERA
    # (2) TRANSLATION -> DISTANCE BETWEEN THE CAMERA AND THE SUBJECT, HEIGHT OF THE CAMERA, 2D SCALE OF THE PERSON
    # (3) PITCH OF THE CAMERA

    random_seed = 0
    np.random.seed(random_seed)

    # input
    FILE_TO_3D_POSE = '/ssd/yzhan/data/benchmark/3D/h36m/annotations/gt/data_3d_h36m.npz'

    # output
    FILE_TO_3D_POSE_PERSON_SCALE_TEST = '/ssd/ray3d/data_3d_h36m_aug_test.npz'
    PATH_TO_OUTPUT = '/ssd/ray3d/'

    PLOT_PATH = os.path.join(PATH_TO_OUTPUT, 'visualization.pdf')
    PERSON_PATH = os.path.join(PATH_TO_OUTPUT, 'person.pdf')
    pose_3d = np.load(FILE_TO_3D_POSE, allow_pickle=True)['positions_3d'].item()
    mkdirs(PATH_TO_OUTPUT)
    # ------------------------         PERSON SCALE         ------------------------
    INDICES = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16)
    ]
    COLORS = [
        (0 / 255., 215 / 255., 255 / 255.), (0 / 255., 255 / 255., 204 / 255.),
        (0 / 255., 134 / 255., 255 / 255.), (0 / 255., 255 / 255., 50 / 255.),
        (77 / 255., 255 / 255., 222 / 255.), (77 / 255., 196 / 255., 255 / 255.),
        (77 / 255., 135 / 255., 255 / 255.), (191 / 255., 255 / 255., 77 / 255.),
        (77 / 255., 255 / 255., 77 / 255.), (0 / 255., 127 / 255., 255 / 255.),
        (255 / 255., 127 / 255., 77 / 255.), (0 / 255., 77 / 255., 255 / 255.),
        (77 / 255., 255 / 255., 77 / 255.), (0 / 255., 127 / 255., 255 / 255.),
        (255 / 255., 127 / 255., 77 / 255.), (0 / 255., 77 / 255., 255 / 255.),
    ]
    def drawer_3d(ax, viz_3d, cidx):
        ax.scatter(viz_3d[0], viz_3d[1], viz_3d[2])
        for ii, idx in enumerate(INDICES):
            ax.plot(
                np.concatenate([viz_3d[0][idx[0]].reshape(1), viz_3d[0][idx[1]].reshape(1)]),
                np.concatenate([viz_3d[1][idx[0]].reshape(1), viz_3d[1][idx[1]].reshape(1)]),
                np.concatenate([viz_3d[2][idx[0]].reshape(1), viz_3d[2][idx[1]].reshape(1)]),
                color=COLORS[cidx][::-1]
            )

    PERSON_RANGE_TEST = [-0.4, -0.3, -0.2, -0.1, 0.1]
    pose_3d_augmented_test = dict()
    for ratio in PERSON_RANGE_TEST:
       pose_3d_origin = copy.deepcopy(pose_3d)

       for sbj in pose_3d_origin.keys():
           new_sbj = '{}_{}'.format(sbj, 1 + ratio)
           pose_3d_augmented_test[new_sbj] = copy.deepcopy(pose_3d_origin[sbj])
           action_to_pose = pose_3d_augmented_test[new_sbj]

           for act in action_to_pose.keys():
               poses = action_to_pose[act]
               valid_pose_3d = poses[:, H36M_KPT_IDX]
               valid_pose_3d_copy = poses[:, H36M_KPT_IDX].copy()

               batch_size, num_kpt, feat_dim = valid_pose_3d.shape
               valid_pose_3d = torch.from_numpy(valid_pose_3d).view(batch_size, 1, num_kpt, feat_dim)

               root_origin = valid_pose_3d[:, :, 0:1, :]
               bone_length = get_bone_length_from_3d_pose(valid_pose_3d)
               bone_unit_vect = get_bone_unit_vector_from_3d_pose(valid_pose_3d)
               bone_length_ratio = torch.tensor([ratio]).float().repeat([batch_size, 1, num_kpt - 1, 1])
               modified_bone_length = torch.mul(bone_length, bone_length_ratio) + bone_length
               modifyed_bone = bone_unit_vect * modified_bone_length

               modifyed_pose_3d = get_3d_pose_from_bone_vector(modifyed_bone, root_origin).view(batch_size, num_kpt,
                                                                                                feat_dim).numpy()
               modifyed_pose_3d[:, :, 2:] -= np.min(modifyed_pose_3d[:, :, 2:], axis=1, keepdims=True)
               modifyed_pose_3d[:, :, 2:] += np.min(valid_pose_3d_copy[:, :, 2:], axis=1, keepdims=True)
               action_to_pose[act][:, H36M_KPT_IDX] = modifyed_pose_3d

    pose_3d_augmented_test.update(pose_3d)
    np.savez(FILE_TO_3D_POSE_PERSON_SCALE_TEST, positions_3d=pose_3d_augmented_test)

    # ------------------------ ROTATION, TRANSLATION, PITCH ------------------------
    CENTER_POINT = [0, 0, 1.8]  # the origin coordinate for the camera augmentation, in which the subject stands

    camera_info = init_camera_h36m()
    camera_pose = get_camera_pose(camera_info)

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(CENTER_POINT[0], CENTER_POINT[1], CENTER_POINT[2], color='y', s=20)  # center point
    ax.scatter(camera_pose[:, 0], camera_pose[:, 1], camera_pose[:, 2], color='g', s=20)  # original camera pose

    AUGMENTATION_CONFIG = {
        'Train': [
            [60, 180, 300],
            [2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
            [-26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
        ],
        'Rotation':[
            [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
            [2.0, 3.0],
            [-10, 0, 10]
        ],
        'Translation':[
            [60, 180, 300],
            [1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9],
            [0]
        ],
        'Pitch':[
            [60, 180, 300],
            [2.0],
            [-49, -45, -41, -37, -33, -29, -25, -21, -17, -15, -13, -9, -5, -1, 3, 5, 9, 11]
        ],
        'Scale':
        [
            [60, 180, 300],
            [2.0],
            [0]
        ]
    }

    # h36m has four initial camera positions, iterate them
    # augmentation for rotation
    sbj = TRAIN_SUBJECTS[0]  # 'S1'
    cam_idx = 1  # '55011271'
    distort = False
    for set_type in AUGMENTATION_CONFIG.keys():
        ROTATION_RANGE = AUGMENTATION_CONFIG[set_type][0]
        TRANSLATION_RANGE = AUGMENTATION_CONFIG[set_type][1]
        PITCH_RANGE = AUGMENTATION_CONFIG[set_type][2]

        for yaw_degree in ROTATION_RANGE:

            # augmentation for translation
            for dist_ratio in TRANSLATION_RANGE:

                # augmentation for pitch
                for pitch_degree in PITCH_RANGE:
                    CAMERA_PARAM_PATH = os.path.join(PATH_TO_OUTPUT, '{}/{}'.format(set_type, 'json'))
                    mkdirs(CAMERA_PARAM_PATH)

                    CAMERA_FILENAME = os.path.join(CAMERA_PARAM_PATH,
                                                   'TRANSLATION{}_YAW{}_PITCH{}_CAM{}.json'.format(dist_ratio,
                                                                                                   yaw_degree,
                                                                                                   pitch_degree,
                                                                                                   cam_idx))
                    if os.path.exists(CAMERA_FILENAME):
                        print('{} already exists, skip '.format(CAMERA_FILENAME))
                        continue


                    selected_cam = copy.deepcopy(camera_info[sbj][cam_idx])
                    Rw2c = selected_cam.Rw2c
                    Tw2c = selected_cam.Tw2c
                    t = np.array(CENTER_POINT).reshape(3, 1)

                    # translation
                    Tw2c_translation_aumented = camera_translation(Tw2c, t, dist_ratio)

                    # rotation
                    yaw = convertdegree2euler(yaw_degree)
                    Rw2c_rotation_augmented, Tw2c_rotation_augmented = rotate_camera(Rw2c, Tw2c_translation_aumented, t,
                                                                                     np.array(AXIS_Z), yaw)

                    # pitch
                    pitch = convertdegree2euler(pitch_degree)
                    camera_position = - np.dot(Rw2c_rotation_augmented.T, Tw2c_rotation_augmented)

                    axis = np.array([-camera_position[1][0], camera_position[0][0], 0])
                    Rw2c_pitch_augmented, Tw2c_pitch_augmented = rotate_camera(Rw2c_rotation_augmented,
                                                                               Tw2c_rotation_augmented, t, axis,
                                                                               pitch)
                    camera_position = - np.dot(Rw2c_pitch_augmented.T, Tw2c_pitch_augmented)
                    K, dist_coeff = get_intrinsic(cam_idx)
                    camera_augmented = CameraInfoPacket(P=None, K=K, R=copy.deepcopy(Rw2c_pitch_augmented),
                                                        t=copy.deepcopy(Tw2c_pitch_augmented),
                                                        dist_coeff=dist_coeff)

                    cameras_params = []
                    camera_param = {}
                    camera_param.update(h36m_cameras_intrinsic_params[cam_idx])
                    camera_param['R'] = camera_augmented.Rw2c.tolist()
                    camera_param['translation'] = camera_augmented.Tw2c.tolist()

                   # save camera parameter
                    with open(CAMERA_FILENAME, 'w') as file_obj:
                        json.dump([camera_param], file_obj, indent=4)

    # The reason we split camera json saving with 2D npz saving in two loops
    # is for memory concern

    for set_type in AUGMENTATION_CONFIG.keys():
        ROTATION_RANGE = AUGMENTATION_CONFIG[set_type][0]
        TRANSLATION_RANGE = AUGMENTATION_CONFIG[set_type][1]
        PITCH_RANGE = AUGMENTATION_CONFIG[set_type][2]

        for yaw_degree in ROTATION_RANGE:

            # augmentation for translation
            for dist_ratio in TRANSLATION_RANGE:

                # augmentation for pitch
                for pitch_degree in PITCH_RANGE:

                    CAMERA_PARAM_PATH = os.path.join(PATH_TO_OUTPUT, '{}/{}'.format(set_type, 'json'))
                    CAMERA_FILENAME = os.path.join(CAMERA_PARAM_PATH,
                                                   'TRANSLATION{}_YAW{}_PITCH{}_CAM{}.json'.format(dist_ratio,
                                                                                                   yaw_degree,
                                                                                                   pitch_degree,
                                                                                                   cam_idx))

                    POSE_2D_PATH = os.path.join(PATH_TO_OUTPUT, '{}/{}'.format(set_type, 'npz'))
                    mkdirs(POSE_2D_PATH)

                    # save projected 2d pose
                    POSE2D_FILENAME = os.path.join(POSE_2D_PATH,
                                                   'TRANSLATION{}_YAW{}_PITCH{}_CAM{}.npz'.format(dist_ratio,
                                                                                                  yaw_degree,
                                                                                                  pitch_degree,
                                                                                                  cam_idx))
                    if os.path.exists(POSE2D_FILENAME):
                        print('POSE2D_FILENAME {} already exists! skip'.format(POSE2D_FILENAME))
                        continue

                    camera = json.load(open(CAMERA_FILENAME, 'r'))[0]
                    for k, v in camera.items():
                        if k not in ['id', 'res_w', 'res_h', 'azimuth', 'translation_scale', 'degree', 'pitch']:
                            camera[k] = np.array(v, dtype='float32')

                    K = np.eye(3, dtype=np.float64)
                    K[0, 0] = camera['focal_length'][0]
                    K[1, 1] = camera['focal_length'][1]
                    K[0, 2] = camera['center'][0]
                    K[1, 2] = camera['center'][1]

                    R = camera['R']
                    dist_coeff = np.concatenate(
                        (camera['radial_distortion'][:2], camera['tangential_distortion'],
                         camera['radial_distortion'][2:])
                    ).reshape((5,))

                    t = np.array(camera['translation'])
                    camera_augmented = CameraInfoPacket(P=None, K=K, R=R,
                                                        t=t,
                                                        dist_coeff=dist_coeff)

                    # sanity check
                    if set_type == 'Scale':
                        pose_3d_augmented = copy.deepcopy(pose_3d_augmented_test)
                    else:
                        pose_3d_augmented = copy.deepcopy(pose_3d)

                    invalid = False
                    pose_2d = {}
                    for sbj in pose_3d_augmented.keys():
                        pose_2d.setdefault(sbj, dict())
                        acts = list(pose_3d_augmented[sbj].keys())
                        for act_idx in range(len(acts)):
                            act = acts[act_idx]
                            pose_2d[sbj].setdefault(act, list())
                            kpt_3d = pose_3d_augmented[sbj][act][:, H36M_KPT_IDX]
                            kpt_3d_hom = catesian2homogenous(kpt_3d)
                            kpt_2d = copy.deepcopy(camera_augmented).project(kpt_3d_hom)
                            if distort:
                                kpt_2d_distorted = np.zeros_like(kpt_2d)
                                for fidx in range(kpt_2d_distorted.shape[0]):
                                    kpt_2d_distorted[fidx] = distortPoint(kpt_2d[fidx].T, camera_augmented.K, camera_augmented.dist_coeff.reshape(1, -1)).T
                                kpt_2d = kpt_2d_distorted
                            pose_2d[sbj][act].append(kpt_2d)
                            if not check_in_frame(kpt_2d, RES_W, RES_H):
                                invalid = True
                                break
                        if invalid:
                            break
                    if invalid:
                        del pose_2d
                        print('remove invalid pose config {}'.format(CAMERA_FILENAME))
                        cmd = 'rm {}'.format(CAMERA_FILENAME)
                        os.system(cmd)
                        continue
                    else:

                        #if set_type == 'TRAIN':
                        #    ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='r', s=10)
                        #elif set_type == 'Rotation':
                        #    ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='b', s=10)
                        #elif set_type == 'Translation':
                        #    ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='y', s=10)
                        #elif set_type == 'Pitch':
                        #    ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='g', s=10)
                        #elif set_type == 'Scale':
                        #    ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='k', s=10)

                        print('\tYAW: {}, DISTANCE: {}, PITCH: {}'.format(yaw_degree, dist_ratio, pitch_degree))


                        METADATA = {
                            'layout': 'h36m_aug',
                            'num_joints': 17,
                            'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]
                        }
                        np.savez(POSE2D_FILENAME, metadata=METADATA, positions_2d=pose_2d)
                        del pose_2d

    plt.show()
    plt.savefig(PLOT_PATH)
    plt.close()
