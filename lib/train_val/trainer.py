import os
import torch
import shutil
import numpy as np

from lib.loss.loss import mpjpe, n_mpjpe, p_mpjpe, mean_velocity_error, weighted_mpjpe
from lib.dataloader.generators import UnchunkedGenerator
from lib.camera.camera import image_coordinates
from lib.skeleton.bone import get_bone_length_from_3d_pose, get_bone_unit_vector_from_3d_pose


class Trainer():
    def __init__(self,
                 data_config, model_config, train_config, plot_config,
                 train_generator, test_generator,
                 models, optimizer,
                 kps_left, kps_right, joints_left, joints_right, plotter, best_performance):

        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.plot_config = plot_config

        self.lr = train_config['LEARNING_RATE']
        self.optimizer = optimizer

        self.train_generator = train_generator
        self.test_generator = test_generator

        self.pos_model_train = models['train_pos']
        self.pos_model_test = models['test_pos']
        self.trj_model_train = models['train_trj']
        self.trj_model_test = models['test_trj']

        self.min_loss = 1e5 if best_performance is None else best_performance
        self.losses_3d_train = []
        self.losses_3d_valid = []

        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.receptive_field = model_config['NUM_FRAMES']

        self.plotter = plotter

    @staticmethod
    def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
        inputs_2d_p = inputs_2d
        if inputs_3d is not None:
            inputs_3d_p = inputs_3d.unsqueeze(dim=1)
        else:
            inputs_3d_p = inputs_3d
        out_num = inputs_2d_p.shape[0] - receptive_field + 1
        eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
        for i in range(out_num):
            eval_input_2d[i, :, :, :] = inputs_2d_p[i:i + receptive_field, :, :]
        return eval_input_2d, inputs_3d_p

    def train(self, epoch, mlog):

        N = 0
        epoch_loss_3d_train = 0
        epoch_loss_3d_pos = 0
        epoch_loss_3d_trj = 0
        epoch_loss_3d_bone = 0

        self.pos_model_train.train()
        if self.model_config['TRAJECTORY_MODEL']:
            self.trj_model_train.train()

        camera_info = self.train_generator.dataset.get_camera_info()
        iter = 0
        for i, train_data in enumerate(self.train_generator):
            inputs_3d, inputs_2d, cam_indices = train_data
            cam_indices = cam_indices.numpy()
            inputs_param = np.zeros((cam_indices.shape[0], 2))
            for cidx in range(cam_indices.shape[0]):
                cam = camera_info[cam_indices[cidx]]
                inputs_param[cidx, 0] = (-cam.Rw2c.T @ cam.Tw2c)[2][0]
                inputs_param[cidx, 1] = cam.cam_pitch_rad
            inputs_param = torch.from_numpy(inputs_param.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_3d = inputs_3d.cuda()
                inputs_param = inputs_param.cuda()

            if self.model_config['TRAJECTORY_MODEL']:
                inputs_traj = inputs_3d[:, :, :1].clone()

            if self.data_config['RAY_ENCODING']:
                # do nothing
                if self.model_config['TRAJECTORY_MODEL']:
                    inputs_3d[:, :, 1:] -= inputs_3d[:, :, 0:1]
                    inputs_3d[:, :, 0] = 0
            else:
                inputs_3d[:, :, 1:] -= inputs_3d[:, :, 0:1]
                inputs_3d[:, :, 0] = 0

            self.optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos = self.pos_model_train(inputs_2d, inputs_param)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            total_loss = loss_3d_pos

            if self.model_config['BONE_COMPARISON']:
                predicted_bone_length = get_bone_length_from_3d_pose(predicted_3d_pos)
                target_bone_length = get_bone_length_from_3d_pose(inputs_3d)
                loss_3d_bone_length = mpjpe(predicted_bone_length, target_bone_length)

                predicted_bone_unit_vector = get_bone_unit_vector_from_3d_pose(predicted_3d_pos)
                target_bone_unit_vector = get_bone_unit_vector_from_3d_pose(inputs_3d)
                loss_3d_bone_angle = mpjpe(predicted_bone_unit_vector, target_bone_unit_vector)

                epoch_loss_3d_bone += inputs_3d.shape[0] * inputs_3d.shape[1] * (loss_3d_bone_length.item() + loss_3d_bone_angle.item())
                total_loss += (loss_3d_bone_length + loss_3d_bone_angle)

            if self.model_config['TRAJECTORY_MODEL']:
                predicted_3d_trj = self.trj_model_train(inputs_2d, inputs_param)
                w = torch.abs(1 / inputs_traj[:, :, :, 2])  # Weight inversely proportional to depth
                loss_3d_traj = weighted_mpjpe(predicted_3d_trj, inputs_traj, w)
                assert inputs_traj.shape[0] * inputs_traj.shape[1] == inputs_3d.shape[0] * inputs_3d.shape[1]
                epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_traj.item()
                epoch_loss_3d_trj += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_traj.item()
                total_loss += loss_3d_traj

            # ---------------- visualization ---------------- #
            if iter % 2048 == 0 and self.plotter is not None and epoch % 64 == 0:
                self.plotter.show_plot(
                    epoch, inputs_2d.detach().cpu().numpy(),
                    inputs_3d.detach().cpu().numpy(),
                    predicted_3d_pos.detach().cpu().numpy(),
                    dataset=self.data_config['DATASET'],
                    gt=self.data_config['KEYPOINTS']
                )
            # ---------------- visualization ---------------- #

            iter += 1

            total_loss.backward()
            self.optimizer.step()

            if i % 20 == 0:
                mlog.info('({})/({}) loss: {}'.format(i + 1, len(self.train_generator), epoch_loss_3d_train / N))

        self.losses_3d_train.append(epoch_loss_3d_train / N)
        torch.cuda.empty_cache()

        if self.plotter:
            # plot all the losses
            self.plotter.log_metric('train', self.losses_3d_train[-1] * 1000, epoch)
            self.plotter.log_metric('train_pos', epoch_loss_3d_pos / N * 1000, epoch)
            self.plotter.log_metric('train_trj', epoch_loss_3d_trj / N * 1000, epoch)
            self.plotter.log_metric('train_bone', epoch_loss_3d_bone / N * 1000, epoch)

            # plot all the learning rates
            self.plotter.log_metric('lr', self.lr, epoch)

        # return the current epoch's mpjme
        return self.losses_3d_train[-1], self.lr

    def test(self, epoch, mlog):

        with torch.no_grad():
            self.pos_model_test.load_state_dict(self.pos_model_train.state_dict(), strict=True)
            self.pos_model_test.eval()
            if self.model_config['TRAJECTORY_MODEL']:
                self.trj_model_test.load_state_dict(self.trj_model_train.state_dict(), strict=True)
                self.trj_model_test.eval()

            epoch_loss_3d_valid = 0
            epoch_loss_3d_pos = 0
            epoch_loss_3d_trj = 0
            epoch_loss_3d_bone = 0
            N = 0
            camera_info = self.test_generator.get_camera_info()
            # Evaluate on test set
            for i, test_data in enumerate(self.test_generator):
                inputs_3d, inputs_2d, cidx = test_data

                cam = camera_info[cidx]
                cam_param = np.array([(-cam.Rw2c.T @ cam.Tw2c)[2][0], cam.cam_pitch_rad]).astype('float32')

                ##### convert size
                inputs_2d, inputs_3d = self.eval_data_prepare(self.receptive_field, inputs_2d, inputs_3d)
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_3d = inputs_3d.cuda()


                if self.model_config['TRAJECTORY_MODEL']:
                    inputs_traj = inputs_3d.clone()

                if self.data_config['RAY_ENCODING']:
                    # do nothing
                    if self.model_config['TRAJECTORY_MODEL']:
                        inputs_3d[:, :, 1:] -= inputs_3d[:, :, 0:1]
                        inputs_3d[:, :, 0] = 0
                else:
                    inputs_3d[:, :, 1:] -= inputs_3d[:, :, 0:1]
                    inputs_3d[:, :, 0] = 0
                inputs_param = torch.from_numpy(np.tile(cam_param, (inputs_2d.shape[0], 1)))
                if torch.cuda.is_available():
                    inputs_param = inputs_param.cuda()
                predicted_3d_pos = self.pos_model_test(inputs_2d, inputs_param)
                epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * mpjpe(predicted_3d_pos, inputs_3d).item()

                if self.model_config['BONE_COMPARISON']:
                    predicted_bone_length = get_bone_length_from_3d_pose(predicted_3d_pos)
                    target_bone_length = get_bone_length_from_3d_pose(inputs_3d)
                    loss_3d_bone_length = mpjpe(predicted_bone_length, target_bone_length)

                    predicted_bone_unit_vector = get_bone_unit_vector_from_3d_pose(predicted_3d_pos)
                    target_bone_unit_vector = get_bone_unit_vector_from_3d_pose(inputs_3d)
                    loss_3d_bone_angle = mpjpe(predicted_bone_unit_vector, target_bone_unit_vector)

                    epoch_loss_3d_bone += inputs_3d.shape[0] * inputs_3d.shape[1] * (loss_3d_bone_length.item() + loss_3d_bone_angle.item())

                if self.model_config['TRAJECTORY_MODEL']:
                    predicted_3d_trj = self.trj_model_test(inputs_2d, inputs_param)
                    predicted_3d_pos += predicted_3d_trj
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_traj)
                    w = torch.abs(1 / inputs_traj[:, :, 0, 2])
                    epoch_loss_3d_trj += inputs_3d.shape[0] * inputs_3d.shape[1] * weighted_mpjpe(predicted_3d_trj, inputs_traj[:, :, 0:1], w).item()
                else:
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)

                epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                if i % 20 == 0:
                    mlog.info('({})/({}) loss: {}'.format(i + 1, len(self.test_generator), epoch_loss_3d_valid / N))

            self.losses_3d_valid.append(epoch_loss_3d_valid / N)

            # Save checkpoint if necessary
            if epoch % self.train_config['CHECKPOINT_FREQUENCY'] == 0:
                chk_path = os.path.join(self.train_config['CHECKPOINT'], 'epoch_{}.bin'.format(epoch))
                mlog.info('Saving epochs {}\'s checkpoint to {}.'.format(epoch, chk_path))
                if self.model_config['TRAJECTORY_MODEL']:
                    torch.save({
                        'epoch': epoch,
                        'lr': self.lr,
                        'best_performance': self.losses_3d_valid[-1] * 1000 if self.losses_3d_valid[-1] * 1000 < self.min_loss else self.min_loss,
                        'random_state': np.random.RandomState(1234),
                        'optimizer': self.optimizer.state_dict(),
                        'model_pos': self.pos_model_train.state_dict(),
                        'model_trj': self.trj_model_train.state_dict()
                    }, chk_path)
                else:
                    torch.save({
                        'epoch': epoch,
                        'lr': self.lr,
                        'best_performance': self.losses_3d_valid[-1] * 1000 if self.losses_3d_valid[-1] * 1000 < self.min_loss else self.min_loss,
                        'random_state': np.random.RandomState(1234),
                        'optimizer': self.optimizer.state_dict(),
                        'model_pos': self.pos_model_train.state_dict(),
                    }, chk_path)

                #### save best checkpoint
                best_chk_path = os.path.join(self.train_config['CHECKPOINT'], 'best_epoch.bin'.format(epoch))
                if self.losses_3d_valid[-1] * 1000 < self.min_loss:
                    self.min_loss = self.losses_3d_valid[-1] * 1000
                    mlog.info('Saving best checkpoint to {} with mpjpe: {}.'.format(best_chk_path, self.min_loss))
                    shutil.copy(chk_path, best_chk_path)

                cmd = 'rm {}'.format(chk_path)
                os.system(cmd)

            # Decay learning rate exponentially
            self.lr *= self.train_config['LR_DECAY']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.train_config['LR_DECAY']

            # Decay BatchNorm momentum
            if self.model_config['MODEL'] == 'VideoPose3D':
                momentum = self.train_config['INITIAL_MOMENTUM'] * np.exp(-(epoch-1) / self.train_config['EPOCHS'] * np.log(self.train_config['INITIAL_MOMENTUM'] / self.train_config['FINAL_MOMENTUM']))
                self.pos_model_train.module.set_bn_momentum(momentum)
                if self.model_config['TRAJECTORY_MODEL']:
                    self.trj_model_train.module.set_bn_momentum(momentum)

        if self.plotter:
            # plot all the losses
            self.plotter.log_metric('test', self.losses_3d_valid[-1] * 1000, epoch)
            self.plotter.log_metric('test_pos', epoch_loss_3d_pos / N * 1000, epoch)
            self.plotter.log_metric('test_trj', epoch_loss_3d_trj / N * 1000, epoch)
            self.plotter.log_metric('test_bone', epoch_loss_3d_bone / N * 1000, epoch)

        # return the current epoch's mpjme
        return self.losses_3d_valid[-1]
