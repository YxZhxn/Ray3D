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
        inputs_2d_p = torch.squeeze(inputs_2d)
        inputs_3d_p = inputs_3d.permute(1, 0, 2, 3)
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

        iter = 0
        for _, batch_3d, batch_2d in self.train_generator.next_epoch():

            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_3d = inputs_3d.cuda()

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
            predicted_3d_pos = self.pos_model_train(inputs_2d)
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
                predicted_3d_trj = self.trj_model_train(inputs_2d)
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

            # Evaluate on test set
            for cam, batch, batch_2d in self.test_generator.next_epoch():

                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                inputs_3d = torch.from_numpy(batch.astype('float32'))

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

                predicted_3d_pos = self.pos_model_test(inputs_2d)
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
                    predicted_3d_trj = self.trj_model_test(inputs_2d)
                    predicted_3d_pos += predicted_3d_trj
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_traj)
                    w = torch.abs(1 / inputs_traj[:, :, 0, 2])
                    epoch_loss_3d_trj += inputs_3d.shape[0] * inputs_3d.shape[1] * weighted_mpjpe(predicted_3d_trj, inputs_traj[:, :, 0:1], w).item()
                else:
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)

                epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

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
                        'random_state': self.train_generator.random_state(),
                        'optimizer': self.optimizer.state_dict(),
                        'model_pos': self.pos_model_train.state_dict(),
                        'model_trj': self.trj_model_train.state_dict()
                    }, chk_path)
                else:
                    torch.save({
                        'epoch': epoch,
                        'lr': self.lr,
                        'best_performance': self.losses_3d_valid[-1] * 1000 if self.losses_3d_valid[-1] * 1000 < self.min_loss else self.min_loss,
                        'random_state': self.train_generator.random_state(),
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

    def evaluate_core(self, test_generator, action=None, return_predictions=False, flip_test=False):

        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_pos_scale = 0
        epoch_loss_3d_vel = 0
        epoch_loss_3d_root = 0
        with torch.no_grad():
            self.pos_model_test.eval()
            if self.model_config['TRAJECTORY_MODEL']:
                self.trj_model_test.eval()

            N = 0
            for cam, batch, batch_2d in test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if flip_test:
                    inputs_2d_flip = inputs_2d.clone()
                    inputs_2d_flip[:, :, :, 0] *= -1
                    inputs_2d_flip[:, :, self.kps_left + self.kps_right, :] = inputs_2d_flip[:, :,
                                                                              self.kps_right + self.kps_left, :]

                if return_predictions:
                    if self.model_config['TRAJECTORY_MODEL']:
                        return (
                                self.pos_model_test(inputs_2d) + self.trj_model_test(inputs_2d)
                        ).squeeze(0).cpu().numpy()
                    else:
                        return self.pos_model_test(inputs_2d).squeeze(0).cpu().numpy()

                if self.model_config['TRAJECTORY_MODEL'] or self.data_config['RAY_ENCODING']:
                    # do nothing
                    pass
                else:
                    batch[:, :, 1:] -= batch[:, :, 0:1]
                    batch[:, :, 0] = 0

                inputs_3d = torch.from_numpy(batch.astype('float32'))
                inputs_2d, inputs_3d = self.eval_data_prepare(self.receptive_field, inputs_2d, inputs_3d)
                if flip_test:
                    inputs_2d_flip, _ = self.eval_data_prepare(self.receptive_field, inputs_2d_flip, inputs_3d)

                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_3d = inputs_3d.cuda()
                    if flip_test:
                        inputs_2d_flip = inputs_2d_flip.cuda()

                if self.model_config['TRAJECTORY_MODEL']:
                    predicted_3d_pos = self.pos_model_test(inputs_2d)
                    if flip_test:
                        predicted_3d_pos_flip = self.pos_model_test(inputs_2d_flip)
                        predicted_3d_pos_flip[:, :, :, 0] *= -1
                        predicted_3d_pos_flip[:, :, self.kps_left + self.kps_right] = predicted_3d_pos_flip[:, :,
                                                                                      self.kps_right + self.kps_left]
                        predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1),
                                                      dim=1,
                                                      keepdim=True)
                    predicted_3d_trj = self.trj_model_test(inputs_2d)
                    if flip_test:
                        predicted_3d_trj_flip = self.trj_model_test(inputs_2d_flip)
                        predicted_3d_trj_flip[:, :, :, 0] *= -1
                        predicted_3d_trj = torch.mean(torch.cat((predicted_3d_trj, predicted_3d_trj_flip), dim=1),
                                                      dim=1,
                                                      keepdim=True)
                    predicted_3d_pos += predicted_3d_trj
                    if not cam is None:
                        pred = predicted_3d_pos.cpu().numpy()
                        target = inputs_3d.cpu().numpy()
                        if self.data_config['RAY_ENCODING']:
                            pred_world = cam.normalized2world(pred)
                            target_world = cam.normalized2world(target)
                        else:
                            pred_world = cam.camera2world(pred)
                            target_world = cam.camera2world(target)
                        predicted_3d_pos = torch.from_numpy(pred_world)
                        inputs_3d = torch.from_numpy(target_world)
                else:
                    predicted_3d_pos = self.pos_model_test(inputs_2d)
                    if flip_test:
                        predicted_3d_pos_flip = self.pos_model_test(inputs_2d_flip)
                        predicted_3d_pos_flip[:, :, :, 0] *= -1
                        predicted_3d_pos_flip[:, :, self.kps_left + self.kps_right] = predicted_3d_pos_flip[:, :,
                                                                                      self.kps_right + self.kps_left]
                        predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                                      keepdim=True)
                    if self.data_config['RAY_ENCODING']:
                        # do nothing
                        pred = predicted_3d_pos.cpu().numpy()
                        target = inputs_3d.cpu().numpy()
                        pred_world = cam.normalized2world(pred)
                        target_world = cam.normalized2world(target)
                        predicted_3d_pos = torch.from_numpy(pred_world)
                        inputs_3d = torch.from_numpy(target_world)
                    else:
                        # do nothing
                        pass

                epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * mpjpe(predicted_3d_pos, inputs_3d).item()
                epoch_loss_3d_root += inputs_3d.shape[0] * inputs_3d.shape[1] * mpjpe(predicted_3d_pos[:, :, 0:1, :], inputs_3d[:, :, 0:1, :]).item()
                epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

                inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

                epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

                epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

                N += inputs_3d.shape[0] * inputs_3d.shape[1]

        e1 = (epoch_loss_3d_pos / N) * 1000
        e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
        e3 = (epoch_loss_3d_pos_scale / N) * 1000
        ev = (epoch_loss_3d_vel / N) * 1000
        er = (epoch_loss_3d_root / N) * 1000

        return e1, e2, e3, ev, er

    def evaluate(self, mlog, subjects_test, pose_data, action_filter, pad, causal_shift, epoch, plot=False):

        all_actions = dict()
        for subject in subjects_test:
            # all_actions.setdefault('Sitting 1', list()).append((subject, 'Sitting 1'))
            if action_filter == None:
                action_keys = pose_data.get_dataset()[subject].keys()
            else: 
                action_keys = action_filter
            for action in action_keys:
                all_actions.setdefault(action.split(' ')[0], list()).append((subject, action))

        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []
        errors_root = []

        if 'CAMERA_WISE_PERFORMANCE' in self.data_config and self.data_config['CAMERA_WISE_PERFORMANCE']:
            camera_dist = pose_data.get_dataset().camera_dist
            for cam_idx in range(len(camera_dist)):
                cam_id = camera_dist[cam_idx]
                for action_key in all_actions.keys():

                    poses_cam, poses_act, poses_2d_act = pose_data.fetch_via_action(all_actions[action_key], camera_idx=cam_idx)
                    action_generator = UnchunkedGenerator(poses_cam, poses_act, poses_2d_act,
                                                          pad=pad, causal_shift=causal_shift,
                                                          kps_left=self.kps_left, kps_right=self.kps_right,
                                                          joints_left=self.joints_left, joints_right=self.joints_right)
                    e1, e2, e3, ev, er = self.evaluate_core(action_generator, action_key,
                                                        flip_test=self.train_config['TEST_TIME_AUGMENTATION'])
                    errors_p1.append(e1)
                    errors_p2.append(e2)
                    errors_p3.append(e3)
                    errors_vel.append(ev)
                    errors_root.append(er)

                p1, p2, p3, p4 = round(np.mean(errors_p1), 1), round(np.mean(errors_p2), 1), round(np.mean(errors_p3), 1), round(np.mean(errors_vel), 1)
                p5 = round(np.mean(errors_root), 1)
                mlog.info('CAM ID {}, {} {} {} {} {}'.format(cam_id, p1, p2, p3, p4, p5))
        else:
            for action_key in all_actions.keys():

                poses_cam, poses_act, poses_2d_act = pose_data.fetch_via_action(all_actions[action_key])
                action_generator = UnchunkedGenerator(poses_cam, poses_act, poses_2d_act,
                                                      pad=pad, causal_shift=causal_shift,
                                                      kps_left=self.kps_left, kps_right=self.kps_right,
                                                      joints_left=self.joints_left, joints_right=self.joints_right)
                if action_key is None:
                    mlog.info('----------')
                else:
                    mlog.info('----' + action_key + '----')
                e1, e2, e3, ev, er = self.evaluate_core(action_generator, action_key,
                                                    flip_test=self.train_config['TEST_TIME_AUGMENTATION'])
                mlog.info('Protocol #1 Error (MPJPE):   {} mm'.format(e1))
                mlog.info('Protocol #2 Error (P-MPJPE): {} mm'.format(e2))
                mlog.info('Protocol #3 Error (N-MPJPE): {} mm'.format(e3))
                mlog.info('Velocity    Error (MPJVE):   {} mm'.format(ev))
                mlog.info('Root        Error (MRPE):   {} mm'.format(er))
                mlog.info('----------')
                errors_p1.append(e1)
                errors_p2.append(e2)
                errors_p3.append(e3)
                errors_vel.append(ev)
                errors_root.append(er)

            mlog.info('Protocol #1   (MPJPE) action-wise average: {} mm'.format(round(np.mean(errors_p1), 1)))
            mlog.info('Protocol #2 (P-MPJPE) action-wise average: {} mm'.format(round(np.mean(errors_p2), 1)))
            mlog.info('Protocol #3 (N-MPJPE) action-wise average: {} mm'.format(round(np.mean(errors_p3), 1)))
            mlog.info('Velocity      (MPJVE) action-wise average: {} mm'.format(round(np.mean(errors_vel), 1)))
            mlog.info('Root           (MRPE) action-wise average: {} mm'.format(round(np.mean(errors_root), 1)))

            if self.plotter and plot:
                self.plotter.log_metric('MPJPE', round(np.mean(errors_p1), 1), epoch)
                self.plotter.log_metric('P-MPJPE', round(np.mean(errors_p2), 1), epoch)
                self.plotter.log_metric('N-MPJPE', round(np.mean(errors_p3), 1), epoch)
                self.plotter.log_metric('MPJVE', round(np.mean(errors_vel), 1), epoch)

    def render(self, dataset, keypoints, keypoints_metadata,
               pad, causal_shift, kps_left, kps_right, joints_left, joints_right, file_names=None):

        viz_subject = self.plot_config['VIZ_SUBJECT']
        viz_action = self.plot_config['VIZ_ACTION']
        viz_camera = self.plot_config['VIZ_CAMERA']
        input_keypoints = keypoints[viz_subject][viz_action][viz_camera].copy()
        ground_truth = None
        if self.data_config['WORLD_3D_GT_EVAL']:
            if viz_subject in dataset.subjects() and viz_action in dataset[viz_subject]:
                if 'positions_3d' in dataset[viz_subject][viz_action]:
                    ground_truth = dataset[viz_subject][viz_action]['positions_3d'][viz_camera].copy()
            if ground_truth is None:
                print('INFO: this action is unlabeled. Ground truth will not be rendered.')

        render_generator = UnchunkedGenerator(None, None, [input_keypoints],
                                              pad=pad, causal_shift=causal_shift, augment=False,
                                              kps_left=kps_left, kps_right=kps_right,
                                              joints_left=joints_left, joints_right=joints_right)
        prediction = self.evaluate_core(render_generator, return_predictions=True)

        if ground_truth is not None and not self.model_config['TRAJECTORY_MODEL'] and not self.data_config['RAY_ENCODING']:
            # Reapply trajectory
            trajectory = ground_truth[:, :1]
            prediction += trajectory

        # Invert camera transformation
        cam = dataset.camera_info[viz_subject][viz_camera]
        if ground_truth is not None:
            if self.data_config['RAY_ENCODING']:
                pred_world = cam.normalized2world(prediction)
                target_world = cam.normalized2world(ground_truth)
            else:
                pred_world = cam.camera2world(prediction)
                target_world = cam.camera2world(ground_truth)
            prediction = pred_world
            ground_truth = target_world
        else:
            if self.data_config['RAY_ENCODING']:
                pred_world = cam.normalized2world(prediction)
            else:
                pred_world = cam.camera2world(prediction)
            prediction = pred_world

            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        anim_output = {'Reconstruction': prediction}
        if ground_truth is not None:
            anim_output['Ground truth'] = ground_truth

        if self.data_config['RAY_ENCODING']:
            if self.data_config['ADD_HEIGHT']:
                pt_cam = cam.get_uv_given_cam_ray(input_keypoints[:, :-1])
            else:
                pt_cam = cam.get_uv_given_cam_ray(input_keypoints)
            input_keypoints = pt_cam[..., :2]
        elif self.data_config['INTRINSIC_ENCODING']:
            kps_orig = cam.decouple_uv_with_intrinsic(input_keypoints)
            input_keypoints = kps_orig
        else:
            input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam.res_w, h=cam.res_h)

        # original video
        if self.data_config['DATASET'] == '3dhp':
            input_file_names = file_names[viz_subject][viz_action][viz_camera]
            input_video_path = None
        elif self.data_config['DATASET'] == 'humaneva':
            input_file_names = None
            input_video_path = None
        else:
            input_file_names = None
            input_video_path = self.plot_config['VIZ_VIDEO']

        from lib.visualization.visualization import render_animation
        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), viz_camera, cam.azimuth,
                         self.plot_config['VIZ_OUTPUT'],
                         limit=self.plot_config['VIZ_LIMIT'],
                         downsample=self.plot_config['VIZ_DOWNSAMPLE'],
                         size=self.plot_config['VIZ_SIZE'],
                         input_video_path=input_video_path,
                         viewport=(cam.res_w, cam.res_h),
                         input_video_skip=self.plot_config['VIZ_SKIP'],
                         input_file_names = input_file_names
                         )
