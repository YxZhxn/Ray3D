import os
import ipdb
import torch
import shutil
import numpy as np

from lib.loss.loss import mpjpe, p_mpjpe, weighted_l1_loss
from lib.dataloader.generators import UnchunkedGenerator


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

        self.min_loss = 1e5 if best_performance is None else best_performance
        self.losses_3d_train = []
        self.losses_3d_valid = []

        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        self.plotter = plotter

    def train(self, epoch, mlog):

        N = 0
        iter = 0
        epoch_loss_3d_train = 0

        self.pos_model_train.train()

        for cam, batch, batch_2d in self.train_generator.next_epoch():

            assert batch_2d.shape[1] == 1
            assert batch.shape[1] == 1

            cam_cp = cam[:,0:2]
            cam_fl = cam[:,2:4]

            # 2D pose
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')).squeeze(dim=1)
            batch_size = inputs_2d.shape[0]

            # 3D pose
            depth_root_canonical = batch[:, 0, 0, 2].copy() / np.sqrt(np.prod(cam_fl, axis=1)) * self.data_config['F0']
            depth_root_canonical = torch.from_numpy(depth_root_canonical.astype('float32'))

            batch_3d_relative = batch - batch[:, :, 0:1, :]
            batch_3d = np.delete(batch_3d_relative, (0), axis=2)
            inputs_3d = torch.from_numpy(batch_3d.astype('float32')).squeeze(dim=1)

            # center point
            cam_cp = torch.from_numpy(cam_cp.astype('float32'))

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_3d = inputs_3d.cuda()
                depth_root_canonical = depth_root_canonical.cuda()
                cam_cp = cam_cp.cuda()

            # Predict 3D poses
            self.optimizer.zero_grad()
            outputs = self.pos_model_train(inputs_2d, cam_cp)

            loss = 0
            loss += weighted_l1_loss(outputs[0], inputs_3d)
            loss += weighted_l1_loss(outputs[1], depth_root_canonical)
            epoch_loss_3d_train += batch_size * loss.item()
            N += batch_size

            # ---------------- visualization ---------------- #
            if iter % 2048 == 0 and self.plotter is not None and epoch % 64 == 0:
                pose = torch.zeros(batch_size, outputs[0].shape[1] + 1, outputs[0].shape[2])
                pose[:, 1:] = outputs[0].detach().cpu()
                pose = pose.unsqueeze(dim=1)
                self.plotter.show_plot(
                    epoch,
                    batch_2d,
                    batch_3d_relative,
                    pose.numpy(),
                    dataset=self.data_config['DATASET'],
                    gt=self.data_config['KEYPOINTS']
                )
            # ---------------- visualization ---------------- #

            iter += 1
            loss.backward()
            self.optimizer.step()

        self.losses_3d_train.append(epoch_loss_3d_train / N)
        torch.cuda.empty_cache()

        if self.plotter:
            self.plotter.log_metric('train', self.losses_3d_train[-1], epoch)
            self.plotter.log_metric('lr', self.lr, epoch)

        return self.losses_3d_train[-1], self.lr

    def test(self, epoch, mlog):

        with torch.no_grad():
            self.pos_model_test.load_state_dict(self.pos_model_train.state_dict(), strict=True)
            self.pos_model_test.eval()

            epoch_loss_3d_valid = 0
            epoch_mpjpe = 0
            N = 0

            # Evaluate on test set
            for cam, batch, batch_2d in self.test_generator.next_epoch():

                batch = np.squeeze(batch, axis=0)
                batch_2d = np.squeeze(batch_2d, axis=0)

                cam_cp = np.array([[c.K[0, 2], c.K[1, 2]] for c in cam]).astype('float32')
                cam_fl = np.array([[c.K[0, 0], c.K[1, 1]] for c in cam]).astype('float32')

                # 2D pose
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                batch_size = inputs_2d.shape[0]

                # 3D pose
                depth_root_canonical = batch[:, 0, 2].copy() / np.sqrt(np.prod(cam_fl, axis=1)) * self.data_config['F0']
                depth_root_canonical = torch.from_numpy(depth_root_canonical.astype('float32'))

                batch_3d_relative = batch - batch[:, 0:1]
                batch_3d = np.delete(batch_3d_relative, (0), axis=1)
                inputs_3d = torch.from_numpy(batch_3d.astype('float32'))

                # center point
                cam_cp = torch.from_numpy(cam_cp.astype('float32'))

                # focal length
                cam_fl = torch.from_numpy(cam_fl.astype('float32'))

                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_3d = inputs_3d.cuda()
                    depth_root_canonical = depth_root_canonical.cuda()
                    cam_cp = cam_cp.cuda()
                    cam_fl = cam_fl.cuda()

                outputs = self.pos_model_test(inputs_2d, cam_cp)

                loss = 0
                loss += weighted_l1_loss(outputs[0], inputs_3d)
                loss += weighted_l1_loss(outputs[1], depth_root_canonical)
                epoch_loss_3d_valid += batch_size * loss.item()
                N += batch_size

                # get predicted root coordinates
                pose2d = inputs_2d.detach()
                x = pose2d[:, 0, 0]
                y = pose2d[:, 0, 1]
                cx = cam_cp[:, 0].detach()
                cy = cam_cp[:, 1].detach()
                Z = outputs[1].detach()
                X = (x - cx) * Z / self.data_config['F0']
                Y = (y - cy) * Z / self.data_config['F0']
                f = torch.sqrt(torch.prod(cam_fl.detach(), 1)) / self.data_config['F0']
                Z = Z * f
                pred_root = torch.cat((X.view(batch_size, 1), Y.view(batch_size, 1), Z.view(batch_size, 1)), 1)

                pose_pred = torch.zeros(batch_size, outputs[0].shape[1] + 1, outputs[0].shape[2]).cuda()
                pose_pred[:, 1:] = outputs[0].detach()
                pose_pred += pred_root.unsqueeze(dim=1)
                pose_gt = torch.from_numpy(batch.astype('float32')).cuda()
                epoch_mpjpe += batch_size * mpjpe(pose_pred, pose_gt).item()

            self.losses_3d_valid.append(epoch_loss_3d_valid / N)

            # Save checkpoint if necessary
            if epoch % self.train_config['CHECKPOINT_FREQUENCY'] == 0:
                chk_path = os.path.join(self.train_config['CHECKPOINT'], 'epoch_{}.bin'.format(epoch))
                mlog.info('Saving epochs {}\'s checkpoint to {}.'.format(epoch, chk_path))
                torch.save({
                    'epoch': epoch,
                    'lr': self.lr,
                    'best_performance': (epoch_mpjpe / N * 1000) if (epoch_mpjpe / N * 1000) < self.min_loss else self.min_loss,
                    'random_state': self.train_generator.random_state(),
                    'optimizer': self.optimizer.state_dict(),
                    'model_pos': self.pos_model_train.state_dict(),
                }, chk_path)

                #### save best checkpoint
                best_chk_path = os.path.join(self.train_config['CHECKPOINT'], 'best_epoch.bin'.format(epoch))
                if (epoch_mpjpe / N * 1000) < self.min_loss:
                    self.min_loss = (epoch_mpjpe / N * 1000)
                    mlog.info('Saving best checkpoint to {} with mpjpe: {}.'.format(best_chk_path, self.min_loss))
                    shutil.copy(chk_path, best_chk_path)

                cmd = 'rm {}'.format(chk_path)
                os.system(cmd)

            # Decay learning rate exponentially
            self.lr *= self.train_config['LR_DECAY']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.train_config['LR_DECAY']

        if self.plotter:
            # plot all the losses
            self.plotter.log_metric('test', self.losses_3d_valid[-1], epoch)

        # return the current epoch's mpjme
        return self.losses_3d_valid[-1]

    def evaluate_core(self, test_generator, action=None, return_predictions=False, flip_test=False):

        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_pos_relative = 0
        epoch_loss_3d_pos_procrustes_relative = 0
        epoch_loss_3d_root = 0

        with torch.no_grad():
            self.pos_model_test.eval()

            N = 0
            for cam, batch, batch_2d in test_generator.next_epoch():

                batch = np.squeeze(batch, axis=0)
                batch_2d = np.squeeze(batch_2d, axis=0)

                cam_cp = np.array([[cam.K[0, 2], cam.K[1, 2]]]).astype('float32')
                cam_fl = np.array([[cam.K[0, 0], cam.K[1, 1]]]).astype('float32')

                # 2D pose
                inputs_2d = torch.from_numpy(batch_2d.astype('float32')).squeeze(dim=1)
                batch_size = inputs_2d.shape[0]

                batch_3d_relative = batch - batch[:, 0:1]
                batch_3d = np.delete(batch_3d_relative, (0), axis=1)
                inputs_3d = torch.from_numpy(batch_3d.astype('float32'))

                if flip_test:
                    inputs_2d_flip = inputs_2d.clone()
                    inputs_2d_flip[:, :, 0] = cam.res_w - inputs_2d_flip[:, :, 0]
                    inputs_2d_flip[:, self.kps_left + self.kps_right, :] = inputs_2d_flip[:, self.kps_right + self.kps_left, :]

                # center point
                cam_cp = torch.from_numpy(cam_cp.astype('float32')).repeat(batch_size, 1)

                # focal length
                cam_fl = torch.from_numpy(cam_fl.astype('float32'))

                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_3d = inputs_3d.cuda()
                    cam_cp = cam_cp.cuda()
                    cam_fl = cam_fl.cuda()
                    if flip_test:
                        inputs_2d_flip = inputs_2d_flip.cuda()

                outputs = self.pos_model_test(inputs_2d, cam_cp)
                if flip_test:
                    kps_left = [item - 1 for item in self.kps_left]
                    kps_right = [item - 1 for item in self.kps_right]
                    cam_cp_copy = cam_cp.clone()
                    cam_cp_copy[:, 0] = cam.res_w - cam_cp_copy[:, 0]
                    outputs_flip = self.pos_model_test(inputs_2d_flip, cam_cp_copy)
                    outputs_flip[0][:, :, 0] *= -1
                    outputs_flip[0][:, kps_left + kps_right] = outputs_flip[0][:, kps_right + kps_left]
                    outputs[0] = (outputs[0] + outputs_flip[0]) / 2
                    outputs[1] = (outputs[1] + outputs_flip[1]) / 2

                epoch_loss_3d_pos_relative += batch_size * mpjpe(outputs[0], inputs_3d).item()
                epoch_loss_3d_pos_procrustes_relative += batch_size * p_mpjpe(outputs[0].cpu().numpy(), inputs_3d.cpu().numpy()).item()

                # get predicted root coordinates
                pose2d = inputs_2d.detach()
                x = pose2d[:, 0, 0]
                y = pose2d[:, 0, 1]
                cx = cam_cp[:, 0].detach()
                cy = cam_cp[:, 1].detach()
                Z = outputs[1].detach()
                X = (x - cx) * Z / self.data_config['F0']
                Y = (y - cy) * Z / self.data_config['F0']
                f = torch.sqrt(torch.prod(cam_fl.detach(), 1)) / self.data_config['F0']
                Z = Z * f
                pred_root = torch.cat((X.view(batch_size, 1), Y.view(batch_size, 1), Z.view(batch_size, 1)), 1)

                pose_pred = torch.zeros(batch_size, outputs[0].shape[1] + 1, outputs[0].shape[2]).cuda()
                pose_pred[:, 1:] = outputs[0].detach()
                pose_pred += pred_root.unsqueeze(dim=1)
                pose_gt = torch.from_numpy(batch.astype('float32')).cuda()

                pred = pose_pred.cpu().numpy()
                target = pose_gt.cpu().numpy()
                pred_world = cam.camera2world(pred)
                target_world = cam.camera2world(target)
                pose_pred = torch.from_numpy(pred_world)
                pose_gt = torch.from_numpy(target_world)

                epoch_loss_3d_pos += batch_size * mpjpe(pose_pred, pose_gt).item()
                epoch_loss_3d_root += batch_size * mpjpe(pose_pred[:, 0:1, :], pose_gt[:, 0:1, :])
                epoch_loss_3d_pos_procrustes += batch_size * p_mpjpe(pose_pred.cpu().numpy(), pose_gt.cpu().numpy()).item()

                N += batch_size

        e1 = (epoch_loss_3d_pos / N) * 1000
        e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
        e1_relative = (epoch_loss_3d_pos_relative / N) * 1000
        e2_relative = (epoch_loss_3d_pos_procrustes_relative / N) * 1000
        er = (epoch_loss_3d_root / N) * 1000

        return e1, e2, e1_relative, e2_relative, er

    def evaluate(self, mlog, subjects_test, pose_data, action_filter, pad, causal_shift, epoch, plot=False):

        all_actions = dict()
        for subject in subjects_test:
            if action_filter == None:
                action_keys = pose_data.get_dataset()[subject].keys()
            else: 
                action_keys = action_filter
            for action in action_keys:
                all_actions.setdefault(action.split(' ')[0], list()).append((subject, action))

        errors_p1 = []
        errors_p2 = []
        errors_p1_relative = []
        errors_p2_relative = []
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
                    e1, e2, e1_relative, e2_relative, er = self.evaluate_core(action_generator, action_key,
                                                        flip_test=self.train_config['TEST_TIME_AUGMENTATION'])
                    errors_p1.append(e1)
                    errors_p2.append(e2)
                    errors_p1_relative.append(e1_relative)
                    errors_p2_relative.append(e2_relative)
                    errors_root.append(er)

                p1, p2 = round(np.mean(errors_p1), 1), round(np.mean(errors_p2), 1)
                p5 = round(np.mean(errors_root), 1)
                p1_relative, p2_relative = round(np.mean(errors_p1_relative), 1), round(np.mean(errors_p2_relative), 1)
                mlog.info('CAM ID {}, {} {} {} {} {}'.format(cam_id, p1, p2, p1_relative, p2_relative, p5))
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
                e1, e2, e1_relative, e2_relative, er = self.evaluate_core(action_generator, action_key,
                                                    flip_test=self.train_config['TEST_TIME_AUGMENTATION'])
                mlog.info('Protocol #1 Error (MPJPE):   {} mm'.format(e1))
                mlog.info('Protocol #2 Error (P-MPJPE): {} mm'.format(e2))
                mlog.info('Protocol #1 Error (RELATIVE MPJPE):   {} mm'.format(e1_relative))
                mlog.info('Protocol #2 Error (RELATIVE P-MPJPE): {} mm'.format(e2_relative))
                mlog.info('Root        Error (MRPE):   {} mm'.format(er))
                mlog.info('----------')
                errors_p1.append(e1)
                errors_p2.append(e2)
                errors_p1_relative.append(e1_relative)
                errors_p2_relative.append(e2_relative)
                errors_root.append(er)

            mlog.info('Protocol #1   (MPJPE) action-wise average: {} mm'.format(round(np.mean(errors_p1), 1)))
            mlog.info('Protocol #2 (P-MPJPE) action-wise average: {} mm'.format(round(np.mean(errors_p2), 1)))
            mlog.info('Protocol #1   (RELATIVE MPJPE) action-wise average: {} mm'.format(round(np.mean(errors_p1_relative), 1)))
            mlog.info('Protocol #2 (RELATIVE P-MPJPE) action-wise average: {} mm'.format(round(np.mean(errors_p2_relative), 1)))
            mlog.info('Root           (MRPE) action-wise average: {} mm'.format(round(np.mean(errors_root), 1)))

            if self.plotter and plot:
                self.plotter.log_metric('MPJPE', round(np.mean(errors_p1), 1), epoch)
                self.plotter.log_metric('P-MPJPE', round(np.mean(errors_p2), 1), epoch)
