import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import GRAILNet, CosineIncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
import os
from models.GRAIL.GranularBall import GBList
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import time
import copy
from scipy.spatial.distance import cdist
from utils.toolkit import tensor2numpy, accuracy
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics.pairwise import cosine_similarity

T = 2
EPSILON = 1e-8

class GRAIL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = GRAILNet(args, False)

        self._protos = []
        self._covs = []
        self._radiuses = []
        self.granularball_list_obj = None
        self.seed = args["seed"]
        self.init_epoch = args['init_epoch']
        self.init_lr = args['init_lr']
        self.init_milestones =args['init_milestones']
        self.init_lr_decay = args['init_lr_decay']
        self.init_weight_decay = args['init_weight_decay']
        self.epochs = args['epochs']
        self.lrate = args['lrate']
        self.milestones = args['milestones']
        self.lrate_decay = args['lrate_decay']
        self.batch_size = args['batch_size']
        self.weight_decay = args['weight_decay']
        self.num_workers = args['num_workers']
        self.w_kd = args['w_kd']
        self.w_old_cls = args['w_old_cls']
        self.w_new_cls = args['w_new_cls']
        self.w_cont = args['w_cont']
        self.w_trsf = args['w_trsf']
        self.remain_ball_num = args['remain_ball_num']
        self.use_past_model = args['use_past_model']
        self.save_model = args['save_model']
        self.use_multigranularity = args.get('use_multigranularity', True)
        self.use_class_aware_weight = args.get('use_class_aware_weight', True)
        self.blur_factor = args.get('blur_factor', 0.03)
        self.scale_factor = args.get('scale_factor', 1)
        self.model_dir = args['model_dir']
        self.dataset = args['dataset']
        self.init_cls = args['init_cls']
        self.increment = args['increment']
        self.plot2D = args['plot2D']
        self._process_id = args['process_id']
        self.plot_fea_var = args.get('plot_fea_var', False)
        self.aug_sim = args.get('aug_sim', 0)
        self.is_task0 = args.get('is_task0', False)
        self.args["temp"] = 0.1

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        if self.is_task0:     # Save the model of the initial task
            path = self.model_dir + "/{}/{}".format(self.dataset, self.seed)
            if not os.path.exists(path):
                os.makedirs(path)
            self.save_checkpoint("{}/{}".format(path, self.init_cls))
            np.save("{}/{}_first_task_balls.npy".format(path, self.init_cls), self.granularball_list_obj)

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        task_size = self.data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc_aug(self._known_classes*4, self._total_classes*4, int((task_size-1)*task_size/2))
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        self.shot = None
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train",mode="train",)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        # Wrap the model for data parallelism across multiple GPUs; split input across devices to run the same model in parallel
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        # Restore the model back to a single GPU
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        if self._old_network is not None:
            self._old_network.to(self._device)
        model_dir = "{}/{}/{}/{}_{}.pkl".format(self.args["model_dir"],self.args["dataset"], self.args["seed"],self.args["init_cls"],self._cur_task)
        if self._cur_task == 0:
            if self.use_past_model and os.path.exists(model_dir):
                self._network.load_state_dict(torch.load(model_dir)["model_state_dict"], strict=True)
                self._network.to(self._device)
            else:
                self._network.to(self._device)
                if self.dataset == "imagenetsubset":
                    base_lr = 0.1
                    lr_strat = [80, 120, 150]
                    lr_factor = 0.1
                    custom_weight_decay = 5e-4
                    custom_momentum = 0.9
                    optimizer = torch.optim.SGD(self._network.parameters(), lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
                    scheduler = MultiStepLR(optimizer, milestones=lr_strat, gamma=lr_factor)
                else:
                    optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lrate, weight_decay=self.weight_decay)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=self.lrate_decay)
                optimizer = optim.SGD(self._network.parameters(),momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.init_milestones, gamma=self.init_lr_decay)
                self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._network.init_de()
            if self.use_past_model and os.path.exists(model_dir):
                self._network.load_state_dict(torch.load(model_dir)["model_state_dict"], strict=True)
                self._network.to(self._device)
            else:
                self._network.to(self._device)
                optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lrate, weight_decay=self.weight_decay)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=self.lrate_decay)
                self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if self.w_trsf > 0:
                self._update_memory(train_loader)

        self._build_protos()
        first_task_ball_dir = "{}/{}/{}/{}_first_task_balls.npy".format(self.args["model_dir"],self.args["dataset"],self.args["seed"], self.args["init_cls"])
        if self._cur_task == 0 and os.path.exists(first_task_ball_dir):
            self.granularball_list_obj = np.load(first_task_ball_dir, allow_pickle=True).item()
        else:
            self._generate_gbs()
        
    def _drift_estimator(self, train_loader):
        if hasattr(self._network, "module"):
            _network = self._network.module
        else:
            _network = self._network

        optimizer = optim.Adam(_network.drift_estimator.parameters(), lr=0.001)
        for epoch in range(20):
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs_aug, targets_aug, _ = self._class_aug(inputs,targets)
                fea_aug = _network(inputs_aug)["features"]
                fea_aug_old = self._old_network(inputs_aug)["features"]
                fea_aug_transfer = _network.drift_estimator(fea_aug_old)["logits"]

                loss = self.l2loss(fea_aug, fea_aug_transfer)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _update_memory(self, train_loader):
        self._network.drift_estimator.eval()

        proto_features_raw = torch.tensor(np.array(self._protos)).to(self._device).to(torch.float32)
        trsf_proto = self._network.drift_estimator(proto_features_raw)['logits'].detach().clone()
        drift_strength = torch.norm(trsf_proto - proto_features_raw, p=2, dim=1)  # [N]
        if self.use_class_aware_weight:
            # Weighting scheme 1
            drift_weight = drift_strength / (drift_strength.max() + 1e-8)  # [N]
            drift_weight = self.scale_factor * torch.sigmoid(drift_weight)
        else:
            drift_weight = torch.ones_like(drift_strength)
        logging.info(str(drift_weight.cpu().tolist()))
        with torch.no_grad():
            for cls_index in range(0, self._known_classes):
                tmp = self._network.drift_estimator(torch.tensor(self._protos[cls_index]).to(self._device).to(torch.float32))['logits'].detach()
                self._protos[cls_index] = self._protos[cls_index] + drift_weight[cls_index].cpu().numpy() * (tmp.cpu().numpy() - self._protos[cls_index])
        # Update granular-ball centers
        with torch.no_grad():
            for gb in self.granularball_list_obj.granular_balls:
                if gb.label in range(0, self._known_classes):
                    center = self._network.drift_estimator(torch.tensor(gb.center).to(self._device).to(torch.float32))['logits'].detach()
                    gb.center = gb.center + drift_weight[gb.label].cpu().numpy() * (center.cpu().numpy()-gb.center)
                else:
                    print('Please dont to correct a new class')
        self._network.drift_estimator.train()        

    def _generate_gbs(self):
        vectors, targets, idx_dataset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',mode='test', shot=self.shot, ret_data=True)
        idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        vectors, _ = self._extract_vectors(idx_loader)
        new_granularball_list_obj = GBList(vectors, targets, task_id=self._cur_task)
        # Initial split
        new_granularball_list_obj.init_granular_balls4(purity=1, min_sample=1, remain_ball_num=self.remain_ball_num)
        ball1, ball2, ball3, ball4 = [], [], [], []
        purity = []
        for _class in range(self._known_classes,self._total_classes):
            ball1.append( "balls-label-{} [init-split]  : {}".format(_class, sorted(new_granularball_list_obj.get_data_size_with_class_filter(_class=_class), reverse=True)))
        # Re-split
        # new_granularball_list_obj.resplit()
        # for _class in range(self._known_classes,self._total_classes):
        #     ball2.append( "balls-label-{} [resplit]     : {}".format(_class, sorted(new_granularball_list_obj.get_data_size_with_class_filter(_class=_class), reverse=True)))
        # # fusion
        # new_granularball_list_obj.ball_fusion()
        # for _class in range(self._known_classes,self._total_classes):
        #     ball3.append( "balls-label-{} [fusion]      : {}".format(_class, sorted(new_granularball_list_obj.get_data_size_with_class_filter(_class=_class), reverse=True)))
        #     # purity.append( "balls-label-{}: {}".format(_class, sorted(new_granularball_list_obj.get_purity_with_class_filter(_class=_class), reverse=True)))
        # Representative ball selection
        if self._cur_task == 0:
            new_granularball_list_obj.ball_selection()
        else:
            new_granularball_list_obj.ball_selection(_max=5)
        for _class in range(self._known_classes, self._total_classes):
            ball4.append( "balls-label-{} [select]      : {}".format(_class, sorted(new_granularball_list_obj.get_data_size_with_class_filter(_class=_class), reverse=True)))
        for _class in range(len(ball1)):

            logging.info(ball4[_class])
        if self._cur_task == 0:
            self.max_ball_num = new_granularball_list_obj.max_first_task_size
            self.granularball_list_obj = new_granularball_list_obj
        else:
            self.granularball_list_obj.merge_new_ball_list(new_granularball_list_obj)

    def _build_protos(self):
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',mode='test', shot=self.shot, ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            class_mean = np.mean(vectors, axis=0)
            self._protos.append(class_mean)
            cov = np.cov(vectors.T)
            self._covs.append(cov)
            self._radiuses.append(np.trace(cov)/vectors.shape[1])
        self._radius = np.sqrt(np.mean(self._radiuses))
    
    
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.init_epoch), colour='red', position=self._process_id, dynamic_ncols=True, ascii=" =", leave=True)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            L_all = 0.0
            L_new_cls = 0.0
            L_new_aug_cls = 0.0
            L_cont = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs_aug, targets_aug, _ = self._class_aug(inputs,targets)

                fea2 = self._network(inputs_aug)["features"]
                logits2 = self._network.fc_aug(fea2)["logits"]
                
                loss_new_cls2 = F.cross_entropy(logits2/self.args["temp"], targets_aug)
                L_new_aug_cls += loss_new_cls2.item()


                loss =  loss_new_cls2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                L_all += loss.item()

                _, preds = torch.max(logits2, dim=1)
                correct += preds.eq(targets_aug.expand_as(preds)).cpu().sum()
                total += len(targets_aug)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_new_aug_cls {:.3f}, L_cont {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.init_epoch,
                    L_all / len(train_loader), 
                    L_new_cls  / len(train_loader), 
                    L_new_aug_cls / len(train_loader), 
                    L_cont / len(train_loader), 
                    train_acc,
                    test_acc,
                )
            else:
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_new_aug_cls {:.3f}, L_cont {:.3f}, Train_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.init_epoch,
                    L_all / len(train_loader), 
                    L_new_cls  / len(train_loader),
                    L_new_aug_cls / len(train_loader), 
                    L_cont / len(train_loader), 
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        if hasattr(self._network, "module"):
            _network = self._network.module
        else:
            _network = self._network
        
        prog_bar = tqdm(range(self.epochs), colour='red', position=self._process_id, dynamic_ncols=True, ascii=" =", leave=True)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            L_all = 0.0
            L_new_cls = 0.0
            L_new_cls_aug = 0.0
            L_old_cls = 0.0 
            L_kd = 0.0
            L_trsf = 0.0
            L_cont = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                loss_new_cls_aug, loss_new_cls, loss_kd, loss_cont, loss_transfer, loss_old_cls = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs_aug, targets_aug, _ = self._class_aug(inputs,targets)
                fea_aug = _network(inputs_aug)["features"]
                logits_aug = _network.fc_aug(fea_aug)["logits"]
                fea_aug_old = self._old_network(inputs_aug)["features"]
                loss_new_cls_aug = F.cross_entropy(logits_aug/self.args["temp"], targets_aug)
                L_new_cls_aug += loss_new_cls_aug.item()
                loss_kd = self.l2loss(fea_aug, fea_aug_old, mean=False) * self.w_kd
                L_kd += loss_kd.item()
                fea_aug_transfer = _network.drift_estimator(fea_aug_old)["logits"]
                loss_transfer = self.l2loss(fea_aug, fea_aug_transfer) * self.w_trsf
                L_trsf += loss_transfer.item()
                index = np.random.choice(range(self._known_classes),size=self.args["batch_size"],replace=True)
                proto_targets = np.array(range(self._known_classes))
                proto_features_raw = torch.tensor(np.array(self._protos)).to(self._device).to(torch.float32)
                
                trsf_proto = _network.drift_estimator(proto_features_raw)['logits'].detach().clone()
                drift_strength = torch.norm(trsf_proto - proto_features_raw, p=2, dim=1)  # [N]

                if self.use_class_aware_weight:
                    drift_weight = drift_strength / (drift_strength.max() + 1e-8)  # [N]
                    drift_weight = self.scale_factor * torch.sigmoid(drift_weight)
                else:
                    drift_weight = torch.ones_like(drift_strength)

                trsf_proto2 = proto_features_raw + drift_weight.unsqueeze(1)  * (trsf_proto - proto_features_raw)
                proto_aug_fea = trsf_proto2[index] + torch.from_numpy(np.random.normal(0,1,trsf_proto2[index].shape)).float().to(self._device) * self._radius
                proto_targets = torch.from_numpy(proto_targets[index] * 4).to(self._device)
                
                if self.use_multigranularity is False:
                    _logits = _network.fc_aug(proto_aug_fea)["logits"][:,:self._total_classes*4]
                    loss_old_cls = self.w_old_cls * F.cross_entropy(_logits/self.args["temp"], proto_targets)
                else:
                    with torch.no_grad():
                        ball_centers_orig, ball_labels_orig, ball_radius_orig = self.granularball_list_obj.get_one_center_label_radius_enlarged_per_class(start_class=0, end_class=self._known_classes, cur_task=self._cur_task, blur_factor=self.blur_factor)
                        
                        while True:
                            if len(ball_centers_orig) < len(inputs):
                                ball_centers_orig_new, ball_labels_orig_new, ball_radius_orig_new = self.granularball_list_obj.get_one_center_label_radius_enlarged_per_class(start_class=0, end_class=self._known_classes, cur_task=self._cur_task, blur_factor=self.blur_factor)
                                ball_centers_orig = np.vstack((ball_centers_orig, ball_centers_orig_new))
                                ball_labels_orig = np.concatenate((ball_labels_orig,  ball_labels_orig_new))
                                ball_radius_orig = np.concatenate((ball_radius_orig,  ball_radius_orig_new))
                            else:
                                break
                    ball_centers_orig = torch.from_numpy(ball_centers_orig).to(self._device).to(torch.float32)
                    ball_labels = torch.from_numpy(ball_labels_orig).to(self._device)

                    trsf_ball_center = self._network.drift_estimator(ball_centers_orig)['logits'].detach().clone()  # [N, D]
                    trsf_ball_center2 = ball_centers_orig + drift_weight[ball_labels].unsqueeze(1) * (trsf_ball_center - ball_centers_orig)  # [N, D]
                    
                    ball_aug_fea =  trsf_ball_center2 + torch.from_numpy(np.random.normal(0,1,trsf_ball_center2.shape)).float().to(self._device) * torch.from_numpy(ball_radius_orig).to(self._device).to(torch.float32).unsqueeze(1)
                    random_idx = torch.randperm(len(trsf_ball_center2))[:min(len(trsf_ball_center2), self.args["batch_size"])]
                    ball_aug_fea = ball_aug_fea[random_idx]
                    ball_labels2 = ball_labels[random_idx]*4

                    combined_features = torch.cat([ball_aug_fea, proto_aug_fea], dim=0)
                    combined_targets = torch.cat([ball_labels2, proto_targets], dim=0)     

                    combined_logits = _network.fc_aug(combined_features)["logits"][:,:self._total_classes*4]
                    loss_old_cls = F.cross_entropy(combined_logits/self.args["temp"], combined_targets) * self.w_old_cls
                
                L_old_cls += loss_old_cls.item()
                
                if self.use_multigranularity is False:
                    loss_cont = _contrastive_loss(fea_aug, targets_aug, torch.cat((fea_aug, proto_aug_fea), dim=0), torch.cat((targets_aug, proto_targets), dim=0)) * self.w_cont
                    L_cont += loss_cont.item()
                else:
                    loss_cont = _contrastive_loss(fea_aug, targets_aug, torch.cat((fea_aug, ball_aug_fea, proto_aug_fea), dim=0), torch.cat((targets_aug, ball_labels2, proto_targets), dim=0)) * self.w_cont
                    L_cont += loss_cont.item()

                loss =  loss_new_cls_aug + loss_kd + loss_transfer + loss_old_cls + loss_cont

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                L_all += loss.item()

                with torch.no_grad():
                    _, preds = torch.max(logits_aug, dim=1)
                    correct += preds.eq(targets_aug.expand_as(preds)).cpu().sum()
                    total += len(targets_aug)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_new_cls_aug {:.3f}, L_kd {:.3f}, L_trsf {:.3f},  L_old_cls {:.3f}, L_cont {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.epochs,
                    L_all / len(train_loader), 
                    L_new_cls  / len(train_loader), 
                    L_new_cls_aug / len(train_loader), 
                    L_kd  / len(train_loader), 
                    L_trsf  / len(train_loader), 
                    L_old_cls  / len(train_loader),
                    L_cont / len(train_loader), 
                    train_acc,
                    test_acc,
                )
            else:
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_new_cls_aug {:.3f}, L_kd {:.3f}, L_trsf {:.3f},  L_old_cls {:.3f}, L_cont {:.3f}, Train_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.epochs,
                    L_all / len(train_loader), 
                    L_new_cls  / len(train_loader), 
                    L_new_cls_aug / len(train_loader), 
                    L_kd  / len(train_loader), 
                    L_trsf  / len(train_loader), 
                    L_old_cls / len(train_loader),
                    L_cont / len(train_loader), 
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)

    def _class_aug(self,inputs,targets,alpha=20., mix_time=4,inputs_aug=None):
        inputs2 = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
        inputs2 = inputs2.view(-1, 3, inputs2.shape[-2], inputs2.shape[-1])
        targets2 = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)
        inputs_aug2 = None
        if inputs_aug is not None:
            inputs_aug2 = torch.stack([torch.rot90(inputs_aug, k, (2, 3)) for k in range(4)], 1)
            inputs_aug2 = inputs_aug2.view(-1, 3, inputs_aug2.shape[-2], inputs_aug2.shape[-1])
        
        mixup_inputs = []
        mixup_targets = []

        for _ in range(mix_time):
            index = torch.randperm(inputs.shape[0])
            perm_inputs = inputs[index]
            perm_targets = targets[index]
            mask = perm_targets!= targets

            select_inputs = inputs[mask]
            select_targets = targets[mask]
            perm_inputs = perm_inputs[mask]
            perm_targets = perm_targets[mask]

            lams = np.random.beta(alpha,alpha,sum(mask))
            lams = np.where((lams<0.4)|(lams>0.6),0.5,lams)
            lams = torch.from_numpy(lams).to(self._device)[:,None,None,None].float()

            mixup_inputs.append(lams*select_inputs+(1-lams)*perm_inputs)
            mixup_targets.append(self._map_targets(select_targets,perm_targets))
            
        mixup_inputs = torch.cat(mixup_inputs,dim=0)
        mixup_targets = torch.cat(mixup_targets,dim=0)
        inputs = torch.cat([inputs2,mixup_inputs],dim=0)
        targets = torch.cat([targets2,mixup_targets],dim=0)

        return inputs,targets, inputs_aug2
    
    def _map_targets(self,select_targets,perm_targets):
        assert (select_targets != perm_targets).all()
        large_targets = torch.max(select_targets,perm_targets)-self._known_classes
        small_targets = torch.min(select_targets,perm_targets)-self._known_classes

        mixup_targets = (large_targets*(large_targets-1)/2  + small_targets + self._total_classes*4).long()
        return mixup_targets
    
    def l2loss(self,inputs,targets,mean=True):
        if not mean :
            delta = torch.sqrt(torch.sum( torch.pow(inputs-targets,2) ))
            return delta 
        else :
            delta = torch.sqrt(torch.sum( torch.pow(inputs-targets,2),dim=-1 ))
            return torch.mean(delta)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                fea = model(inputs)["features"]
                outputs = model.fc_aug(fea)["logits"][:,:self._total_classes*4][:,::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def eval_task(self, save_conf=False):
        self._network.to(self._device)
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        y_pred, y_true = self._eval_nme(self.test_loader, self._protos)
        nme_accy = self._evaluate(y_pred, y_true)

        y_pred, y_true = self._eval_ncm(self.test_loader, self._protos)
        ncm_accy = self._evaluate(y_pred, y_true)

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")
    
        return cnn_accy, nme_accy, ncm_accy

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                fea = self._network(inputs)["features"]
                outputs = self._network.fc_aug(fea)["logits"][:,:self._total_classes*4][:,::4]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true) 
    
    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance
        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _eval_ncm(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        
        norms = np.linalg.norm(class_means, axis=1, keepdims=True) 
        class_means = class_means / (norms + EPSILON)

        dists = cdist(class_means, vectors, "sqeuclidean")
        scores = dists.T 
        return np.argsort(scores, axis=1)[:, :self.topk], y_true



def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def _contrastive_loss(X1, labels1,  X2, labels2):
    cosine_sim = F.cosine_similarity(X1.unsqueeze(1), X2.unsqueeze(0), dim=2) / T 
    exp_cosine_sim =  torch.exp(cosine_sim)
    same_class_mask = (labels1.unsqueeze(1) == labels2.unsqueeze(0))
    numerator = exp_cosine_sim * same_class_mask 
    numerator_mean = numerator.sum(dim=1) / (same_class_mask.sum(dim=1) + 1e-8) 
    denominator, _ = torch.max(exp_cosine_sim * (~same_class_mask), dim=1,) 
    result = -torch.log(numerator_mean / (numerator_mean + denominator + 1e-8))
    return result.mean()



def visualize_tsne(sample_fea, labels, ball_centers, ball_labels, args, task_id):

    save_path = "results/{}/{}_{}/2Dplot/".format(args["dataset"], args["init_cls"], args["increment"])
    os.makedirs(save_path, exist_ok=True)

    all_features = np.vstack([sample_fea, ball_centers])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_features = tsne.fit_transform(all_features)

    reduced_sample_fea = reduced_features[:len(sample_fea)]
    reduced_ball_centers = reduced_features[len(sample_fea):]

    unique_labels = np.unique(np.concatenate([labels, ball_labels]))
    num_classes = len(unique_labels)
    color_list = plt.cm.get_cmap("tab10", num_classes) 
    norm = mcolors.BoundaryNorm(boundaries=np.arange(num_classes+1)-0.5, ncolors=num_classes)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_sample_fea[:, 0], reduced_sample_fea[:, 1], 
                          c=labels, cmap=color_list, norm=norm, alpha=0.7, label="Samples")

    plt.scatter(reduced_ball_centers[:, 0], reduced_ball_centers[:, 1], 
                marker="*", s=200, c=ball_labels, cmap=color_list, norm=norm, 
                edgecolors="black", linewidth=1.5, label="Prototypes")

    cbar = plt.colorbar(scatter, ticks=np.arange(num_classes))
    cbar.set_label("Class Labels")
    cbar.set_ticks(np.arange(num_classes))
    cbar.set_ticklabels(unique_labels) 


    plt.legend()
    plt.title("t-SNE Visualization with Class Prototypes")
    plt.savefig(f"{save_path}/{int(time.time())}_task{task_id}.png")
