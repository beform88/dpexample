import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers.optimization import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from ..data import DataHub, DataReader, TargetScaler
from ..descriptior.unimol_tools.unimol_tools.models.unimol import UniMolModel
from ..descriptior.unimol_tools.unimol_tools.utils import pad_1d_tokens, pad_2d, pad_coords
from tqdm import tqdm
import time

from torch.nn.utils import clip_grad_norm_
from ..utils import Metrics, logger
import os

NNDATALOADER_REGISTER = {

}

FINETUNE_MODELS = {
    'unimol': UniMolModel(output_dim=1, data_type='molecule', remove_hs=False)
}

class Trainer(object):
    def __init__(self, **kwargs):
        self.task = kwargs.get('task','regression')
        self.metrics_str = kwargs.get('metrics_str','none')
        self.metrics = Metrics(self.task, self.metrics_str, **kwargs)
        self._init_trainer(**kwargs)

    def _init_trainer(self,**kwargs):
        self.max_epoch = kwargs.get('max_epoch',1)
        self.learning_rate = kwargs.get('learning_rate',1e-4)
        self.seed = kwargs.get('seed',42)
        self.set_seed(self.seed)
        self.batch_size = kwargs.get('batch_size',32)
        self.warmup_ratio = kwargs.get('warmup_ratio',0.1)
        self.patience = kwargs.get('patience', 10)
        self.max_norm = kwargs.get('max_norm', 1.0)
        self.cuda = kwargs.get('cuda', True)
        self.amp = kwargs.get('amp', True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' and self.amp==True else None

    def decorate_batch(self, batch):
        net_input, net_target = batch
        if isinstance(net_input[0],dict):
            return net_input, net_target.to(self.device)
        else:
            return net_input.to(self.device), net_target.to(self.device)

    def fit_predict(self,train_dataset, valid_dataset, model, loss_func, dump_dir, target_scaler):
        model = model.to(self.device)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,shuffle=True,collate_fn=model.batch_collate_fn,drop_last=True)
        min_val_loss = float("inf")
        max_score = float("-inf")
        wait = 0
        num_training_steps = len(train_dataloader) * self.max_epoch
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)

        # optimizer
        params = model.parameters()
        optimizer = Adam(params, lr=self.learning_rate, eps=1e-6) #1e-6

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        for epoch in range(self.max_epoch):
            model.train()
            start_time = time.time()
            batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)
            train_loss = []
            for i, batch in enumerate(train_dataloader):
                net_input, net_target = self.decorate_batch(batch)
                # net_input = net_input.detach()
                optimizer.zero_grad()
                if self.scaler and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(net_input)
                        loss = loss_func(outputs, net_target)
                else:
                    with torch.set_grad_enabled(True):
                        outputs = model(net_input)
                        loss = loss_func(outputs, net_target)
                train_loss.append(float(loss.data))
                batch_bar.set_postfix(
                        Epoch="Epoch {}/{}".format(epoch+1, self.max_epoch),
                        loss="{:.04f}".format(float(sum(train_loss) / (i + 1))),
                        lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
                
                if self.scaler and self.device.type == 'cuda':
                    loss = self.scaler.scale(loss).backward() # This is a replacement for loss.backward()
                    self.scaler.unscale_(optimizer) # unscale the gradients of optimizer's assigned params in-place
                    clip_grad_norm_(params, self.max_norm)  # Clip the norm of the gradients to max_norm.
                    self.scaler.step(optimizer) # This is a replacement for optimizer.step()
                    self.scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.max_norm)
                    optimizer.step()
                
                scheduler.step()
                batch_bar.update()

            batch_bar.close()
            total_train_loss = np.mean(train_loss)

            y_preds, val_loss, metric_score = self.predict(model, valid_dataset, loss_func, dump_dir, target_scaler, epoch, load_model=False)
            end_time = time.time()
            total_val_loss = np.mean(val_loss)
            _score = list(metric_score.values())[0]
            _metric = list(metric_score.keys())[0]
            message = self.print_message(epoch, self.max_epoch, total_train_loss, total_val_loss, _metric, _score, optimizer.param_groups[0]['lr'], (end_time - start_time), self.task)
            logger.info(message)
            is_early_stop, min_val_loss, wait, max_score = self._early_stop_choice(wait, total_val_loss, min_val_loss, metric_score, max_score, model, dump_dir, self.patience, epoch)
            if is_early_stop:
                break

        y_preds, _, _ = self.predict(model, valid_dataset, loss_func, dump_dir, target_scaler, epoch, load_model=True)
        return y_preds
    
    def predict(self, model, dataset, loss_func, dump_dir, target_scaler=None, epoch=1, load_model=False):
        model = model.to(self.device)
        if load_model == True:
            load_model_path = os.path.join(dump_dir, f'model.pth')
            model_dict = torch.load(load_model_path, map_location=self.device)["model_state_dict"]
            model.load_state_dict(model_dict)
            logger.info("load model success!")
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size,  shuffle=False, collate_fn=model.batch_collate_fn,)
        model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='val', ncols=5)
        val_loss = []
        y_preds = []
        y_truths = []
        for i, batch in enumerate(dataloader):
            net_input, net_target = self.decorate_batch(batch)
            # Get model outputs
            with torch.no_grad():
                outputs = model(net_input)
                if not load_model:
                    loss = loss_func(outputs, net_target)
                    val_loss.append(float(loss.data))
            y_preds.append(outputs.cpu().numpy())
            y_truths.append(net_target.detach().cpu().numpy())
            if not load_model:
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epoch),
                    loss="{:.04f}".format(float(np.sum(val_loss) / (i + 1))))

            batch_bar.update()
            
        y_preds = np.concatenate(y_preds)
        y_truths = np.concatenate(y_truths)
        if target_scaler is not None:
            inverse_y_preds = target_scaler.inverse_transform(y_preds)
            inverse_y_truths = target_scaler.inverse_transform(y_truths)
            metric_score = self.metrics.cal_metric(inverse_y_truths, inverse_y_preds) if not load_model else None
        else:
            metric_score = self.metrics.cal_metric(y_truths, y_preds) if not load_model else None
        batch_bar.close()
        return y_preds, val_loss, metric_score
    
    def fit_predict_finetune(self,train_dataset, valid_dataset, model, finetune_models, loss_func, dump_dir, target_scaler):
        model = model.to(self.device)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,shuffle=True,collate_fn=model.batch_collate_fn,drop_last=True)
        min_val_loss = float("inf")
        max_score = float("-inf")
        wait = 0
        num_training_steps = len(train_dataloader) * self.max_epoch
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)

        # optimizer
        if finetune_models == None:
            params = model.parameters()
            optimizer = Adam(params, lr=self.learning_rate, eps=1e-6) #1e-6
        else:
            # optimizer = Adam(model.parameters(), lr=self.learning_rate, eps=1e-6)

            # method00 创建不同的参数列表
            # params = [{'params':model.parameters()}]
            # for finetune_model in finetune_models.keys():
            #     params.append({'params':finetune_models[finetune_model].parameters()})

            # method01 合并参数列表
            params = list(model.parameters())
            self.model_list = {}
            for finetune_model in finetune_models.keys():
                # finetune models
                self.model_list[finetune_model] = FINETUNE_MODELS[finetune_model].to(self.device)
                # optimizer
                params = params + list(self.model_list[finetune_model].parameters())

            optimizer = Adam(params, lr=self.learning_rate, eps=1e-6)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        for epoch in range(self.max_epoch):
            model.train()
            start_time = time.time()
            batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)
            train_loss = []
            for i, batch in enumerate(train_dataloader):
                net_input, net_target = self.decorate_batch(batch)
                # net_input = net_input.detach()
                optimizer.zero_grad()
                if self.scaler and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        net_input = self.cat_inputs(net_input)
                        outputs = model(net_input)
                        loss = loss_func(outputs, net_target)
                else:
                    with torch.set_grad_enabled(True):
                        net_input = self.cat_inputs(net_input)
                        outputs = model(net_input)
                        loss = loss_func(outputs, net_target)
                train_loss.append(float(loss.data))
                batch_bar.set_postfix(
                        Epoch="Epoch {}/{}".format(epoch+1, self.max_epoch),
                        loss="{:.04f}".format(float(sum(train_loss) / (i + 1))),
                        lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
                
                if self.scaler and self.device.type == 'cuda':
                    loss = self.scaler.scale(loss).backward() # This is a replacement for loss.backward()
                    self.scaler.unscale_(optimizer) # unscale the gradients of optimizer's assigned params in-place
                    clip_grad_norm_(params, self.max_norm)  # Clip the norm of the gradients to max_norm.
                    self.scaler.step(optimizer) # This is a replacement for optimizer.step()
                    self.scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.max_norm)
                    optimizer.step()
                
                scheduler.step()
                batch_bar.update()

            batch_bar.close()
            total_train_loss = np.mean(train_loss)

            y_preds, val_loss, metric_score = self.predict_finetune(model, valid_dataset, loss_func, dump_dir, target_scaler, epoch, load_model=False)
            end_time = time.time()
            total_val_loss = np.mean(val_loss)
            _score = list(metric_score.values())[0]
            _metric = list(metric_score.keys())[0]
            message = self.print_message(epoch, self.max_epoch, total_train_loss, total_val_loss, _metric, _score, optimizer.param_groups[0]['lr'], (end_time - start_time), self.task)
            logger.info(message)
            is_early_stop, min_val_loss, wait, max_score = self._early_stop_choice(wait, total_val_loss, min_val_loss, metric_score, max_score, model, dump_dir, self.patience, epoch)
            if is_early_stop:
                break

        y_preds, _, _ = self.predict_finetune(model, valid_dataset, loss_func, dump_dir, target_scaler, epoch, load_model=True)
        return y_preds

    def predict_finetune(self, model, dataset, loss_func, dump_dir, target_scaler=None, epoch=1, load_model=False):
        model = model.to(self.device)
        if load_model == True:
            load_model_path = os.path.join(dump_dir, f'model.pth')
            model_dict = torch.load(load_model_path, map_location=self.device)["model_state_dict"]
            model.load_state_dict(model_dict)
            logger.info("load model success!")
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size,  shuffle=False, collate_fn=model.batch_collate_fn,)
        model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='val', ncols=5)
        val_loss = []
        y_preds = []
        y_truths = []
        for i, batch in enumerate(dataloader):
            net_input, net_target = self.decorate_batch(batch)
            # Get model outputs
            with torch.no_grad():
                net_input = self.cat_inputs(net_input,train =False)
                outputs = model(net_input)
                if not load_model:
                    loss = loss_func(outputs, net_target)
                    val_loss.append(float(loss.data))
            y_preds.append(outputs.cpu().numpy())
            y_truths.append(net_target.detach().cpu().numpy())
            if not load_model:
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epoch),
                    loss="{:.04f}".format(float(np.sum(val_loss) / (i + 1))))

            batch_bar.update()
            
        y_preds = np.concatenate(y_preds)
        y_truths = np.concatenate(y_truths)
        if target_scaler is not None:
            inverse_y_preds = target_scaler.inverse_transform(y_preds)
            inverse_y_truths = target_scaler.inverse_transform(y_truths)
            metric_score = self.metrics.cal_metric(inverse_y_truths, inverse_y_preds) if not load_model else None
        else:
            metric_score = self.metrics.cal_metric(y_truths, y_preds) if not load_model else None
        batch_bar.close()
        return y_preds, val_loss, metric_score

    def print_message(self, epoch, max_epoch, total_trn_loss, total_val_loss, _metric, _score, lr, time, task):
        if task != 'multilabel_regression':
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}, lr: {:.6f}, ' \
                '{:.1f}s'.format(epoch+1, max_epoch,
                                total_trn_loss, total_val_loss, 
                                _metric, _score,
                                lr,
                                time)
        else:
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, '.format(epoch+1, max_epoch,total_trn_loss, total_val_loss)
                for i in range(len(_score)):
                    val_mae = _score[i]
                    message += 'val_{}_{}: {:.14f}, '.format(_metric, i, val_mae)
                message += 'lr: {:.6f}, {:.1f}s'.format(lr, time)
        return message

    def _early_stop_choice(self, wait, loss, min_loss, metric_score, max_score, model, dump_dir, patience, epoch):
        ### hpyerparameter need to tune if you want to use early stop, currently find use loss is suitable in benchmark test. ###
        if not isinstance(self.metrics_str, str) or self.metrics_str in ['loss', 'none', '']:
            ## loss 作为早停 直接用trainer里面的早停函数
            is_early_stop, min_val_loss, wait = self._judge_early_stop_loss(wait, loss, min_loss, model, dump_dir, patience, epoch)
        else:
            ## 到metric进行判断
            is_early_stop, min_val_loss, wait, max_score = self.metrics._early_stop_choice(wait, min_loss, metric_score, max_score, model, dump_dir, patience, epoch)
        return is_early_stop,min_val_loss,wait, max_score

    def _judge_early_stop_loss(self, wait, loss, min_loss, model, dump_dir, patience, epoch):
        is_early_stop = False
        if loss <= min_loss :
            min_loss = loss
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model.pth'))
        elif loss >= min_loss:
            wait += 1
            if wait == self.patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_loss, wait

    def set_seed(self, seed):
        """function used to set a random seed
        Arguments:
            seed {int} -- seed number, will set to torch and numpy
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def cat_inputs(self,net_inputs,train = True):
        
        with torch.cuda.amp.autocast():
            batch = {}
            for k in net_inputs[0].keys():
                if k == 'structure': 
                    continue
                elif k == 'unimol':
                    batch[k] = {}
                    for dim in net_inputs[0][k].keys():
                        if dim == 'src_coord':
                            batch[k][dim] = [item[k][dim][0] for item in net_inputs]
                            batch[k][dim] = pad_coords(batch[k][dim], pad_idx=0.0).to(self.device)
                        elif dim == 'src_edge_type':
                            batch[k][dim] = [item[k][dim][0] for item in net_inputs]
                            batch[k][dim] = pad_2d(batch[k][dim], pad_idx=0).to(self.device)
                        elif dim == 'src_distance':
                            batch[k][dim] = [item[k][dim][0] for item in net_inputs]
                            batch[k][dim] = pad_2d(batch[k][dim], pad_idx=0.0).to(self.device)
                        elif dim == 'src_tokens':
                            batch[k][dim] = [item[k][dim][0] for item in net_inputs]
                            batch[k][dim] = pad_1d_tokens(batch[k][dim], pad_idx=0).to(self.device)
                else:
                    batch[k] = torch.stack([torch.tensor(item[k]) for item in net_inputs],dim=0)
                
            final_input = batch['base'].to(self.device)
            for model_name in self.model_list.keys():
                if model_name == 'unimol':
                    model_output = self.model_list[model_name](return_repr=True, **batch[model_name])
                    model_output = torch.stack([torch.cat([model_output['cls_repr'][i],model_output['atomic_reprs'][i][batch['atom'][i]-1]],axis = 0) 
                                    for i in range(len(model_output['cls_repr']))],dim = 0)

                final_input = torch.cat([final_input,model_output], dim = 1)

            if train:
                return final_input.half()
            else:
                return final_input.float()
