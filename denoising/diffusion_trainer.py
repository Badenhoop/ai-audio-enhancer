from distutils.command.config import config
import torch
import torch.nn as nn
import numpy as np
import os
import wandb
from tqdm import tqdm
from config import Config

# TODO:
#   - Save config to disk and in wandb
#   - Understand relevance of fp16
class DiffusionTrainer:
    def __init__(self,
                 work_dir,
                 model,
                 data_loader,
                 optimizer,
                 noise_schedule,
                 group_name,
                 run_name,
                 config):
        self.work_dir = work_dir
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.config = config
        noise_level = np.cumprod(1 - noise_schedule)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        self.loss_fn = nn.MSELoss()
        self.step = 1
        self.group_name = group_name
        self.run_name = run_name
        self.run_id = None
    
    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return dict(
            step=self.step,
            model={ k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
            optimizer={ k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
            run_id=self.run_id,
            config=self.config)
    
    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.step = state_dict['step']
        self.run_id = state_dict['run_id']
        self.config = Config(state_dict['config'])

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pth'
        save_name = f'{self.work_dir}/{save_basename}'
        link_name = f'{self.work_dir}/{filename}.pth'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.work_dir}/{filename}.pth')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self, steps):
        self._init_wandb()
        device = next(self.model.parameters()).device
        while True:
            for batch in tqdm(self.data_loader, desc=f'Epoch {(self.step - 1) // len(self.data_loader)}'):
                if self.step > steps:
                    return
                x = batch.to(device)
                loss = self._train_step(x)
                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')
                if self.step % 5 == 0 or self.step == steps:
                    self._write_summary(batch, loss)
                if self.step % len(self.data_loader) == 0 or self.step == steps:
                    self.save_to_checkpoint()
                self.step += 1

    def _init_wandb(self):
        run = wandb.init(
            name=self.run_name,
            group=self.group_name,
            config=self.config,
            id=self.run_id,
            resume='auto')
        self.run_id = run.id
    
    def _train_step(self, x):
        N, T = x.shape
        device = x.device
        self.noise_level = self.noise_level.to(device)
        t = torch.randint(0, len(self.noise_level), [N], device=x.device)
        noise_scale = self.noise_level[t].unsqueeze(1)
        noise = torch.randn_like(x)
        noisy_audio = noise_scale**0.5 * x + (1.0 - noise_scale)**0.5 * noise

        self.optimizer.zero_grad()
        predicted = self.model(noisy_audio, t)
        loss = self.loss_fn(noise, predicted)
        loss.backward()
        self.optimizer.step()
        return loss

    def _write_summary(self, batch, loss):
        data = { 'loss': loss }
        wandb.log(
            data=data,
            step=self.step - 1,
            commit=True)