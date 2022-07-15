from argparse import ArgumentParser
import numpy as np
import torch
import os
from config import Config
from diffusion_trainer import DiffusionTrainer
from model import DiffusionUNetModel
from dataset import build_dataloader
from utils import set_seed
from uuid import uuid4


def train(args):
    torch.backends.cudnn.benchmark = True
    set_seed(0)

    config = Config.fromfile(args.config)
    
    noise_schedule = np.linspace(
        start=config.noise_schedule.start,
        stop=config.noise_schedule.stop,
        num=config.noise_schedule.num)
    num_diffusion_steps = len(noise_schedule)
    model = DiffusionUNetModel(num_diffusion_steps=num_diffusion_steps)
    model = model.cuda()

    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay)
    
    data_loader = build_dataloader(
        directory=config.data_dir,
        audio_format='mp4',
        batch_size=config.batch_size,
        audio_length=config.audio_length,
        shuffle=True)

    work_dir = f'work_dirs/{uuid4()}'
    os.makedirs(work_dir)

    trainer = DiffusionTrainer(
        work_dir=work_dir,
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        noise_schedule=noise_schedule,
        group_name=config.group_name,
        run_name=config.run_name,
        config=config)
    trainer.train(steps=config.steps)


if __name__ == '__main__':
    parser = ArgumentParser('Trains an audio diffusion model.')
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    train(args)