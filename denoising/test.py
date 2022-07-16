from argparse import ArgumentParser
import numpy as np
import torch
import torchaudio
import os
import torchaudio
from yaml import load
from model import DiffusionUNetModel
from dataset import load_audio
from config import Config
from utils import build_noise_schedule


def apply_noise(x0, t, noise_schedule):
    # See Appendix A of https://arxiv.org/pdf/2009.09761.pdf#cite.ho2020denoising
    beta = noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(talpha)
    noise = torch.randn_like(x0)
    xt = alpha_cum**0.5 * x0 + (1. - alpha_cum) * noise
    return xt


def reverse_noise(xt, t, noise_schedule, model):
    # Equation 5 in https://arxiv.org/pdf/2009.09761.pdf#cite.ho2020denoising
    beta = noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    xs = xt
    for s in range(t, -1, -1):
        c1 = 1 / alpha[n]**0.5
        c2 = beta[n] / (1 - alpha_cum[n])**0.5
        xs = c1 * (xs - c2 * model(xs, torch.tensor([t]), device=xs.device))
        if s > 0:
            noise = torch.randn_like(xs)
            sigma = ((1.0 - alpha_cum[s - 1]) / (1.0 - alpha_cum[s]) * beta[s])**0.5
            xs += sigma * noise
        xs = torch.clamp(xs, -1.0, 1.0)
    return xs


def main(args):
    if os.path.exists(f'{args.model_dir}/weights.pth'):
        checkpoint = torch.load(f'{args.model_dir}/weights.pth')
    else:
        checkpoint = torch.load(args.model_dir)
    
    config = Config(checkpoint['config'])

    noise_schedule = build_noise_schedule(config.training_noise_schedule)

    model = DiffusionUNetModel(num_diffusion_steps=len(noise_schedule)).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()

    start = int(args.offset * config.sample_rate)
    end = start + config.audio_length
    input_audio = load_audio(args.input_audio)
    input_audio = input_audio[start:end]

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with torch.no_grad():
        audio = torch.from_numpy(input_audio).unsqueeze(0).cuda()
        noise = torch.randn_like(audio).cuda()

        T = len(noise_schedule) - 1
        reverse_steps = [0] + [2**i for i in range(int(np.log2(T)))]
        for t in reverse_steps:
            audio_noisy = apply_noise(audio, t, noise_schedule)
            audio_denoised = reverse_noise(audio_noisy, t, beta, noise_schedule, model)
            torchaudio.save(f'{out_dir}/denoised-{t}.wav', audio_denoised.cpu(), sample_rate=config.sample_rate)


if __name__ == '__main__':
    parser = ArgumentParser(description='Tests an audio diffusion model.')
    parser.add_argument('input_audio', help='Input filename')
    parser.add_argument('out_dir', help='Output directory')
    parser.add_argument('model_dir', help='Directory containing a trained model (or full path to weights.pth file)')
    parser.add_argument('--offset', '-o', type=float, default=0., help='Offset of the target sequence in the given audio clip in seconds')
    main(parser.parse_args())
