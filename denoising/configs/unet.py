batch_size = 4
learning_rate = 1e-4
weight_decay = 1e-4
sample_rate = 44100
audio_length = 5 * sample_rate
steps = 1000
training_noise_schedule = dict(start=1e-4, stop=0.05, num=50)
inference_noise_schedule = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]
run_name = '4 samples'
group_name = 'unet'
data_dir = 'data/4'