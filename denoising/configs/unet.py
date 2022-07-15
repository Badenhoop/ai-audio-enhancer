batch_size = 4
learning_rate = 1e-4
weight_decay = 1e-4
audio_length = 5 * 44100
steps = 1000
noise_schedule = dict(start=1e-4, stop=0.05, num=50)
run_name = 'initial run'
group_name = 'unet'
data_dir = 'data'