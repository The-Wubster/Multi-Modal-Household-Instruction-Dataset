import torch
import os
import glob
from pathlib import Path
from tqdm import tqdm

torch.set_num_threads(1)

os.chdir("/Volumes/Ryan_Extern/Normal_Dataset_Full/Training_Data/bound_images")
current_folder = Path('./recordings/')
new_folder = Path('./reduced_recordings/')

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)

sampling_rate = 16000
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils 

# Loop through the files in the source folder
for curr_wav_path in tqdm(current_folder.glob("*.wav")):
    wav = read_audio(curr_wav_path, sampling_rate=sampling_rate)
    # Get speech timestamps from the full audio file
    print(curr_wav_path)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    save_audio(new_folder/curr_wav_path.name, collect_chunks(speech_timestamps, wav), sampling_rate=sampling_rate)
