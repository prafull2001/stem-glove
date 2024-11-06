# import os
# import torch
# import torchaudio
# from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS, HDEMUCS_HIGH_MUSDB
# from torchaudio.transforms import Fade

# # Load the pre-trained model
# # bundle = CONVTASNET_BASE_LIBRI2MIX
# # bundle = HDEMUCS_HIGH_MUSDB_PLUS
# bundle = HDEMUCS_HIGH_MUSDB
# model = bundle.get_model()

# # Use CPU or GPU depending on your environment
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Sample rate of the model
# sample_rate = bundle.sample_rate
# print(f"Sample rate: {sample_rate}")

# # Function for separating the sources
# def separate_sources(model, mix, segment=10.0, overlap=0.1, device=None):
#     if device is None:
#         device = mix.device
#     else:
#         device = torch.device(device)

#     batch, channels, length = mix.shape
#     chunk_len = int(sample_rate * segment * (1 + overlap))
#     start = 0
#     end = chunk_len
#     overlap_frames = overlap * sample_rate
#     fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

#     final = torch.zeros(batch, len(model.sources), channels, length, device=device)

#     while start < length - overlap_frames:
#         chunk = mix[:, :, start:end]
#         with torch.no_grad():
#             out = model(chunk)
#         out = fade(out)
#         final[:, :, :, start:end] += out
#         if start == 0:
#             fade.fade_in_len = int(overlap_frames)
#             start += int(chunk_len - overlap_frames)
#         else:
#             start += chunk_len
#         end += chunk_len
#         if end >= length:
#             fade.fade_out_len = 0
#     return final

# # Load your audio file (make sHDEMUCS_HIGH_MUSDB_PLUSure to change the path to your local file path)
# SAMPLE_SONG = "/Users/prafullsharma/Desktop/stem-glove/RAF.wav"
# waveform, sample_rate = torchaudio.load(SAMPLE_SONG)

# # Resample if necessary
# if sample_rate != bundle.sample_rate:
#     print(f"Resampling from {sample_rate} Hz to {bundle.sample_rate} Hz")
#     resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
#     waveform = resample(waveform)
#     sample_rate = bundle.sample_rate

# waveform = waveform.to(device)

# # Normalization
# ref = waveform.mean(0)
# waveform = (waveform - ref.mean()) / ref.std()

# # Separate the sources
# sources = separate_sources(
#     model,
#     waveform[None],
#     device=device,
#     segment=10,
#     overlap=0.1,
# )[0]
# sources = sources * ref.std() + ref.mean()

# # Extract individual sources
# audios = dict(zip(model.sources, list(sources)))

# # Create the subfolder to save stems
# song_name = os.path.splitext(os.path.basename(SAMPLE_SONG))[0]  # Extract the song name without extension
# output_dir = os.path.join(os.getcwd(), "stems", f"{song_name}-stems")  # Create the path for the subfolder
# os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# # Save the separated sources
# for source_name, audio in audios.items():
#     output_path = os.path.join(output_dir, f"{source_name}_output.wav")
#     torchaudio.save(output_path, audio.cpu(), sample_rate)
#     print(f"{source_name.capitalize()} saved to {output_path}")







# import os
# import torch
# import torchaudio
# import numpy as np
# import scipy.signal
# from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS, HDEMUCS_HIGH_MUSDB
# from torchaudio.transforms import Fade

# # Load the pre-trained model
# bundle = HDEMUCS_HIGH_MUSDB
# model = bundle.get_model()

# # Use CPU or GPU depending on your environment
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Sample rate of the model
# sample_rate = bundle.sample_rate
# print(f"Sample rate: {sample_rate}")

# # Function for separating the sources
# def separate_sources(model, mix, segment=10.0, overlap=0.1, device=None):
#     if device is None:
#         device = mix.device
#     else:
#         device = torch.device(device)

#     batch, channels, length = mix.shape
#     chunk_len = int(sample_rate * segment * (1 + overlap))
#     start = 0
#     end = chunk_len
#     overlap_frames = overlap * sample_rate
#     fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

#     final = torch.zeros(batch, len(model.sources), channels, length, device=device)

#     while start < length - overlap_frames:
#         chunk = mix[:, :, start:end]
#         with torch.no_grad():
#             out = model(chunk)
#         out = fade(out)
#         final[:, :, :, start:end] += out
#         if start == 0:
#             fade.fade_in_len = int(overlap_frames)
#             start += int(chunk_len - overlap_frames)
#         else:
#             start += chunk_len
#         end += chunk_len
#         if end >= length:
#             fade.fade_out_len = 0
#     return final

# # Function to apply Wiener filter to audio data
# def apply_wiener_filter(audio):
#     # Convert to numpy array for Wiener filtering
#     audio_np = audio.cpu().numpy()
#     # Apply Wiener filter along each channel
#     filtered_audio = np.apply_along_axis(scipy.signal.wiener, axis=-1, arr=audio_np)
#     # Convert back to Torch tensor and ensure type is float32
#     return torch.tensor(filtered_audio, dtype=torch.float32, device=audio.device)

# # def apply_wiener_filter(audio, passes=1, window_size=5):
# #     audio_np = audio.cpu().numpy()
# #     for _ in range(passes):
# #         # Apply Wiener filter multiple times
# #         audio_np = np.apply_along_axis(scipy.signal.wiener, axis=-1, arr=audio_np, mysize=window_size)
# #     return torch.tensor(audio_np, dtype=torch.float32, device=audio.device)


# # Load your audio file
# SAMPLE_SONG = "/Users/prafullsharma/Desktop/stem-glove/RAF.wav"
# waveform, sample_rate = torchaudio.load(SAMPLE_SONG)

# # Resample if necessary
# if sample_rate != bundle.sample_rate:
#     print(f"Resampling from {sample_rate} Hz to {bundle.sample_rate} Hz")
#     resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
#     waveform = resample(waveform)
#     sample_rate = bundle.sample_rate

# waveform = waveform.to(device)

# # Normalization
# ref = waveform.mean(0)
# waveform = (waveform - ref.mean()) / ref.std()

# # Separate the sources
# sources = separate_sources(
#     model,
#     waveform[None],
#     device=device,
#     segment=10,
#     overlap=0.1,
# )[0]
# sources = sources * ref.std() + ref.mean()

# # Extract individual sources
# audios = dict(zip(model.sources, list(sources)))

# # Apply Wiener filter to each separated source
# filtered_audios = {source_name: apply_wiener_filter(audio) for source_name, audio in audios.items()}

# # Create the subfolder to save stems
# song_name = os.path.splitext(os.path.basename(SAMPLE_SONG))[0]  # Extract the song name without extension
# output_dir = os.path.join(os.getcwd(), "stems", f"{song_name}-stems")  # Create the path for the subfolder
# os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# # Save the separated and Wiener-filtered sources
# for source_name, audio in filtered_audios.items():
#     # Ensure the tensor is cast to float32 before saving
#     audio = audio.to(torch.float32)
#     output_path = os.path.join(output_dir, f"{source_name}_output.wav")
#     torchaudio.save(output_path, audio.cpu(), sample_rate)
#     print(f"{source_name.capitalize()} saved to {output_path}")






import os
import torch
import torchaudio
import numpy as np
import scipy.signal
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
from torchaudio.transforms import Fade

# Load the pre-trained model
bundle = HDEMUCS_HIGH_MUSDB
model = bundle.get_model()

# Use CPU or GPU depending on your environment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Sample rate of the model
sample_rate = bundle.sample_rate
print(f"Sample rate: {sample_rate}")

# Function for separating the sources
def separate_sources(model, mix, segment=10.0, overlap=0.1, device=None):
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape
    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final

# Function to apply high-pass filter using scipy
def apply_high_pass(audio, cutoff_freq=100, sample_rate=44100):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = scipy.signal.butter(1, normal_cutoff, btype='high', analog=False)
    return torch.tensor(scipy.signal.lfilter(b, a, audio.cpu().numpy()), device=audio.device)

# Function to apply low-pass filter using scipy
def apply_low_pass(audio, cutoff_freq=5000, sample_rate=44100):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = scipy.signal.butter(1, normal_cutoff, btype='low', analog=False)
    return torch.tensor(scipy.signal.lfilter(b, a, audio.cpu().numpy()), device=audio.device)

# Load your audio file
SAMPLE_SONG = "/Users/prafullsharma/Desktop/stem-glove/impala.wav"
waveform, sample_rate = torchaudio.load(SAMPLE_SONG)

# Resample if necessary
if sample_rate != bundle.sample_rate:
    print(f"Resampling from {sample_rate} Hz to {bundle.sample_rate} Hz")
    resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
    waveform = resample(waveform)
    sample_rate = bundle.sample_rate

waveform = waveform.to(device)

# Normalization
ref = waveform.mean(0)
waveform = (waveform - ref.mean()) / ref.std()

# Separate the sources
sources = separate_sources(
    model,
    waveform[None],
    device=device,
    segment=10,
    overlap=0.1,
)[0]
sources = sources * ref.std() + ref.mean()

# Extract individual sources
audios = dict(zip(model.sources, list(sources)))

# Apply filters where appropriate
filtered_audios = {}
for source_name, audio in audios.items():
    if source_name == "bass":
        # Apply a low-pass filter to bass
        filtered_audio = apply_low_pass(audio, cutoff_freq=150, sample_rate=sample_rate)  # Cutoff for bass around 150Hz
    elif source_name == "vocals":
        # Apply a high-pass filter to vocals
        filtered_audio = apply_high_pass(audio, cutoff_freq=120, sample_rate=sample_rate)  # Cutoff for vocals around 120Hz
    elif source_name == "drums":
        # Apply both low-pass and high-pass filters for drums
        filtered_audio = apply_high_pass(audio, cutoff_freq=60, sample_rate=sample_rate)  # High-pass for low-end noise
        filtered_audio = apply_low_pass(filtered_audio, cutoff_freq=6000, sample_rate=sample_rate)  # Low-pass for high-end noise
    else:
        # For "other", leave it as is or add specific filters if needed
        filtered_audio = audio
    filtered_audios[source_name] = filtered_audio

# Create the subfolder to save stems
song_name = os.path.splitext(os.path.basename(SAMPLE_SONG))[0]  # Extract the song name without extension
output_dir = os.path.join(os.getcwd(), "stems", f"{song_name}-stems")  # Create the path for the subfolder
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Save the filtered stems
for source_name, audio in filtered_audios.items():
    # Ensure the tensor is cast to float32 before saving
    audio = audio.to(torch.float32)
    output_path = os.path.join(output_dir, f"{source_name}_output.wav")
    torchaudio.save(output_path, audio.cpu(), sample_rate)
    print(f"{source_name.capitalize()} saved to {output_path}")
