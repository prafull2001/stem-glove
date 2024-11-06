import serial
import time
import pygame
import os

# Set up the serial connection (update the port based on your system)
ser = serial.Serial('/dev/cu.usbmodem14101', 9600)  # Update with your serial port

# Function to load audio files from the correct directory
def load_audio_files(song_name):
    # Remove the extension (.wav) from the song name to get the folder name
    song_name_no_ext = os.path.splitext(song_name)[0]
    base_folder = f"/Users/prafullsharma/Desktop/stem-glove/stems/{song_name_no_ext}-stems"

    # Define the paths for the 4 stem audio files
    audio_files = [
        os.path.join(base_folder, "bass_output.wav"),
        os.path.join(base_folder, "drums_output.wav"),
        os.path.join(base_folder, "other_output.wav"),
        os.path.join(base_folder, "vocals_output.wav")
    ]

    return audio_files

# Provide the song name (e.g., "impala.wav")
song_name = "impala.wav"
audio_files = load_audio_files(song_name)

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Load the 4 audio files
channels = [pygame.mixer.Sound(file) for file in audio_files]

# Initialize the audio channels and set them to play muted (volume = 0)
for i in range(4):
    channels[i].play(loops=-1)  # Play each file on loop
    channels[i].set_volume(0)    # Start muted

def map_fsr_to_volume(fsr_value):
    # Map FSR values (0 to 1023) to volume (0 to 1)
    return min(fsr_value / 1023, 1.0)

try:
    while True:
        if ser.in_waiting > 0:
            # Read the data from Arduino and decode the byte string
            data = ser.readline().decode('utf-8').strip()

            # Check if the data is valid and not an empty string
            if data:
                fsr_values = data.split(",")
                
                # Only proceed if we have exactly 4 FSR values
                if len(fsr_values) == 4:
                    try:
                        fsr_values = [int(x) for x in fsr_values]  # Convert to integers
                        
                        # Set volume based on FSR values
                        for i in range(4):
                            volume = map_fsr_to_volume(fsr_values[i])
                            channels[i].set_volume(volume)
                    
                    except ValueError:
                        # Silently ignore any invalid data
                        pass

        time.sleep(0.01)  # Small delay to avoid overwhelming the serial buffer

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    ser.close()  # Close the serial connection when done
    pygame.mixer.quit()  # Stop audio playback



# import serial
# import time
# import pygame
# import os

# # Set up the serial connection (update the port based on your system)
# ser = serial.Serial('/dev/cu.usbmodem14101', 9600)  # Update with your serial port

# # Function to load audio files from the correct directory
# def load_audio_files(song_name):
#     song_name_no_ext = os.path.splitext(song_name)[0]
#     base_folder = f"/Users/prafullsharma/Desktop/stem-glove/stems/{song_name_no_ext}-stems"

#     # Define the paths for the 3 stem audio files (omitting bass)
#     audio_files = [
#         os.path.join(base_folder, "drums_output.wav"),
#         os.path.join(base_folder, "other_output.wav"),
#         os.path.join(base_folder, "vocals_output.wav")
#     ]

#     return audio_files

# # Provide the song name (e.g., "impala.wav")
# song_name = "impala.wav"
# audio_files = load_audio_files(song_name)

# # Initialize pygame mixer for audio playback
# pygame.mixer.init()

# # Load the 3 audio files (omitting bass)
# channels = [pygame.mixer.Sound(file) for file in audio_files]

# # Initialize the audio channels and set them to play muted (volume = 0)
# for i in range(3):
#     channels[i].play(loops=-1)  # Play each file on loop
#     channels[i].set_volume(0)    # Start muted

# # Function to map FSR values (0-1023) to volume (0-1)
# def map_fsr_to_volume(fsr_value):
#     return min(fsr_value / 1023, 1.0)

# # Function to adjust volume dynamically without introducing lag
# def adjust_volume(fsr_value, channel):
#     """Adjust volume in real-time based on FSR value without lag."""
#     volume = map_fsr_to_volume(fsr_value)
#     channel.set_volume(volume)

# try:
#     while True:
#         if ser.in_waiting > 0:
#             # Read the data from Arduino and decode the byte string
#             data = ser.readline().decode('utf-8').strip()

#             # Check if the data is valid and not an empty string
#             if data:
#                 fsr_values = data.split(",")
                
#                 # Only proceed if we have exactly 4 FSR values (using only 3 for audio)
#                 if len(fsr_values) == 4:
#                     try:
#                         fsr_values = [int(x) for x in fsr_values]  # Convert to integers

#                         # Set volume based on FSR values for the first 3 sensors
#                         for i in range(3):
#                             adjust_volume(fsr_values[i], channels[i])  # Adjust volume
                    
#                     except ValueError:
#                         # Silently ignore any invalid data
#                         pass

#         time.sleep(0.01)  # Small delay to avoid overwhelming the serial buffer

# except KeyboardInterrupt:
#     print("Program stopped by user")

# finally:
#     ser.close()  # Close the serial connection when done
#     pygame.mixer.quit()  # Stop audio playback

