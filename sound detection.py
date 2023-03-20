#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install librosa


# In[3]:


pip install numpy==1.23.5


# In[1]:


import librosa
import numpy as np


# In[66]:


# Load the audio file
y, sr = librosa.load('ptk_2.wav')


# In[69]:


y


# In[7]:


# Define the "Pa Ta Ka" sequence as a list of onsets in seconds
onsets = [0.0004, 0.0008, 1.2, 1.7, 2.2, 2.7]   # adjust these values to match the specific audio file

# Calculate the onset times of the audio signal
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')

# Convert the onset frames to times in seconds
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# Check if each "Pa Ta Ka" onset is present in the audio signal
count = 0
for i in range(len(onsets)):
    if np.any(np.isclose(onset_times, onsets[i], rtol=0, atol=0.1)):
        count += 1

# Display the count of "Pa Ta Ka" units
print("Number of 'Pa Ta Ka' units:", count)


# In[9]:


librosa.onset.onset_detect(y=y, sr=sr, units='time')


# In[10]:


import matplotlib.pyplot as plt

plt.figure()
librosa.display.waveshow(y, sr=sr)
plt.title('nutcracker waveform')
plt.show()


# In[11]:


tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)


# In[12]:


print(tempo)
print(beat_frames)


# In[14]:


librosa.onset.onset_strength(y=y, sr=sr)


# In[15]:


pip install pydub


# In[73]:


from pydub import AudioSegment
from pydub.silence import split_on_silence

# Load the audio file
audio_file = AudioSegment.from_wav('ptk_2.wav')

# Define the "Pa Ta Ka" phrase as a string
phrase = "pa ta ka"

# Split the audio into chunks on silence
chunks = split_on_silence(audio_file, min_silence_len=10, silence_thresh=-40)

# Count the number of "Pa Ta Ka" units in each chunk
count = 0
for chunk in chunks:
    print(str(chunk.raw_data))
    # Convert the chunk to lowercase and remove whitespace and punctuation
    #chunk_text = ''.join([c.lower() for c in chunk.raw_data if c.isalpha()])
    #chunk_text = chunk_text.replace(' ', '').replace('.', '').replace(',', '')
    #if phrase in chunk_text:
        #count += 1

# Display the count of "Pa Ta Ka" units
#print("Number of 'Pa Ta Ka' units:", count)


# In[65]:


len(chunks)


# In[55]:


phrase.encode('utf-16')


# In[60]:


def transcribe_audio(filename):
    
    import speech_recognition as sr

    recognizer = sr.Recognizer()


  # Import the audio file and convert to audio data
    audio_file = sr.AudioFile(filename)
    with audio_file as source:
        audio_data = recognizer.record(source)


  # Return the transcribed text
    return recognizer.recognize_google(audio_data)


# In[76]:


transcribe_audio('ptk_2.wav')


# In[59]:


pip install SpeechRecognition


# In[80]:


from pydub import AudioSegment
from pydub.silence import split_on_silence

# Load the audio file
audio_file = AudioSegment.from_wav('ptk_2.wav')

# Define the "Pa Ta Ka" phrase as a list of tuples of (start time, end time)
phrase = [("pa", 0, 30), ("ta", 30, 60), ("ka", 60, 90)]

# Split the audio into chunks on silence
chunks = split_on_silence(audio_file, min_silence_len=50, silence_thresh=-40)

# Count the number of "Pa Ta Ka" units in each chunk
count = 0
for chunk in chunks:
    # Loop over the phrase tuples and check if each one is present in the chunk
    found = True
    for p in phrase:
        start_time = p[1]
        end_time = p[2]
        phrase_audio = chunk[start_time:end_time]
        if not phrase_audio.dBFS > -40:
            found = False
            break
    if found:
        count += 1

# Display the count of "Pa Ta Ka" units
print("Number of 'Pa Ta Ka' units:", count)


# In[119]:


# Load the audio file
y,sr = librosa.load('ptk_2.wav')

# Define the "Pa Ta Ka" phrase as a list of amplitude thresholds
phrase = [0.2, -0.2, 0.2]

onset_env = librosa.onset.onset_strength(y=y, sr=sr,

                                         hop_length=512,

                                         aggregate=np.median)

# Find the peaks in the audio data
#peaks = librosa.util.peak_pick(onset_env, pre_max=10, post_max=10, pre_avg=10, post_avg=10, delta=0.5, wait=10)

# Count the number of "Pa Ta Ka" units in each peak
count = 0
for p in peaks:
    # Extract the audio data around the peak and normalize it
    peak_audio = audio_data[p:p+len(phrase)]
    peak_audio_norm = peak_audio / max(abs(peak_audio))

    # Check if the audio data matches the "Pa Ta Ka" phrase
    if all(abs(peak_audio_norm - phrase) < 0.1):
        count += 1

# Display the count of "Pa Ta Ka" units
print("Number of 'Pa Ta Ka' units:", count)


# In[95]:


peak_audio_norm


# In[112]:


onset_env = librosa.onset.onset_strength(y=y, sr=sr,

                                         hop_length=512,

                                         aggregate=np.median)

peaks = librosa.util.peak_pick(onset_env, pre_max=5, post_max=5, pre_avg=8, post_avg=5, delta=0.5, wait=10)

peaks


# In[113]:


times = librosa.times_like(onset_env, sr=sr, hop_length=512)

fig, ax = plt.subplots(nrows=2, sharex=True)

D = np.abs(librosa.stft(y))

librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),

                         y_axis='log', x_axis='time', ax=ax[1])

ax[0].plot(times, onset_env, alpha=0.8, label='Onset strength')

ax[0].vlines(times[peaks], 0,

             onset_env.max(), color='r', alpha=0.8,

             label='Selected peaks')

ax[0].legend(frameon=True, framealpha=0.8)

ax[0].label_outer()


# In[114]:


len(peaks)


# In[120]:


import speech_recognition as sr

# Load the audio file
filename = "ptk_2.wav"
r = sr.Recognizer()
with sr.AudioFile(filename) as source:
    audio_data = r.record(source)

# Set up a list of the target syllables
target_syllables = ["pa", "ta", "ka"]

# Use the speech recognition library to transcribe the audio
transcription = r.recognize_google(audio_data)

# Split the transcription into individual words
words = transcription.lower().split()

# Initialize a counter for each target syllable
counts = {syllable: 0 for syllable in target_syllables}

# Loop through each word in the transcription and count the target syllables
for i in range(len(words) - 2):
    if words[i:i+3] == target_syllables:
        counts["pa"] += 1

# Print the counts for each target syllable
for syllable, count in counts.items():
    print(f"{syllable}: {count}")


# In[ ]:




