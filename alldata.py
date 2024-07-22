import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import warnings
import pandas as pd
import numpy as np
import os
import sys
import librosa
import librosa.display
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import warnings
import joblib

# load models
# model = load_model(".venv/audio.h5", compile=False)
# tmodel = load_model("tmodel_all.h5")
# costants
# CAT6 = ['fear', 'angry','happy', 'sad',' disgust', 'surprise']
CAT6=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
CAT3 = ["positive", "negative"]

# page settings
st.set_page_config(layout="wide")

max_width = 1000
padding_top = 0
padding_right = "20%"
padding_left = "10%"
padding_bottom = 0
COLOR = "#1f1f2e"
BACKGROUND_COLOR = "#d1d1e0"
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )

@st.cache_data
def save_audio(file):
    audio_dir = "audio"  # Ensure directory exists
    if not os.path.exists(audio_dir):  # Create directory if it does not exist
        os.makedirs(audio_dir)  # Specific change: Ensure directory creation
    file_path = os.path.join(audio_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path  # Specific change: Return the file path

@st.cache_data
def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return (rgbImage, Xdb)

@st.cache_data
def get_mfccs(audio, limit):
  print("==========================audio",audio)
  print("==========================limit",limit)

  y, sr = librosa.load(audio, sr=44100)
  a = librosa.feature.mfcc(y, sr, n_mfcc = 40)# n_mfcc changed 162 to 40
  if a.shape[1] > limit:
    mfccs = a[:,:limit]
  elif a.shape[1] < limit:
    mfccs = np.zeros((a.shape[0], limit))
    mfccs[:, :a.shape[1]] = a
  #return mfccs
  return np.mean(mfccs.T, axis=0)  # Specific change: Return MFCC features

@st.cache_data
def get_title(predictions, categories=CAT6):
    title = f"Detected emotion: {categories[predictions.argmax()]} \
    - {predictions.max() * 100:.2f}%"
    return title

@st.cache_data
def plot_emotions(fig, data6, data3=None, title="Detected emotion",
                  categories6=CAT6, categories3=CAT3):
# CAT6 = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

    # color_dict = {"disgust":"grey",
  #               "positive":"green",
  #               "happy": "green",
  #               "surprise":"orange",
  #               "fear":"purple",
  #               "negative":"red",
  #               "angry":"red",
  #               "sad":"lightblue"}

  color_dict = {"fear":"grey",
                "positive": "green",
                "angry": "green",
                "happy":"orange",
                "sad":"purple",
                "negative":"red",
                "disgust":"red",
                "surprise":"lightblue"}
  # CAT6 = ['fear', 'angry', 'happy', 'sad', ' disgust', 'surprise']

  if data3 is None:
      pos = data6[3] + data6[5]
      neg = data6[0] + data6[1] + data6[2] + data6[4]
      data3 = np.array([pos,neg])

  ind = categories6[data6.argmax()]
  color6 = color_dict[ind]

  data6 = list(data6)
  n = len(data6)
  data6 += data6[:1]
  angles6 = [i/float(n)*2*np.pi for i in range(n)]
  angles6 += angles6[:1]

  ind = categories3[data3.argmax()]
  color3 = color_dict[ind]

  data3 = list(data3)
  n = len(data3)
  data3 += data3[:1]
  angles3 = [i/float(n)*2*np.pi for i in range(n)]
  angles3 += angles3[:1]

  # fig = plt.figure(figsize=(10, 4))
  fig.set_facecolor('#d1d1e0')
  ax = plt.subplot(122, polar="True")
  # ax.set_facecolor('#d1d1e0')
  plt.polar(angles6, data6, color=color6)
  plt.fill(angles6, data6, facecolor=color6, alpha=0.25)

  ax.spines['polar'].set_color('lightgrey')
  ax.set_theta_offset(np.pi / 3)
  ax.set_theta_direction(-1)
  plt.xticks(angles6[:-1], categories6)
  ax.set_rlabel_position(0)
  plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
  plt.ylim(0, 1)

  ax = plt.subplot(121, polar="True")
  # ax.set_facecolor('#d1d1e0')
  plt.polar(angles3, data3, color=color3, linewidth=2, linestyle="--", alpha=.8)
  plt.fill(angles3, data3, facecolor=color3, alpha=0.25)

  ax.spines['polar'].set_color('lightgrey')
  ax.set_theta_offset(np.pi / 6)
  ax.set_theta_direction(-1)
  plt.xticks(angles3[:-1], categories3)
  ax.set_rlabel_position(0)
  plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
  plt.ylim(0, 1)
  plt.suptitle(title)
  plt.subplots_adjust(top=0.75)


##########################################################################################################

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=0.8)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=0.7)

def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    return result


def get_features(data):
    # filename = librosa.ex(data)

    data, sample_rate = librosa.load(data, duration=2.5, offset=0.6)

    res1 = extract_features(data,sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data,sample_rate)
    result = np.vstack((result, res2))

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res3))

    return result

##########################################################################################################



def main():
    st.title(" Neu-Free emotion analysis for e-care buddy hearing product")
    st.sidebar.markdown("## Use the menu to navigate on the site")

    menu = ["Upload audio", "Dataset analysis", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Upload audio":

        st.subheader("Upload audio")
        audio_file = st.file_uploader("Upload audio file", type=['wav' , 'mp3'])

        if st.button('Record'):
            with st.spinner(f'Recording for 5 seconds ....'):
                st.write("Recording...")
                time.sleep(3)
            st.success("Recording completed")

        if audio_file is not None:
            st.title("Analyzing...")
            #file_details = {"Filename": audio_file.name, "FileSize": audio_file.size}
            file_details = save_audio(audio_file)
            #st.write(file_details)
            # st.subheader(f"File {file_details['Filename']}")
            print("========================audio_file:::",audio_file.name)
           # st.audio(audio_file, format='audio/wav', start_time=0)
            st.audio(file_details)


            ##############################################################################
            path = os.path.join("audio", audio_file.name)
            print("========================path:::", path)
            save_audio(audio_file)

            # extract features
            wav, sr = librosa.load(path, sr=44100)
            Xdb = get_melspec(file_details)[1]
            ##############################################################################
            fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
            fig.set_facecolor('#d1d1e0')

            plt.subplot(211)
            plt.title("Wave-form")
            # librosa.display.waveplot(wav, sr=sr)
            librosa.display.waveshow(wav, sr=sr)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.spines["right"].set_visible(False)
            plt.gca().axes.spines["left"].set_visible(False)
            plt.gca().axes.spines["top"].set_visible(False)
            plt.gca().axes.spines["bottom"].set_visible(False)
            plt.gca().axes.set_facecolor('#d1d1e0')

            plt.subplot(212)
            plt.title("Mel-log-spectrogram")
            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.spines["right"].set_visible(False)
            plt.gca().axes.spines["left"].set_visible(False)
            plt.gca().axes.spines["top"].set_visible(False)
            st.write(fig)

            data3 = np.array([.8, .9, .2])

            st.title("Getting the result...")
        
            model = load_model("/speech_audio.h5",compile=False)
                 ##############################################################################
            data3 = get_mfccs(file_path, limit=3)  # Specific change: Get MFCC features
            data3 = data3.reshape(1, *data3.shape)
            y_pred = model.predict(data3)
                ##############################################################################
            mfccs = get_mfccs(path, model.input_shape[-1])
            # desired_length = 162
            # padded_x = np.pad(mfccs, ((0, 0), (0, desired_length - mfccs.shape[1]), (0, 0)), mode='constant', constant_values=0)
            # print("==========================in",padded_x.shape)
            new_arr = np.interp(np.linspace(0, 1, 162), np.linspace(0, 1, 128), mfccs.flatten())
            new_arr = new_arr.reshape(162,1)
            new_arr = new_arr.reshape(1, *new_arr.shape)

            print("==========================in", new_arr.shape)
            mfccs = mfccs.reshape(1, *mfccs.shape)
            print("==========================in1::",mfccs.shape)
            # pred = model.predict(new_arr)[0]
            # print("===========================pred:::",pred)

            # ***********************************************
            data_fin, sample_rate_fin = librosa.load(
                "/main"+str(path))
            data,sample_rate=data_fin,sample_rate_fin
            X = []

            feature = get_features(
                "/main"+str(path))
            for ele in feature:
                X.append(ele)

            Features = pd.DataFrame(X)
            X1 = Features.values

            x_test = X1
            scaler = joblib.load('/scaler.pkl1')

            # Use the loaded scaler to transform data
            x_test = scaler.transform(x_test)
            x_test = np.expand_dims(x_test, axis=2)
            print('====================>1')
            path_checkpoint = "/training"
            model.load_weights(path_checkpoint)
            pred_test = model.predict(x_test)
            encoder_file = "/encoder1.npy"
            encoder_categories = np.load(encoder_file, allow_pickle=True)
            categories_list = encoder_categories.tolist()[0]
            print("=================categories_list::",categories_list)
            new_encoder = OneHotEncoder(categories=[categories_list])
            new_encoder.fit(np.array(categories_list).reshape(-1, 1))


            pred = new_encoder.inverse_transform(pred_test)
            print("======================pred:::",pred_test[0])

            # df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
            # df['Predicted Labels'] = y_pred[0].flatten()
            # df.head(20)
            # ***********************************************
            ##############################################################################
            # txt = get_title(pred)
            txt="Detected emotion:"+str(pred[0][0])+"- 100.00%"
            print("==========================out",txt)

            fig = plt.figure(figsize=(10, 4))
            plot_emotions(data6=pred_test[0], fig=fig, title=txt)
            st.write(fig)

            # mel = get_melspec(path)
            # mel = mel.reshape(1, *mel.shape)
            # tpred = model.predict(mel)[0]
            # txt = get_title(tpred)
            # fig = plt.figure(figsize=(10, 4))
            # plot_emotions(data3=data3, data6=tpred, fig=fig, title=txt)
            # st.write(fig)

    elif choice == "Dataset analysis":
        st.subheader("Dataset analysis")
        # with st.echo(code_location='below'):


    else:
        st.subheader("About")
        st.info("thiruvikkiramanp@gmail.com")


if __name__ == '__main__':
    main()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.

st.button("Re-run")
