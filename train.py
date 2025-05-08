

"""
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa

# YAMNet modelini indir
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Ses dosyasını yükle
def load_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000)  # YAMNet 16kHz bekler
    return waveform

# YAMNet'ten embedding al
def extract_embedding(waveform):
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return tf.reduce_mean(embeddings, axis=0)  # ortalama embedding (1024 boyutlu)

# Örnek: Dataset yükleme (sen kendi verini buraya entegre edebilirsin)
X = []  # feature vektörleri
y = []  # etiketler (0: Healthy, 1: Sick, 2: None)

# Kendi dosya listene göre dön
for path, label in your_audio_files_and_labels:
    audio = load_audio(path)
    emb = extract_embedding(audio)
    X.append(emb.numpy())
    y.append(label)

X = np.array(X)
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Model tanımı (embedding üzerine Dense katman)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')  # Burada 3 sınıf var
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Modeli eğit
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)


"""


# On scores

"""
def extract_yamnet_scores(waveform):
    # Tek bir dalgaformu için scores döner
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return scores  # (time_frames, 521)

# Keras modeli
input_waveform = tf.keras.Input(shape=(None,), dtype=tf.float32, name="audio")

# Her örnek için ayrı ayrı scores hesapla
yamnet_scores = tf.keras.layers.Lambda(
    lambda x: tf.map_fn(extract_yamnet_scores, x, dtype=tf.float32)
)(input_waveform)  # shape: (batch, time_frames, 521)

# Zaman üzerinden ortalama al
x = tf.keras.layers.GlobalAveragePooling1D()(yamnet_scores)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs=input_waveform, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
"""


# yamnet_layer
"""
yamnet_layer = hub.KerasLayer(
    "https://tfhub.dev/google/yamnet/1",
    trainable=True,            
    input_shape=(None,),       
    dtype=tf.float32,
    name="yamnet"
)
"""