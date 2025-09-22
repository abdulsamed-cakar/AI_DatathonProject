import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import os
import json

# --- Klasör yollarını tanımla ---
DATA_DIR = 'plant-pathology'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')

# --- Eğitim parametreleri ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
NUM_EPOCHS = 5

# --- Veri Yükleme ve Hazırlama ---
# Eğitim verisini yükle
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE
)

# Doğrulama verisini ayır
validation_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE
)

# Sınıf isimlerini al
class_names = train_data.class_names
num_classes = len(class_names)
print("Sınıf İsimleri:", class_names)
print("Toplam Sınıf Sayısı:", num_classes)

# Veriyi 0-1 aralığında normalleştir
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
validation_data = validation_data.map(lambda x, y: (normalization_layer(x), y))

# --- Transfer Öğrenimi (Transfer Learning) Modeli Oluşturma ---
# MobileNetV2 modelini kullan
base_model = MobileNetV2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False,
    weights='imagenet'
)

# Önceden eğitilmiş katmanları dondur
base_model.trainable = False

# Modelin üzerine kendi katmanlarımızı ekle
global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(num_classes, activation='softmax')

# Modeli oluştur
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# --- Modeli Derleme (Compile) ---
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Modelin özetini göster
model.summary()

# --- Modeli Eğitme ---
print("\nModel eğitimi başlıyor...")
history = model.fit(
    train_data,
    epochs=NUM_EPOCHS,
    validation_data=validation_data
)

# --- Eğitilmiş Modeli Kaydetme ---
# Modelinizi daha sonra kullanabilmek için kaydedin
model.save('plant_pathology_model.h5')
print("Model başarıyla kaydedildi.")

# --- Eğitim geçmişini kaydetme ---
# Eğitim ve doğrulama sonuçlarını bir dosyaya kaydedebiliriz.
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)
print("Eğitim geçmişi kaydedildi.")