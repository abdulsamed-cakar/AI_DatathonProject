import tensorflow as tf
from keras.models import load_model
import os
import pandas as pd
import numpy as np
from PIL import Image

# --- Ayarlar ---
IMG_WIDTH, IMG_HEIGHT = 224, 224

# --- Kaydedilmiş Modeli Yükle ---
# Eğitim aşamasında kaydettiğiniz modeli yükleyin
try:
    model = load_model('plant_pathology_model.h5')
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")
    exit()

# Modelin tahmin yapacağı sınıflar
class_names = ['healthy', 'multiple_diseases', 'rust', 'scab']

# --- Tahmin Yapılacak Veriyi Hazırla ---
TEST_DIR = os.path.join('plant-pathology', 'test')
SUBMISSION_CSV_PATH = os.path.join('plant-pathology', 'sample_submission.csv')

# Test resimlerinin isimlerini al
test_df = pd.read_csv(SUBMISSION_CSV_PATH)
test_image_ids = test_df['image_id'].values

# Tahmin sonuçlarını saklamak için boş bir DataFrame oluştur
predictions_df = pd.DataFrame(columns=['image_id'] + class_names)

# --- Her Bir Test Resmi İçin Tahmin Yap ---
print("Test resimleri üzerinde tahminler yapılıyor...")
for image_id in test_image_ids:
    image_path = os.path.join(TEST_DIR, image_id + '.jpg')
    
    # Resmi yükle, boyutlandır ve normalleştir
    img = Image.open(image_path)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0  # Normalleştirme
    img_array = np.expand_dims(img_array, axis=0)  # Tahmin için boyut ekle

    # Model ile tahmin yap
    predictions = model.predict(img_array, verbose=0)
    
    # En yüksek tahmin olasılığını bul ve ilgili sütuna 1 yaz
    result_row = {'image_id': image_id}
    for i, class_name in enumerate(class_names):
        result_row[class_name] = predictions[0][i]
    
    predictions_df.loc[len(predictions_df)] = result_row
    print(f"Tahmin tamamlandı: {image_id}")

# --- Tahmin Sonuçlarını Kaydet ---
# Tahmin sonuçlarını bir CSV dosyasına kaydet
output_csv_path = os.path.join('plant-pathology', 'predictions.csv')
predictions_df.to_csv(output_csv_path, index=False)

print("\nTahminler başarıyla tamamlandı ve 'predictions.csv' dosyasına kaydedildi.")
print("Tahmin Sonuçlarının ilk 5 satırı:\n")
print(predictions_df.head())