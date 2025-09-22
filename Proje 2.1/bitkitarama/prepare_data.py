import os
import pandas as pd
import shutil

# Dosya yollarını tanımlayın
DATA_DIR = 'plant-pathology'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
CSV_PATH = os.path.join(DATA_DIR, 'train.csv')

# Hastalık sınıflarını ve hedef klasörleri tanımlayın
# Bu betik, eğer yoksa klasörleri otomatik olarak oluşturacaktır.
classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
for cls in classes:
    class_dir = os.path.join(DATA_DIR, 'train', cls)
    os.makedirs(class_dir, exist_ok=True)

# train.csv dosyasını okuyun
df = pd.read_csv(CSV_PATH)

# Her resim için doğru klasöre taşıma işlemini yapın
for index, row in df.iterrows():
    img_name = row['image_id'] + '.jpg'
    src_path = os.path.join(IMAGES_DIR, img_name)
    
    # Hangi sınıf olduğunu bulun
    for cls in classes:
        if row[cls] == 1:
            dest_dir = os.path.join(DATA_DIR, 'train', cls)
            dest_path = os.path.join(dest_dir, img_name)
            
            # Resmi ilgili klasöre taşıyın
            shutil.move(src_path, dest_path)
            print(f'{img_name} -> {dest_dir} klasörüne taşındı.')
            break