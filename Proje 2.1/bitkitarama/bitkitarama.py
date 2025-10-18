import os
import sqlite3
import datetime
import random
import threading
import queue
import traceback
from flask import (
    Flask, request, redirect, url_for, session,
    render_template, send_from_directory, jsonify
)
import json
import base64
from PIL import Image
import io

# Kendi Eğittiğiniz Modeli İçe Aktarma
import numpy as np
import tensorflow as tf
from keras.models import load_model


# plant-pathology klasörünü yolu 
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'plant-pathology', 'plant_pathology_model.h5')

# Modeli belleğe yükle
try:
    plant_model = load_model(MODEL_PATH)
    print("Kendi bitki hastalığı modeliniz başarıyla yüklendi!")
except Exception as e:
    print(f"[HATA] Bitki hastalığı modeli yüklenemedi: {e}")
    plant_model = None

# ---------- Gemini API ----------
import google.generativeai as genai

gemini_api_key = "Kendi API nizi yapıştırın."

try:
    if not gemini_api_key or gemini_api_key == "GEMINI_API_KEY":
        print("[HATA] Lütfen 'bitkitarama.py' dosyasındaki API anahtarınızı girin!")
        raise KeyError
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    print(f"[HATA] Gemini API yapılandırma hatası: {e}")
    gemini_api_key = None

GEMINI_AVAILABLE = gemini_api_key is not None

# ---------- Config ----------
DB_FILE = "bitkitarama.db"
STATIC_DIR = "static"
UPLOAD_FOLDER = "uploads"

# Flask uygulamasını, bulunduğu dizine göre başlatır
app = Flask(__name__)
app.secret_key = "bitkitarama_secret_v2"

app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, STATIC_DIR, UPLOAD_FOLDER)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.root_path, "templates"), exist_ok=True)

# ---------- Database init ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        name TEXT,
        password TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS chats (
        chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        chat_title TEXT,
        last_updated TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        sender TEXT,
        message TEXT,
        image_path TEXT,
        timestamp TEXT,
        health_score INTEGER
    )""")
    conn.commit()
    conn.close()

init_db()

# ---------- DB helpers ----------
def save_user(email, name, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO users VALUES (?,?,?)", (email, name, password))
    conn.commit()
    conn.close()

def get_user(email):
    if not email:
        return None
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()
    return {"email": row[0], "name": row[1], "password": row[2]} if row else None

def create_new_chat(email, chat_title="Yeni Sohbet"):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    ts = datetime.datetime.now().isoformat()
    c.execute("INSERT INTO chats (email, chat_title, last_updated) VALUES (?,?,?)", (email, chat_title, ts))
    chat_id = c.lastrowid
    conn.commit()
    conn.close()
    return chat_id

def update_chat_title(chat_id, new_title):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE chats SET chat_title = ? WHERE chat_id = ?", (new_title, chat_id))
    conn.commit()
    conn.close()

def delete_chat_from_db(chat_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    c.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()

def get_user_chats(email):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT chat_id, chat_title, last_updated FROM chats WHERE email = ? ORDER BY last_updated DESC", (email,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "last_updated": r[2]} for r in rows]

def save_message(chat_id, sender, message, image_path=None, health_score=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    ts = datetime.datetime.now().isoformat()
    # image_path 
    c.execute("INSERT INTO messages (chat_id, sender, message, image_path, timestamp, health_score) VALUES (?,?,?,?,?,?)", (chat_id, sender, message, os.path.basename(image_path) if image_path else None, ts, health_score))
    c.execute("UPDATE chats SET last_updated = ? WHERE chat_id = ?", (ts, chat_id))
    conn.commit()
    conn.close()

def get_messages(chat_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT sender, message, image_path, health_score FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,))
    rows = c.fetchall()
    conn.close()
    return [{"sender": r[0], "message": r[1], "image_path": r[2], "health_score": r[3]} for r in rows]

def get_chat_health_scores(chat_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT health_score FROM messages WHERE chat_id = ? AND health_score IS NOT NULL ORDER BY timestamp ASC", (chat_id,))
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]
    
def analyze_plant_health(response_text):
    text = (response_text or "").lower()
    if any(w in text for w in ["sağlıklı", "canlı", "iyi durumda", "hastalık", "zarar", "kuruma", "sararma", "solgun", "iyileşiyor", "gelişiyor"]):
        if any(w in text for w in ["sağlıklı", "canlı", "iyi durumda"]):
            return random.randint(75, 100)
        elif any(w in text for w in ["iyileşiyor", "gelişiyor"]):
            return random.randint(40, 75)
        elif any(w in text for w in ["hastalık", "zarar", "kuruma", "sararma", "solgun"]):
            return random.randint(0, 40)
    return None

def get_past_plant_data(chat_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT message FROM messages WHERE chat_id = ? AND sender = 'ai'", (chat_id,))
    rows = c.fetchall()
    conn.close()
    
    past_plants = []
    for row in rows:
        msg = row[0].lower()
        if "bitki türü" in msg or "bitki adı" in msg:
            pass
    return past_plants

# Kendi modelinizle tahmin yapan yeni fonksiyon
def predict_plant_disease(image_path):
    global plant_model
    if plant_model is None:
        return "Model yüklenemedi. Tahmin yapılamıyor.", None

    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = plant_model.predict(img_array)
        score_healthy = predictions[0][0]
        score_multiple = predictions[0][1]
        score_rust = predictions[0][2]
        score_scab = predictions[0][3]
        
        class_names = ['healthy', 'multiple_diseases', 'rust', 'scab']
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = class_names[predicted_class_index]
        confidence = np.max(predictions)

        disease_map = {
            'healthy': "Sağlıklı",
            'multiple_diseases': "Birden Fazla Hastalık",
            'rust': "Pas Hastalığı",
            'scab': "Kabuk Hastalığı"
        }
        
        if predicted_class == 'healthy':
            health_score = int(confidence * 100)
        else:
            health_score = int((1 - confidence) * 100)
        
        response_text = f"""Görselinizdeki bitkiyi analiz ettim. Bitkinin **{disease_map.get(predicted_class, predicted_class)}** olduğunu düşünüyorum. Bu tahmine olan güvenim %{confidence * 100:.2f}.
        
**Öneriler:**
- Bu hastalığın belirtileri ve tedavi yöntemleri hakkında bilgi edin.
- Gerekirse ilgili ürünleri araştır ve uygula.
"""

        return response_text, health_score

    except Exception as e:
        print(f"[HATA] Model tahmini yapılamadı: {e}")
        traceback.print_exc()
        return "Üzgünüm, görsel analizi sırasında bir hata oluştu.", None

def generate_gemini_response(chat_history, user_input, image=None):
    if not GEMINI_AVAILABLE:
        return "Üzgünüm, Gemini API anahtarı ayarlanmamış. Lütfen 'bitkitarama.py' dosyasındaki API anahtarınızı girin."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        conversation = []
        for msg in chat_history:
            if msg['sender'] == 'user':
                conversation.append({'role': 'user', 'parts': [msg['message']]})
            elif msg['sender'] == 'ai':
                conversation.append({'role': 'model', 'parts': [msg['message']]})

        prompt_parts = []
        
        system_instruction = """Sen bir bitki uzmanısın. Kullanıcıya bitki bakımı, sağlığı ve türleri hakkında yardımcı oluyorsun.
1. Eğer bir bitki fotoğrafı yüklendiyse, önce bitkinin türü ve genel özellikleri hakkında bilgi ver.
2. Ardından, bitkinin sağlık durumunu analiz et ve varsa hastalık veya sorunları teşhis et. Bu analizi, cevabının içinde net bir şekilde belirt.
3. Son olarak, tespit ettiğin sorunlar için nasıl bir tedavi uygulanabileceğini, hangi yöntemlerin ve ürünlerin kullanılabileceğini detaylı bir şekilde açıkla.
4. Sohbet geçmişine dayanarak, kullanıcının bitki zevkini analiz et ve sevebileceği 1-2 farklı bitki türü öner. Bu öneriyi, cevabının en sonunda, yeni bir başlık altında yap.
5. Cevapların tamamen Türkçe olsun.
"""
        prompt_parts.append(system_instruction)

        if image:
            prompt_parts.append(image)
        
        prompt_parts.append(f"Kullanıcı: {user_input}")
        
        response = model.generate_content(prompt_parts)
        
        return response.text.strip()
    except Exception as e:
        traceback.print_exc()
        return f"[HATA] Gemini API hatası: {e}"
        
# ---------- Routes ----------
@app.route("/")
def home():
    if "user" in session and get_user(session.get("user")):
        return redirect(url_for("chat"))
    return redirect(url_for("login"))

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    msg = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        u = get_user(email)
        if u and u["password"] == password:
            session["user"] = email
            return redirect(url_for("welcome"))
        msg = "<div class='small' style='color:#ff7b7b'>Hatalı e-posta veya şifre</div>"
    return render_template("login.html", msg=msg)

@app.route("/register", methods=["GET", "POST"])
def register():
    msg = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        name = request.form.get("name", "").strip()
        password = request.form.get("password", "")
        if email and name and password:
            save_user(email, name, password)
            session["user"] = email
            return redirect(url_for("welcome"))
        msg = "<div class='small' style='color:#ff7b7b'>Lütfen tüm alanları doldurun</div>"
    return render_template("register.html", msg=msg)

@app.route("/welcome")
def welcome():
    email = session.get("user")
    user = get_user(email)
    if not user:
        return redirect(url_for("login"))
    
    new_chat_id = create_new_chat(email, chat_title="Yeni Bitki Sohbeti")
    session["current_chat_id"] = new_chat_id

    return render_template("welcome.html")

@app.route("/new_chat")
def new_chat():
    email = session.get("user")
    if not get_user(email):
        return redirect(url_for("login"))
    new_chat_id = create_new_chat(email, chat_title="Yeni Bitki Sohbeti")
    session["current_chat_id"] = new_chat_id
    return redirect(url_for("chat"))

@app.route("/select_chat/<int:chat_id>")
def select_chat(chat_id):
    email = session.get("user")
    if not get_user(email):
        return redirect(url_for("login"))
    session["current_chat_id"] = chat_id
    return redirect(url_for("chat"))

# Sohbeti silme
@app.route("/delete_chat/<int:chat_id>")
def delete_chat(chat_id):
    email = session.get("user")
    if not get_user(email):
        return jsonify({"error": "Unauthorized"}), 401

    delete_chat_from_db(chat_id)

    all_chats = get_user_chats(email)
    if all_chats:
        session["current_chat_id"] = all_chats[0]['id']
    else:
        new_chat_id = create_new_chat(email, chat_title="Yeni Bitki Sohbeti")
        session["current_chat_id"] = new_chat_id

    return redirect(url_for("chat"))

# Sohbet başlığını değiştirme
@app.route("/rename_chat/<int:chat_id>", methods=["GET"])
def rename_chat(chat_id):
    email = session.get("user")
    if not get_user(email):
        return jsonify({"error": "Unauthorized"}), 401
    
    new_title = request.args.get("new_title")
    if new_title:
        update_chat_title(chat_id, new_title)
    
    return redirect(url_for("chat"))

@app.route("/chat", methods=["GET", "POST"])
def chat():
    email = session.get("user")
    user = get_user(email)
    if not user:
        return redirect(url_for("login"))
    
    chat_id = session.get("current_chat_id")
    if not chat_id:
        return redirect(url_for("new_chat"))

    if request.method == "POST":
        message = request.form.get("message", "").strip()
        image_data_uri = request.form.get("imageDataUri")
        
        if not message and not image_data_uri:
            return redirect(url_for("chat"))

        image_path = None
        user_message = message
        ai_response = ""
        health_score = None

        if image_data_uri:
            try:
                header, encoded = image_data_uri.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                image_name = f"image_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}.png"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                 
                # predict_plant_disease, bitkiyi analiz eder ve bir metin yanıtı ile sağlık puanı döner.
                model_response, health_score = predict_plant_disease(image_path)
              
                prompt_for_gemini = f"""
                Kullanıcının yüklediği bitki görselini kendi modelimle analiz ettim. Analiz sonucum şudur:
                
                {model_response}
                
                Bu analizi temel alarak, bitkinin genel durumu hakkında detaylı bilgi ver ve varsa tespit ettiğim sorun için bakım önerilerinde bulun. Bu cevabı samimi bir dille, sohbetin devamı gibi yaz. Kullanıcının sorduğu ek bir soru varsa ("{message}"), onu da yanıtına dahil et.
                """

                # ADIM 3: Gemini'ye, oluşturduğunuz bu yeni istemle birlikte görseli ve sohbet geçmişini gönderin.
                # Gemini, hem kendi görsel anlama yeteneğini hem de verdiğiniz bilgiyi kullanacaktır.
                chat_history = get_messages(chat_id)
                ai_response = generate_gemini_response(chat_history, prompt_for_gemini, image=None)
                
            except Exception as e:
                print(f"Görsel işleme hatası: {e}")
                traceback.print_exc()
                ai_response = "Görsel analizi sırasında bir hata oluştu."
                health_score = None
        
        else:
            # Görsel yoksa, sadece metin tabanlı bir yanıt oluşturma
            chat_history = get_messages(chat_id)
            ai_response = generate_gemini_response(chat_history, message)
            health_score = analyze_plant_health(ai_response)

        # Mesajları veritabanına kaydetme
        save_message(chat_id, "user", user_message, image_path)
        save_message(chat_id, "ai", ai_response, health_score=health_score)

        return redirect(url_for("chat"))

    chat_history = get_messages(chat_id)
    all_chats = get_user_chats(email)
    
    last_health_score = None
    for msg in reversed(chat_history):
        if msg["sender"] == "ai" and msg["health_score"] is not None:
            last_health_score = msg["health_score"]
            break

    health_class = "blue"
    if last_health_score is not None:
        if 0 <= last_health_score <= 30:
            health_class = "bordo"
        elif 30 < last_health_score <= 50:
            health_class = "orange"
        elif 50 < last_health_score <= 70:
            health_class = "yellow"
        elif 70 < last_health_score <= 100:
            health_class = "green"
    
    messages_html = ""
    for msg in chat_history:
        if msg["sender"] == "user":
            messages_html += f"""
            <div class="msg-user">
                {f'<p>{msg["message"]}</p>' if msg["message"] else ''}
                {f'<img src="{url_for("static", filename="uploads/" + msg["image_path"])}" class="chat-image">' if msg["image_path"] else ''}
            </div>
            """
        elif msg["sender"] == "ai":
            messages_html += f"""
            <div class="msg-ai">
                {msg["message"]}
            </div>
            """

    chats_html = ""
    for chat_item in all_chats:
        is_active = "btn-active" if chat_item["id"] == chat_id else "secondary"
        chats_html += f"""
        <div style="display:flex;align-items:center;margin-bottom:8px;">
            <a class="btn {is_active}" href="{url_for('select_chat', chat_id=chat_item['id'])}" style="flex:1;">
                {chat_item["title"]}
            </a>
            <span class="edit-icon" onclick="renameChat({chat_item['id']})">✏️</span>
            <span class="delete-icon" onclick="deleteChat({chat_item['id']})">🗑️</span>
        </div>
        """
        
    return render_template(
        "chat.html",
        user_name=user["name"],
        user_email=user["email"],
        chats_html=chats_html,
        messages_html=messages_html,
        health_class=health_class
    )

if __name__ == "__main__":
    app.run(debug=True)
