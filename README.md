<h1>Bitki Uzmanı Web Projesi🪴</h1>
<p>   Bu proje, bir bitki hakkında aklınıza takılan her şeyi sorabileceğiniz bir yapay zeka asistanıdır. Karşılıklı soru-cevap şeklinde ilerleyen bir sohbet uygulamasıdır.
  Amacımız, bitki bakımı, bitki hastalıkları ve genel bitki sağlığı gibi konularda kullanıcılara hızlı ve güvenilir bir şekilde yardımcı olmaktır.</p>

<p>Projeyi yaparken kendi eğittiğimiz model üzerine yapay zeka  entegresi yaparak verilen cevapları güçlendirdik.</p>

<ul>
  <li>
    Görsel Analiz: Bir bitkinin fotoğrafını yüklediğinizde, sistem anında bitkinin türünü ve olası hastalıklarını teşhis eder.
 </li>
  <li>
    Detaylı Bilgi: Ardından, bu ilk teşhisi temel alan Gemini, size hastalığın belirtilerini, tedavi yöntemlerini ve bitkinizi nasıl daha sağlıklı hale getirebileceğinizle ilgili kapsamlı öneriler sunar.
  </li>
  <li>
    Sohbet Desteği: İster bir hastalık hakkında detaylı bilgi isteyin, ister bir bitkinin ne kadar suya ihtiyaç duyduğunu sorun, uygulama tüm sorularınıza sohbetin akışına uygun ve doğru cevaplar verir
  </li>
</ul>

<h3>Programlama Dili ve Kütüphaneleri 📚</h3>

<ul>
  <li>
    Python: Projenin backendi Python ile yazıldı. 
  </li>
  <li>
    Flask: Hafif ve esnek bir web çatısı. Kayıt, giriş ve sohbet gibi tüm web sayfalarını yönetmek için Flask'ı kullandım. Bu sayede, uygulamanın temel yapısını tek bir dosyada kurmak mümkün oldu.
  </li>
  <li>
    SQLite3: Kullanıcı verilerini (e-posta, şifre) ve tüm sohbet geçmişini (mesajlar, görseller, sağlık puanları) depolamak için kullanılan sunucusuz bir veritabanı. 
    Projenin tek dosya yapısına çok uygun çünkü ek bir veritabanı sunucusu kurulumu gerektirmiyor.
  </li>
  <li>
    TensorFlow & Keras: Kendi bitki hastalığı teşhis modelimi eğitmek ve yüklemek için bu popüler makine öğrenimi kütüphanelerini kullandım. 
    Modelin hızlı bir şekilde çalışmasını ve bitki fotoğraflarını analiz etmesini sağladılar.
  </li>
  <li>
    Google Generative AI (Gemini API): Google'ın gelişmiş yapay zeka modeli Gemini'ye erişim sağlayan bir kütüphane. 
    Kendi modelimin bulgularını alarak daha detaylı ve doğal dilde yanıtlar üretmesini sağladı.
  </li>
  <li>
    PIL (Pillow): Python'ın görsel işleme kütüphanesi. Kullanıcıların yüklediği görselleri işlemek için kullandım.
 </li>
</ul>

<h4>Projeye Ait Görseller 🖼️</h4>

<p>
   <img width="1919" height="909" alt="Ekran görüntüsü 2025-09-22 212426" src="https://github.com/user-attachments/assets/10322470-9055-4de5-98f0-660e43d55cd7" />
   <br>
  <br>
  Kullanıcı Girişi
  <br>
  <br>
  <img width="1919" height="915" alt="Ekran görüntüsü 2025-09-22 212443" src="https://github.com/user-attachments/assets/f7ba2dfd-64b3-4818-af8f-1994319fb89f" />
  <br>
  <br>
  Kayıt Olma
  <br>
  <br>
  <img width="1919" height="905" alt="Ekran görüntüsü 2025-09-22 212602" src="https://github.com/user-attachments/assets/dec69543-e401-4e9b-a002-ce0e5620cfac" />
  <br>
  <br>
  Web Arayüzü
  <br>
  <br>
  <img width="1917" height="907" alt="Ekran görüntüsü 2025-09-22 212701" src="https://github.com/user-attachments/assets/19beb23e-fb9d-46d9-a64b-211e611655b5" />
   <br>
  <br>
  Örnek Kullanım
  <br>
  <br>
  <img width="1912" height="905" alt="Ekran görüntüsü 2025-09-22 212857" src="https://github.com/user-attachments/assets/8b1fad28-79c1-4688-9e2d-8cbcaa13e8fb" />
  <br>
  <br>
  Örnek Kullanım
</p>


