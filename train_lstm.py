"""
Problem Tanimi:LSTM ile metin uretme:Verilen kelimelerden anlamli Turkce cumleler olusturma.

LSTM:Long-Short Term Memory

Veri Seti:chatgpt ile olusturulan 100 adet gunluk hayat cumlesi

Plan/Program

install libraries(pip)(terminalden), requirements.txt

import libraries
"""

# import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# egitim verisini chatgpt ile olusturma
data = [
    "Bugün hava top oynamak için çok güzel.",
    "Sabah kahvesini içmeden güne başlayamıyorum.",
    "Akşam yemeği için ne hazırlamayı düşünüyorsun?",
    "Yarın ofise biraz geç gideceğim.",
    "Hafta sonu arkadaşlarla sinemaya gitmeyi planlıyoruz.",
    "Marketten gelirken iki ekmek alır mısın?",
    "Otobüs yine çok kalabalıktı, ayakta geldim.",
    "Bu akşam en sevdiğim dizinin yeni bölümü var.",
    "Telefonumun şarjı bitmek üzere, şarj aleti nerede?",
    "Toplantı ne zaman başlayacak?",
    "Bugün işte çok yoruldum, hemen uyumak istiyorum.",
    "Çocukları okuldan almaya ben giderim.",
    "Hava durumu yarın yağmurlu gösteriyor, şemsiyeni unutma.",
    "Bu ara çok fazla tatlı yiyorum, diyete başlamam lazım.",
    "Yeni aldığım ayakkabı ayağımı vurdu.",
    "Bilgisayarın ekranı aniden karardı, ne olduğunu anlamadım.",
    "Sana anlattığım kitabı mutlaka okumalısın.",
    "Evdeki internet bağlantısı yine koptu.",
    "Çiçekleri sulamayı unutmuşum, kurumak üzereler.",
    "Gelecek yaz tatilinde nereye gitsek acaba?",
    "Mutfaktaki ampul patladı, yenisini takmak lazım.",
    "Trafik o kadar sıkışıktı ki eve zor vardım.",
    "Buzdolabında hiç süt kalmamış.",
    "Bugün kendimi çok enerjik hissediyorum.",
    "Sınav sonuçları açıklandığında bana haber ver.",
    "Kedinin mamasını değiştirdim, pek sevmedi gibi.",
    "Saçımı kestirmek için kuaförden randevu aldım.",
    "Kargom bugün teslim edilecekmiş.",
    "Akşam yürüyüşüne çıkalım mı?",
    "Ocağın altını kapattın mı, emin misin?",
    "Faturaları yatırmak için bankaya gitmem gerekiyor.",
    "Bu şarkıyı ne zaman duysam eski günleri hatırlarım.",
    "Kapı çalıyor, kimin geldiğine bir bakar mısın?",
    "Havalar soğudu, artık mont giymenin vakti geldi.",
    "Bugün öğle yemeğinde döner yedik.",
    "Bulaşık makinesini boşaltmak senin sıran.",
    "Anahtarlarımı nerede bıraktığımı hatırlayamıyorum.",
    "Arabayı bakıma götürmem lazım, çok ses yapıyor.",
    "Bugün canım hiçbir şey yapmak istemiyor.",
    "Alarm çaldı ama ben yine uyuya kalmışım.",
    "Maçın özetini izlemek için televizyonu açtım.",
    "Annemleri arayıp nasıl olduklarını sormam lazım.",
    "Çay demlendi, bardakları getiriyorum.",
    "Bu ay çok masrafımız oldu, biraz tasarruf etmeliyiz.",
    "Komşunun köpeği bütün gece havladı, hiç uyutmadı.",
    "Hastaneden tahlil sonuçlarını almaya gideceğim.",
    "Şu dosyayı bana e-posta ile gönderebilir misin?",
    "Hafta içi çok yoğunum, hafta sonu görüşelim.",
    "Dışarı çıkarken çöpü de atıver, lütfen.",
    "Yarın sabah erkenden yola çıkacağız, eşyaları hazırlayalım.",
    "Sabah uyanır uyanmaz ilk işim telefonu kontrol etmek oldu.",
    "Çay demlenene kadar kahvaltıyı hazırlayayım.",
    "Dışarı çıkarken üstüne kalın bir şeyler al, hava serin.",
    "Bugün toplantı o kadar uzadı ki canım sıkıldı.",
    "Yemek çok güzel olmuş, ellerine sağlık.",
    "Bana biraz borç verebilir misin, ay başında öderim.",
    "Sipariş ettiğim yemek hala gelmedi, açlıktan öleceğim.",
    "Akşamki maçı kimin kazanacağını düşünüyorsun?",
    "Dün gece çok geç yattığım için bugün gözümü açamıyorum.",
    "Marketten gelirken unutma, peçete de al.",
    "Arabanın benzini bitmek üzere, ilk benzinliğe girmeliyiz.",
    "Bu hafta sonu evi temizlemeye ayıracağım.",
    "Seninle konuşmayalı uzun zaman oldu, çok özledim.",
    "Şarj aletimi evde unutmuşum, seninkini kullanabilir miyim?",
    "Bu bilgisayar çok yavaşladı, format atmak lazım.",
    "Ayakkabının bağcığı çözülmüş, düşmeden bağla.",
    "Saçımı boyatmayı düşünüyorum ama renge karar veremedim.",
    "Otobüsü kaçırdım, bir sonraki yarım saat sonra gelecek.",
    "Evin anahtarını kapının üzerinde unutmuşum.",
    "Çocuğun ödevine yardım etmem gerekiyor.",
    "Tuzu uzatır mısın, çorba biraz tatsız olmuş.",
    "Bugün hava çok sıcak, klimalı bir yere geçelim.",
    "Şu camı biraz aç da içeri temiz hava girsin.",
    "Geçen gün gördüğümüz o kazak çok indirime girmiş.",
    "Gözlüklerimi nereye koyduğumu bir türlü bulamıyorum.",
    "Dişçiden randevu aldım ama gitmeye çok korkuyorum.",
    "Banyodaki musluk damlatıyor, tamirci çağırmak şart.",
    "Kedim sabah erkenden kalkıp yüzümü yalayarak beni uyandırdı.",
    "Pazar kahvaltısı için güzel bir menemen yapalım.",
    "Arkadaşın doğum günü için ne hediye alsak?",
    "Metroya binerken kartımda bakiye kalmadığını fark ettim.",
    "Patron bugün yine çok sinirli, yanına yaklaşma.",
    "Bu akşam yemeği dışarıda yiyelim, hiç yemek yapasım yok.",
    "Yeni komşularla henüz tanışma fırsatım olmadı.",
    "Saat kaç oldu, hala ortalarda yoksun.",
    "Gömleğimi ütülemem lazım ama hiç vaktim kalmadı.",
    "O kadar güldüm ki karnıma ağrılar girdi.",
    "Tatlı olarak dondurma mı istersin yoksa baklava mı?",
    "Cüzdanımı evde unutmuşum, hesabı sen öder misin?",
    "Televizyonun kumandası koltuğun arasına sıkışmış.",
    "Çocukken en sevdiğim çizgi film buydu.",
    "Kredi kartı ekstresi gelmiş, görünce kalp krizi geçirecektim.",
    "Çok susadım, bana bir bardak soğuk su verir misin?",
    "Kış gelmeden kombinin bakımını yaptırmamız lazım.",
    "Yeni aldığım tişörtü ilk yıkamada çektirdim.",
    "Akşam yatmadan önce kitap okumak beni çok rahatlatıyor.",
    "Bu filmi daha önce izlemiştim ama sonunu hatırlamıyorum.",
    "Ayakkabılarım çamur oldu, kapıda çıkarmam lazım.",
    "Gözlerim çok ağrıyor, biraz ekrandan uzaklaşmalıyım.",
    "Hafta sonu hava güzel olursa deniz kenarına gidelim."
]
# --Preprocessing--
# kelimeleri indexlere(sayilara) cevirme(tokenizer)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1 #padding için +1

print(f"total_words: {total_words}")

# n-gram dizileri olustur yani her cumleden kisa diziler olustur(embedding)
input_sequences = []
for text in data:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        n_gram_sequences = token_list[: i+1]
        input_sequences.append(n_gram_sequences)

print(f"input_sequenes: \n{input_sequences}")
"""
[[2, 7], [2, 7, 73], [2, 7, 73, 74], [2, 7, 73, 74, 3], [2, 7, 73, 74, 3, 1], [2, 7, 73, 74, 3, 1, 15]
"Bugün(2) hava(7) top(73) oynamak(74) için(3) çok(1) güzel(15)."
"""

# Padding:Farkli uzunluktaki dizileri sabitleme
max_sequence_length = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen = max_sequence_length, padding = "pre")

print(f"after padding input_sequenes: \n{input_sequences}")

# girdi(x) ve hedef degiskenler(y) ayirma
X = input_sequences[:, :-1] # n-1 kelimeyi giriş olarak seçiyoruz
y = input_sequences[:, -1] #n inci kelimeyi tahmin et

# hedef degiskene one hot encoding yapma
y =tf.keras.utils.to_categorical(y, num_classes = total_words)
print(f"hedef degisken: {y}")
"""
one hot encoding mantigi:
[1,2,3] ->
1 -> [1,0,0]
2 -> [0,1,0]
3 -> [0,0,1]
"""

#--LSTM Training--
# lstm modeli tanimla
model = Sequential()
model.add(Embedding(total_words, 50, input_length = X.shape[1])) #embedding katmanı
model.add(LSTM(100))
model.add(Dense(total_words, activation = "softmax"))

# compile
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
print(model.summary())

# egitimi baslat
model.fit(X, y, epochs = 100, verbose = 1)

# ornek uretim testi(metin uretimi)
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen = max_sequence_length - 1, padding = "pre")
        predicted_probs = model.predict(token_list, verbose = 0)
        predicted_index = np.argmax(predicted_probs, axis = -1)[0]
        predicted_word = tokenizer.index_word[predicted_index]
        seed_text = seed_text + " " + predicted_word
    return seed_text

print(generate_text("Bugün", 5))