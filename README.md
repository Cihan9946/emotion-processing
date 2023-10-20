# emotion-processing
https://www.kaggle.com/mustafacihanncr/emotion-processing

Metin Sınıflandırma Modeli ve Metin İşleme
Metin sınıflandırma, doğal dil işleme (NLP) alanının önemli bir parçasıdır ve birçok uygulama için kullanılır. Bu örnekte, metin sınıflandırma görevi üzerinde çalışmak için kullanılan bir Python kodu incelenecek. Kod, bir metin veri kümesi üzerinde hisse senedi yorumlarını pozitif veya negatif olarak sınıflandıran bir derin öğrenme modeli oluşturur ve eğitir.

Kullanılan Kütüphaneler
Bu kod, metin işleme ve derin öğrenme için aşağıdaki Python kütüphanelerini kullanır:

pandas: Veri çerçevesi işleme için kullanılır.
numpy: Sayısal hesaplamalar için kullanılır.
tensorflow.keras: Derin öğrenme modeli oluşturmak ve eğitmek için kullanılır.
sklearn.model_selection: Veriyi eğitim ve test kümelerine bölmek için kullanılır.
Ayrıca, kod, zaman ölçümü ve uyarıları yönetmek için Python'ın time ve warnings modüllerini de kullanır.

Veri Kümesi
Kod, metin sınıflandırma modelini eğitmek için bir veri kümesi kullanır. Bu örnekte kullanılan veri kümesi 'hepsiburada.csv' adlı bir CSV dosyasından yüklenir. Veri kümesi, metin yorumlarını ve bu yorumların pozitif veya negatif olarak etiketlerini içerir. Bu, denetimli bir öğrenme görevi olan metin sınıflandırma için temel veridir.

Metin İşleme Adımları
Kod, aşağıdaki temel metin işleme adımlarını içerir:

Veri Setinin Yüklenmesi: Veri kümesi, pandas kütüphanesi kullanılarak bir veri çerçevesine yüklenir. Yorumlar (X) ve etiketler (Y) ayrıştırılır.

Veri Setinin Bölünmesi: Veri kümesi, eğitim ve test kümelerine ayrılır. Bu, modelin eğitim sırasında doğruluk değerini değerlendirmesini sağlar.

Tokenleştirme ve Sözlük Oluşturma: Metin verileri tokenleştirilir, yani kelimeler integer değerlere dönüştürülür. Tokenleştirme işlemi, en sık kullanılan kelimeleri temsil eden bir sözlük oluşturur. Bu sözlük, her kelimenin bir integer'e eşlendiği bir yapıdır.

Padding İşlemi: Metin verileri, maksimum kelime uzunluğuna (max_tokens) sahip olacak şekilde doldurulur veya kırpılır. Bu, modele veri girişi yapılırken tüm verilerin aynı uzunlukta olmasını sağlar.

Model Yapısı
Bu kod, bir derin öğrenme modeli oluşturur. İşte bu modelin temel yapısı:

Giriş Katmanı (Embedding): İlk katman, kelime vektörlerini içerir. Bu katman, tokenleştirilmiş metin verilerini kelime vektörlerine dönüştürür. Bu vektörler, metinlerdeki kelimelerin anlamsal benzerliği hakkında bilgi taşır.

Dört Katmanlı GRU (Gated Recurrent Unit): Model, dört GRU katmanı içerir. Bu katmanlar ardışık olarak metin verilerini işler ve öğrenilen özellikleri çıkarır. Her GRU katmanının birçok hücresi vardır ve dropout uygulanır, bu sayede aşırı uyum (overfitting) önlenir.

Çıkış Katmanı (Dense): Modelin çıkış katmanı, metin yorumlarını pozitif veya negatif olarak sınıflandırır. Bu katmanda sigmoid aktivasyon fonksiyonu kullanılır ve sonuç bir olasılık değeri olarak verilir.

Eğitim: Model, eğitim verileri kullanılarak eğitilir. Eğitim, verilerin model üzerinde iterasyonlarla işlenerek modelin parametrelerinin güncellenmesini içerir. Optimizasyon algoritması olarak Adam kullanılır.

Model Değerlendirmesi
Eğitim sonrası, model test verileri üzerinde değerlendirilir. Bu değerlendirme, modelin doğruluk (accuracy) değerini ve sınıflandırma sonuçlarını içerir. Ayrıca, yanlış sınıflandırılan örnekler de belirlenir ve incelenir.

Metin Verisi ile Modeli Deneme
Son olarak, kod metin verisi üzerinde modelin performansını test etmek için kullanılır. Belirli metin örnekleri, model tarafından pozitif veya negatif olarak sınıflandırılır. Bu, modelin gerçek dünya uygulamalarında nasıl performans gösterebileceğini anlamak için önemlidir.


Vektörleştirme (Embedding) İşlemi
Bu kod, metin verilerini işlemek ve derin öğrenme modeline veri sağlamak için bir vektörleştirme işlemi kullanır. Vektörleştirme, metin verilerini sayısal bir biçimde ifade etmek için kullanılır ve bu sayede modelin metin verileri anlamasına yardımcı olur. İşte vektörleştirme ile ilgili ayrıntılar:

Num_Words Parametresi: num_words değişkeni, metin verilerinde kullanılacak kelime sayısını sınırlayan bir parametreyi temsil eder. Bu, vektörleştirme işlemi sırasında dikkate alınacak en çok kullanılan kelimelerin sayısını belirler. Bu kod örneğinde num_words = 10000 olarak belirlenmiştir, yani en çok 10,000 kelime vektörleştirme işleminde kullanılır. Bu, hem hesaplama maliyetini azaltır hem de modelin en önemli kelimelere odaklanmasını sağlar.

Tokenizer ve Sözlük Oluşturma: Tokenizer sınıfı, metin verilerini vektörlere dönüştürmek için kullanılır. Veri seti üzerinde gezilir ve her kelimeye bir benzersiz token (integer) atanır. Bu tokenler, bir sözlük oluşturur, bu sözlükte her kelimenin token karşılığı bulunur. En çok kullanılan 10,000 kelime, bu sözlüğe eklenir ve tokenler bu kelimeler arasından seçilir.

Embedding Katmanı: İlk katman olarak kullanılan Embedding katmanı, metin verilerini vektörlere dönüştürür. Bu katman, her kelimenin token karşılığını alır ve bu tokenden bir vektör üretir. Örneğin, "merhaba" kelimesi için bir token belirlenir ve bu token, Embedding katmanı tarafından bir vektöre dönüştürülür. Bu vektör, "merhaba" kelimesinin anlamsal temsilini içerir. Bu işlem, metin verilerini sayısal bir biçimde ifade etmek için önemlidir, çünkü derin öğrenme modeli sayılarla çalışır.

Bu vektörleştirme işlemi sayesinde metin verileri modelin anlayabileceği bir formata dönüştürülür ve model, bu vektörler üzerinde işlem yaparak metin sınıflandırma görevini gerçekleştirir. num_words parametresi ile en çok kullanılan kelime sayısı sınırlanarak işlem maliyeti kontrol edilir ve modelin veriyi daha iyi temsil etmesi sağlanır.
