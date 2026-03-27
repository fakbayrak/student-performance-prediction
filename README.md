# 🎓 Öğrenci Başarısı Tahmini — Deep Learning ile CGPA Regresyonu ve Akademik Risk Tespiti

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Proje Hakkında

Bu proje, üniversite öğrencilerinin akademik başarısını makine öğrenmesi ve derin öğrenme yöntemleriyle tahmin etmeyi amaçlamaktadır. İki temel hedef doğrultusunda geliştirilmiştir:

- **CGPA Regresyonu:** Öğrencinin not ortalamasını (0–4 ölçeği) sayısal olarak tahmin etmek
- **Akademik Risk Tespiti:** CGPA < 2.5 olan öğrencileri erken dönemde tespit etmek (ikili sınıflandırma)

Proje; veri analizi, ön işleme, feature engineering, model karşılaştırması, hiperparametre optimizasyonu ve SHAP analizi adımlarından oluşan eksiksiz bir makine öğrenmesi pipeline'ı içermektedir.

---

## 📊 Veri Seti

| Özellik | Değer |
|---|---|
| Kaynak | [Kaggle — Students Performance Dataset](https://www.kaggle.com/datasets/grandmaster07/student-exam-performance-dataset-analysis) |
| Boyut | 1.194 öğrenci, 31 değişken |
| Hedef (Regresyon) | CGPA (0.0 – 4.0) |
| Hedef (Sınıflandırma) | Akademik Risk (CGPA < 2.5) |
| Risk Oranı | %4.5 (52 öğrenci) |
| Eksik Değer | 1 (skills sütunu) |

**Temel Değişkenler:**
- Önceki dönem not ortalaması (prev_sgpa)
- Tamamlanan kredi sayısı
- Devam oranı
- Günlük çalışma süresi
- Sosyal medya kullanım süresi
- İngilizce yeterlilik seviyesi
- Probation (sınıf geçememe) geçmişi
- Aile geliri, yaşam koşulları ve daha fazlası

---

## 🏗️ Proje Pipeline'ı

```
Ham Veri → EDA → Preprocessing → Feature Engineering → SMOTE
    → 5-Fold CV → Model Eğitimi → Threshold Optimizasyonu
    → SHAP Analizi → Sonuçlar
```

---

## ⚙️ Kullanılan Teknolojiler

```
Python 3.9+
TensorFlow / Keras  2.21     — Derin öğrenme modeli
XGBoost             2.1.3    — Gradient boosting
scikit-learn        1.x      — RF, GB, Logistic Regression, CV, SMOTE
pandas / numpy               — Veri işleme
matplotlib / seaborn         — Görselleştirme
```

---

## 🤖 Kullanılan Modeller

### Regresyon (CGPA Tahmini)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Ridge (Baseline) | 0.245 | 0.306 | 0.591 |
| Random Forest | 0.163 | 0.229 | 0.772 |
| Gradient Boosting | 0.161 | 0.227 | 0.785 |
| **XGBoost** | **0.156** | **0.220** | **0.800** ⭐ |
| Keras Deep Learning | 0.263 | 0.337 | 0.503 |

### Sınıflandırma (Akademik Risk — SMOTE + Threshold Opt.)
| Model | AUC | F1 | Recall | T* |
|---|---|---|---|---|
| Logistic Regression | 0.929 | 0.667 | 0.800 | 0.80 |
| Random Forest | 0.941 | 0.667 | 0.800 | 0.35 |
| **Gradient Boosting** | 0.909 | **0.700** | 0.700 | 0.30 |
| XGBoost | 0.932 | 0.640 | 0.800 | 0.50 |
| Keras Deep Learning | 0.947 | 0.560 | 0.700 | 0.70 |

---

## 🧠 Keras Model Mimarisi

```
Input(28) → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
          → Dense(64)  + BatchNorm + ReLU + Dropout(0.3)
          → Dense(32)  + BatchNorm + ReLU + Dropout(0.15)
          → Output(Linear / Sigmoid)

Optimizer : Adam (lr=0.001)
Loss (Reg): MSE
Loss (Clf): Binary Crossentropy
Callbacks : EarlyStopping(patience=20)
            ReduceLROnPlateau(factor=0.5, patience=8)
            ModelCheckpoint(save_best_only=True)
```

---

## 🔍 SHAP Analizi — Önemli Bulgular

**CGPA Regresyonu:**
- `prev_sgpa` → Açık ara en belirleyici faktör (importance: 1.62)
- `credits_completed` → İkinci sıra (importance: 0.48)
- Diğer tüm değişkenler çok düşük etki

**Akademik Risk Tespiti:**
- `probation` → En belirleyici (importance: 0.10)
- `prev_sgpa` → İkinci sıra (importance: 0.09)
- `social_media_hours` → Anlamlı negatif etki

> **Kritik Bulgu:** CGPA tahmini için geçmiş not belirleyiciyken, akademik riske düşmek için probation geçmişi daha kritik. İki farklı soru, iki farklı cevap.

---

## 📁 Dosya Yapısı

```
student-performance-prediction/
│
├── student_performance_full.py    # Tam Jupyter Notebook kodu (17 hücre)
├── Students_Performance_dataset.csv  # Veri seti
├── README.md                      # Bu dosya
│
├── outputs/                       # Grafik çıktıları
│   ├── 01_genel_bakis.png
│   ├── 02_kategorik_analiz.png
│   ├── 03_korelasyon.png
│   ├── 04_scatter_plots.png
│   ├── 05_preprocessing.png
│   ├── 06_split_smote.png
│   ├── 07_keras_reg_egitim.png
│   ├── 08_keras_clf_egitim.png
│   ├── 09_reg_karsilastirma.png
│   ├── 10_clf_karsilastirma.png
│   ├── 11_hyperparameter.png
│   ├── 12_shap_analizi.png
│   └── 13_final_ozet.png
│
└── models/                        # Eğitilmiş modeller
    ├── best_reg_model.h5
    ├── best_clf_model.h5
    ├── rf_reg.pkl
    ├── rf_clf.pkl
    └── scaler.pkl
```

---

## 🚀 Kurulum ve Çalıştırma

```bash
# 1. Repoyu klonla
git clone https://github.com/fakbayrak/student-performance-prediction.git
cd student-performance-prediction

# 2. Kütüphaneleri kur
pip install tensorflow xgboost scikit-learn pandas numpy matplotlib seaborn

# 3. Jupyter Notebook'ta çalıştır
jupyter notebook student_performance_full.py
```

> **Not:** Her `# %% HÜCRE X` satırı yeni bir Jupyter hücresini temsil eder.

---

## 📈 Temel Çıkarımlar

1. **XGBoost en iyi regresyon modelidir** — R²=0.80 ile CGPA'nın %80'ini açıklıyor
2. **Gradient Boosting en iyi sınıflandırma modelidir** — F1=0.700 ile en dengeli performans
3. **Küçük veri setlerinde tree-based modeller Keras'ı geçer** — 1.155 örnek derin öğrenme için az
4. **SMOTE + threshold optimizasyonu kritiktir** — F1 0.18'den 0.70'e yükseldi
5. **Geçmiş not ortalaması (prev_sgpa) her şeyin önüne geçiyor** — Tek başına modelin %80'ini oluşturuyor

---

## 👤 Geliştirici

**Faruk Akbayrak**
Bilgisayar Mühendisliği — Afyon Kocatepe Üniversitesi

[![GitHub](https://img.shields.io/badge/GitHub-fakbayrak-black?logo=github)](https://github.com/fakbayrak)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Faruk_Akbayrak-blue?logo=linkedin)](https://www.linkedin.com/in/faruk-akbayrak-a50475299)
[![Email](https://img.shields.io/badge/Email-farukakbayrak0@gmail.com-red?logo=gmail)](mailto:farukakbayrak0@gmail.com)

---

## 📄 Lisans

Bu proje MIT lisansı altında dağıtılmaktadır.
