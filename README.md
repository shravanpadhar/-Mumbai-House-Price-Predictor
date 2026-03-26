# 🏙️ Mumbai House Price Predictor

A machine learning web application that predicts residential property prices across Mumbai using a Random Forest model trained on 70,000+ real listings. Built with Python, Scikit-learn, and Streamlit.

---

## 📸 Demo

> Launch the app and get instant price predictions by selecting your locality, property type, BHK, and area.

![Dashboard Preview](assets/dashboard_preview.png)

---

## 🚀 Features

- 🔮 **Real-time Price Prediction** — Instant estimates based on locality, BHK, area, and property type
- 📍 **388 Mumbai Localities** — Covers Andheri, Thane, Borivali, Kharghar, Mira Road, and more
- 📊 **Confidence Band** — Every prediction includes a ±12% low/high price range
- 💡 **Sensitivity Analysis** — See how price changes as area increases, within the same locality
- 🧠 **ML Pipeline** — End-to-end sklearn Pipeline with preprocessing + Random Forest
- 💾 **Pickle Deployment** — Trained model saved as `model.pkl` for zero-retraining reuse

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Model | Scikit-learn (Random Forest Regressor) |
| Dashboard | Streamlit |
| Data | Pandas, NumPy |
| Serialization | Pickle |

---

## 📁 Project Structure

```
mumbai-house-price-predictor/
│
├── train_model.py                        # ML pipeline — cleans data, trains RF, saves model
├── dashboard.py                          # Streamlit dashboard
├── mumbai-house-price-data-cleaned.csv   # Dataset (70,000+ listings)
├── model.pkl                             # Trained sklearn Pipeline (auto-generated)
├── meta.pkl                              # Locality list & slider ranges (auto-generated)
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/mumbai-house-price-predictor.git
cd mumbai-house-price-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```
This will generate `model.pkl` and `meta.pkl` in the same directory.

### 4. Launch the dashboard
```bash
streamlit run dashboard.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
streamlit
```

Or install directly:
```bash
pip install pandas numpy scikit-learn streamlit
```

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Training Samples | ~56,000 |
| Test Samples | ~14,000 |
| R² Score | **0.8145 (81.5%)** |
| MAE | ₹38.4 Lakhs |
| RMSE | ₹76.1 Lakhs |
| Target Encoding | log1p transform (reversed on output) |

### Feature Importances

| Feature | Importance |
|---|---|
| BHK (bedroom_num) | 40.0% |
| Locality | 29.9% |
| Area (sq ft) | 29.0% |
| Property Type | 1.1% |

### Preprocessing Pipeline
- **Numerical** (`area`, `bedroom_num`) → `StandardScaler`
- **Categorical** (`locality`, `property_type`) → `OrdinalEncoder` with unknown handling

---

## 📊 Dataset

- **Source:** Mumbai real-estate listings (cleaned)
- **Size:** 71,938 rows → 70,491 after cleaning
- **Columns used:** `area`, `locality`, `property_type`, `bedroom_num`, `price`
- **Cleaning steps:**
  - Removed top/bottom 1% price outliers
  - Filtered area to 150–10,000 sq ft
  - Kept BHK range 1–8
  - Dropped rows with null values in key columns

---

## 🖥️ Dashboard Walkthrough

1. **Sidebar** — Select locality, property type, BHK (slider), and area (slider)
2. **Metric Cards** — View predicted price, low estimate, and high estimate
3. **Summary Table** — All inputs + price-per-sqft at a glance
4. **Bar Chart** — Visual comparison of low / predicted / high price
5. **Sensitivity Table** — Price estimates across ±500 sq ft range

---

## 🔮 Future Improvements

- [ ] Add XGBoost / LightGBM for model comparison
- [ ] SHAP explainability plots to show why a price was predicted
- [ ] Deploy to Streamlit Cloud for public access
- [ ] Add more features: furnished status, floor number, age of property
- [ ] Hyperparameter tuning with GridSearchCV

---

## 👤 Author

**Your Name**
- GitHub: [@shravanpadhar](https://github.com/shravanpadhar)
- LinkedIn: [linkedin.com/in/shravanpadhar](https://linkedin.com/in/shravanpadhar)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> ⭐ If you found this project useful, consider giving it a star on GitHub!
