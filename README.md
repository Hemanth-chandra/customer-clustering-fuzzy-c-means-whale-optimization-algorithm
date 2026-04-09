# 🐋 FCM-WOA Customer Segmentation App

Fuzzy C-Means + Whale Optimization Algorithm for segmenting customers into:
- 🟡 **High Spenders**
- 🔵 **Moderate Spenders**  
- 🔴 **Low Spenders**

---

## 🚀 Deploy on Streamlit Cloud

### Step 1 — Push to GitHub
```bash
git init
git add app.py requirements.txt README.md
git commit -m "FCM-WOA app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2 — Deploy
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **New app**
3. Select your repo → Branch: `main` → Main file: `app.py`
4. Click **Deploy** ✅

---

## 💻 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Required CSV columns
| Column | Description |
|--------|------------|
| `age` | Customer age |
| `income` | Annual income |
| `spending` | Annual spending |

Your `customer_data.csv` already has these columns — just upload it directly.
