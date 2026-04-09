# 🐋 FCM–WOA Customer Clustering

This project implements Fuzzy C-Means (FCM) combined with the Whale Optimization Algorithm (WOA) to perform customer clustering based on age, income, and spending.

--------------------------------------------------

🔗 Live App
https://customer-clustering-fuzzy-c-means-whale-optimization-algorithm.streamlit.app/

--------------------------------------------------

 Overview

- KMeans → hard clustering (one cluster per point)
- FCM → soft clustering (multiple memberships)
- WOA-FCM → optimized soft clustering (better cluster centers)

WOA improves FCM by reducing the objective function and producing more stable clusters.

--------------------------------------------------

Features

- Upload customer dataset (CSV)
- Perform clustering using:
  - KMeans
  - FCM
  - WOA-FCM
- Visual comparison of clustering results
- Objective score comparison (FCM vs WOA)

--------------------------------------------------

 Input Format

CSV file must contain:

- age → Customer age
- income → Annual income
- spending → Spending score

--------------------------------------------------

 Run Locally

pip install -r requirements.txt  
streamlit run app.py

--------------------------------------------------

 Deployment

Deployed using Streamlit Cloud

--------------------------------------------------

 Key Result

FCM Objective : 65.67  
WOA Objective : 65.48  

Lower value indicates better clustering  
WOA improves FCM performance

--------------------------------------------------

 Conclusion

WOA enhances FCM by optimizing cluster centers, resulting in improved clustering quality and stability.
