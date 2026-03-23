# 📬 Spam Classifier with Machine Learning

This project consists of a spam classifier developed using Machine Learning techniques, based on a practical exercise from the book *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron.

The goal is to train models capable of distinguishing between legitimate emails (ham) and spam with high precision and recall, through a preprocessing pipeline, evaluation of multiple classification algorithms, and a REST API for real-time predictions.

**NOTE: Some of the cells may take a while to run, please be patient!**

---

## 📂 Data Used

The data used for training and testing the models comes from the [SpamAssassin Public Corpus](http://spamassassin.apache.org/old/publiccorpus/), a collection of spam and ham emails publicly available for spam filter research.

---

## ⚙️ Technologies and Libraries

- Python 3
- scikit-learn
- XGBoost
- FastAPI + Uvicorn
- NLTK
- BeautifulSoup4
- pandas, numpy, matplotlib
- urlextract

---

## 🚀 How to run the project

1. **Clone the repository:**
```bash
git clone https://github.com/franciscoamador13/Spam-filter.git
cd Spam-filter
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the notebook** (to train and save the model):
```bash
jupyter notebook Notebooks/SpamClassifier.ipynb
```

4. **Start the API:**
```bash
uvicorn app:app --reload
```

5. **Test the API:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Congratulations! You won a free laptop. Click here to claim your prize!\"}"
```

---

## 🧠 What the project does

- Text cleaning and normalization
- Feature extraction from word counts
- Creation of custom transformers with Scikit-Learn
- Model evaluation with **precision** and **recall** metrics
- Comparison of classifiers (Logistic Regression, XGBoost, Random Forest, etc.)
- REST API with FastAPI to serve predictions in real-time

---

## ⚠️ Limitations

- The model was trained exclusively on English emails from the SpamAssassin corpus, so it may not perform well on emails in other languages or on very short text inputs.
- For best results, provide full email content (only text, no HTML) rather than short phrases.

---

## 📚 Source and Credits

This project was inspired by and partially based on the book:

> *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* – Aurélien Géron

Parts of the code were adapted from the official repository:
[https://github.com/ageron/handson-ml3](https://github.com/ageron/handson-ml3)

---

## 📝 License

This project is licensed under the **Apache License 2.0**.
See the [`LICENSE`](LICENSE) file for more details.
