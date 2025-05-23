
# 📬 Spam Classifier with Machine Learning

This project consists of a spam classifier developed using Machine Learning techniques, based on a practical exercise from the book *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron.

The goal is to train models capable of distinguishing between legitimate emails (ham) and spam with high precision and recall, through a preprocessing pipeline and evaluation of multiple classification algorithms.

**NOTE: Some of the cells may take a while to run, please be patient!** 

---

## 📂 Data Used

The data used for training and testing the models comes from the [SpamAssassin Public Corpus](http://spamassassin.apache.org/old/publiccorpus/), a collection of spam and ham emails publicly available for spam filter research.

---

## ⚙️ Technologies and Libraries

- Python 3  
- scikit-learn  
- XGBoost  
- NLTK  
- BeautifulSoup4  
- pandas, numpy, matplotlib  
- urlextract  
- (among others already pre-installed with Python)

---

## 🚀 How to run the project

1. **Clone the repository:**

```bash
git clone https://github.com/franciscoamador13/Spam-filter.git
cd SpamClassifier
```
2. **Install jupyter notebook:**
```
pip install notebook
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the notebook:**

```bash
jupyter notebook SpamClassifier.ipynb
```

---

## 🧠 What the project does

- Text cleaning and normalization
- Feature extraction from word counts
- Creation of a custom transformer with Scikit-Learn
- Model evaluation with **precision** and **recall** metrics
- Comparison of classifiers (Logistic Regression, XGBoost, etc.)

---

## 📚 Source and Credits

This project was inspired by and partially based on the book:

> *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* – Aurélien Géron

Parts of the code were adapted from the official repository::
[https://github.com/ageron/handson-ml3](https://github.com/ageron/handson-ml3)

Specific notebook:
https://github.com/ageron/handson-ml3/blob/main/03_classification.ipynb

---

## 📝 License

This project is licensed under the **Apache License 2.0**.  
See the [`LICENSE`](LICENSE) file for more details.
