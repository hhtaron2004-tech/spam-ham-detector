# 📨 SpamHam Detector

A simple Python-based **spam/ham text classifier** using **Bag-of-Words** and **Euclidean Distance**.  

---

## 🌟 Features
- 📝 Converts raw text into bag-of-words representation
- 📊 Computes centroids for `spam` and `ham`
- 🤖 Classifies messages based on Euclidean distance
- 📈 Calculates misclassification count on test data

---

## ⚙️ Requirements
- **Python 3.x**
- **numpy**
- **pandas**
- **scikit-learn**

>
## 🚀 Usage

Place your dataset file (`spamhamdata.csv`) in the project folder.

Run your Python script implementing the detector.

The script will:
- Split dataset into train/test sets
- Convert messages to bag-of-words
- Compute centroids
- Classify test messages
- Print test size and misclassified count

## 📁 Dataset

The CSV file should be tab-separated with:

- Column 1: Category (spam or ham)  
- Column 2: Text (the message)

Example:

```text
spam	Win a free iPhone now!
ham	Are we still meeting tomorrow?
spam	Congratulations, you won a lottery!
ham	Can you send me the report by today?
```


## 📌 Notes

This project is educational and demonstrates:
- Text preprocessing
- Simple ML classification using centroids
- Basic evaluation metrics
- Bag-of-words feature extraction


```text
spamham-detector/
│
├── spamham_detector.py      # Main Python script (fixed bugs + better row handling)
├── spamhamdata_sample.csv   # Small example dataset
└── README.md                # GitHub README
```
