# 📨 SPAM-HAM Detector

A simple **spam/ham email classifier** implemented in Python using a **Bag-of-Words (BOW) model** and **centroid-based classification**.  
This project demonstrates basic NLP preprocessing, vectorization, and model evaluation without relying on external ML libraries for prediction. 🚀

---

## ✨ Features

- 🧹 **Text preprocessing**: removes punctuation, converts to lowercase.  
- 📊 **Bag-of-Words representation**: creates a matrix of word counts from the training dataset.  
- 🧭 **Centroid-based classifier**: predicts spam or ham using Euclidean distance to class centroids.  
- 📈 **Evaluation metrics**: computes accuracy and number of misclassified samples.  
- 🏗️ Fully **modular Python package** structure (`src/`) for clean imports and reusability.

---

## 🗂 Project Structure

```text
SPAM_HAM_project/
├─ data/
│  └─ spamhamdata.csv        # Dataset (Category, Text)
├─ src/
│  ├─ __init__.py            # Package initializer
│  ├─ vectorizer.py          # BOW functions
│  ├─ model.py               # Centroid calculation & prediction
│  ├─ evaluate.py            # Evaluation functions
│  └─ preprocessing.py       # Text cleaning (to_alpha)
├─ main.py                   # Main script to run the pipeline
└─ .gitignore                # Ignored files (venv, pycache, IDE configs)
```

---

## ⚙️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/hhtaron2004-tech/spam-ham-detector.git
cd spam-ham-detector
```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> Current dependencies: pandas, numpy, scikit-learn.

---

## ▶️ Usage

Run the project from the terminal:

```bash
python -m main
```

Example output:

```Plain text
Test samples: 1393
Misclassified: 105
Accuracy: 0.9246
```

> ⚠️ Ensure your virtual environment is activated before running the script.


---

## 📚 Dataset

- Dataset file: `data/spamhamdata.csv`  
- Format: tab-delimited (`\t`), with two columns:

```text
Category    Text
spam        Free entry in 2 a weekly competition!
ham         Hey, are we meeting today?
```

- Category: spam or ham — the label for each message.
- Text: the content of the message.

> 💡 Make sure your CSV file uses tabs as delimiters, otherwise the script may fail to parse it correctly.
📝 The dataset should be in the data/ folder at the project root.


---

## 📝 Project Workflow
1. Preprocess text (to_alpha) → remove punctuation, lowercase, keep words ≥ 2 characters.

2. Build vocabulary from training data.

3. Convert texts to Bag-of-Words matrices.

4. Compute centroids for spam and ham.

5. Predict labels for test data using Euclidean distance.

6. Evaluate accuracy and misclassifications.


>🧩 This workflow is implemented in main.py and is fully modular using the src/ package.


---


## 🤝 Contributing

1. Fork the repository.  
2. Create a branch for your feature:

```bash
git checkout -b feature-name
```

3. Make your changes and commit:

```bash
git commit -m "Add meaningful message"
```

4. Push to your fork and open a Pull Request 

>💡 Follow standard GitHub workflow for contributions and maintain descriptive commit messages.


---

## 📄 License


MIT License – feel free to use, modify, and redistribute. ✅
