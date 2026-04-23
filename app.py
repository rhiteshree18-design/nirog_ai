from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import os

from PIL import Image
import pytesseract
from werkzeug.utils import secure_filename

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "nirog-secret-key")
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

TESSERACT_PATH = os.environ.get("TESSERACT_PATH")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

df = pd.read_csv("Medicine_Details.csv")
df = df.fillna("")
df.columns = [c.strip() for c in df.columns]

def find_col(possible):
    for col in df.columns:
        for p in possible:
            if p.lower() in col.lower():
                return col
    return None

NAME_COL = find_col(["name", "medicine"])
COMP_COL = find_col(["composition", "salt"])
USE_COL = find_col(["use"])
SIDE_COL = find_col(["side", "effect"])
MANU_COL = find_col(["manufacturer", "company"])
RATING_COL = find_col(["rating"])

if NAME_COL is None:
    NAME_COL = df.columns[0]
if COMP_COL is None:
    COMP_COL = NAME_COL
if USE_COL is None:
    USE_COL = NAME_COL
if SIDE_COL is None:
    SIDE_COL = NAME_COL
if MANU_COL is None:
    MANU_COL = NAME_COL

SEVERE_WORDS = [
    "bleeding", "stroke", "heart attack", "liver damage",
    "kidney failure", "seizure", "coma", "vision loss"
]

MODERATE_WORDS = [
    "vomiting", "rash", "dizziness", "fatigue",
    "palpitations", "swelling", "fever"
]

def severity_from_row(row):
    txt = str(row[SIDE_COL]).lower()

    rating = 4.0
    if RATING_COL:
        try:
            rating = float(row[RATING_COL])
        except:
            rating = 4.0

    severe_hits = sum(word in txt for word in SEVERE_WORDS)
    moderate_hits = sum(word in txt for word in MODERATE_WORDS)

    if severe_hits > 0 or rating < 3.5:
        return "Severe"
    elif moderate_hits > 0 or rating < 4.2:
        return "Moderate"
    return "Mild"

df["severity"] = df.apply(severity_from_row, axis=1)

if df["severity"].nunique() < 2:
    df["severity"] = np.where(
        np.arange(len(df)) % 3 == 0,
        "Mild",
        np.where(np.arange(len(df)) % 3 == 1, "Moderate", "Severe")
    )

df["combined"] = (
    df[NAME_COL].astype(str) + " " +
    df[COMP_COL].astype(str) + " " +
    df[USE_COL].astype(str) + " " +
    df[SIDE_COL].astype(str) + " " +
    df[MANU_COL].astype(str)
)

X = df["combined"]
y = df["severity"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_enc,
    test_size=0.20,
    random_state=42,
    stratify=y_enc
)

base_svm = LinearSVC(C=1.5, random_state=42)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 3),
            stop_words="english",
            sublinear_tf=True
        )
    ),
    (
        "clf",
        CalibratedClassifierCV(base_svm, cv=5)
    )
])

model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

@app.route("/")
def home():
    drugs = sorted(df[NAME_COL].astype(str).unique().tolist())[:5000]
    return render_template(
        "index.html",
        drugs=drugs,
        accuracy=round(accuracy * 100, 2)
    )

@app.route("/predict", methods=["POST"])
def predict():
    drug = request.form["drug_name"]

    row = df[df[NAME_COL].astype(str) == drug].head(1)

    if row.empty:
        return render_template("result.html", error="Medicine not found")

    row = row.iloc[0]

    combined = f"""
    {row[NAME_COL]}
    {row[COMP_COL]}
    {row[USE_COL]}
    {row[SIDE_COL]}
    {row[MANU_COL]}
    """

    pred = model.predict([combined])[0]
    probs = model.predict_proba([combined])[0]

    severity = le.inverse_transform([pred])[0]
    confidence = probs.max() * 100

    effects = str(row[SIDE_COL]).replace(";", ",").split(",")

    return render_template(
        "result.html",
        drug=drug,
        severity=severity,
        probability=round(confidence, 2),
        effects=effects,
        info=row
    )

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").lower()

    result = df[
        df[NAME_COL].astype(str).str.lower().str.contains(q, na=False)
    ][[NAME_COL]].head(10)

    return jsonify(result.to_dict(orient="records"))

@app.route("/upload-image", methods=["POST"])
def upload_image():
    from difflib import get_close_matches

    if "medicine_image" not in request.files:
        return render_template("result.html", error="No image uploaded")

    file = request.files["medicine_image"]

    if file.filename == "":
        return render_template("result.html", error="No image selected")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img = Image.open(filepath).convert("RGB")

    extracted_text = pytesseract.image_to_string(
        img,
        config="--psm 6"
    )

    extracted_text = re.sub(r"[^A-Za-z0-9 ]", " ", extracted_text)

    all_medicines = df[NAME_COL].astype(str).tolist()
    words = extracted_text.split()

    found = None

    for word in words:
        match = get_close_matches(word, all_medicines, n=1, cutoff=0.60)
        if match:
            found = match[0]
            break

    if not found:
        match = get_close_matches(extracted_text, all_medicines, n=1, cutoff=0.50)
        if match:
            found = match[0]

    if not found:
        return render_template(
            "result.html",
            error="Medicine not detected from image"
        )

    row = df[df[NAME_COL].astype(str) == found].head(1).iloc[0]

    combined = f"""
    {row[NAME_COL]}
    {row[COMP_COL]}
    {row[USE_COL]}
    {row[SIDE_COL]}
    {row[MANU_COL]}
    """

    pred = model.predict([combined])[0]
    probs = model.predict_proba([combined])[0]

    severity = le.inverse_transform([pred])[0]
    confidence = probs.max() * 100

    effects = str(row[SIDE_COL]).replace(";", ",").split(",")

    return render_template(
        "result.html",
        drug=found,
        severity=severity,
        probability=round(confidence, 2),
        effects=effects,
        info=row
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)