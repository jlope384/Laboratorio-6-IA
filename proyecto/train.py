"""
CC3085 - Inteligencia Artificial
Proyecto #3: Clasificador SPAM/HAM con Naive Bayes (log-odds + features estructurales)
Archivo: train.py  —  entrena el modelo y guarda model.toml

Mejoras v2:
  - Más features estructurales (ALL_CAPS, call-to-action, urgencia, SMS shortcodes)
  - Tokens desconocidos reciben log-odds = 0 (neutral) en lugar de ignorarse
  - Barrido de threshold más fino (0.05 steps)
  - TOP_N ampliado a 20 para capturar más evidencia por mensaje
"""

import pandas as pd
import nltk
import string
import math
import tomli_w
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

# ── hiperparámetros ───────────────────────────────────────────────────────────
CSV_PATH          = "spam_ham_procesado.csv"
TOP_N_WORDS       = 20
LAPLACE_ALPHA     = 0.5
RANDOM_STATE      = 42
STRUCTURAL_WEIGHT = 3

# ── 1. carga de datos ─────────────────────────────────────────────────────────
data_set = pd.read_csv(CSV_PATH, encoding="latin-1")
data_set = data_set.dropna()
data_set.columns = data_set.columns.str.strip()
data_set = data_set.rename(columns={"label": "Label", "text": "SMS_TEXT"})
data_set["Label"] = data_set["Label"].str.strip().str.lower()
data_set = data_set[data_set["Label"].isin(["spam", "ham"])]

print(f"\n{'='*60}")
print(f"  EDA - ANÁLISIS EXPLORATORIO")
print(f"{'='*60}")
print(f"\n  Total mensajes : {len(data_set)}")
print(f"  Distribución   :")
for label, cnt in data_set["Label"].value_counts().items():
    print(f"    {label.upper():5s}: {cnt:5d}  ({cnt/len(data_set)*100:.1f}%)")

data_set["word_count"] = data_set["SMS_TEXT"].apply(lambda x: len(str(x).split()))
data_set["char_count"] = data_set["SMS_TEXT"].apply(len)
print(f"\n  Longitud promedio:")
for label in ["ham", "spam"]:
    sub = data_set[data_set["Label"] == label]
    print(f"    {label.upper():5s}: {sub['word_count'].mean():.1f} palabras  |  {sub['char_count'].mean():.0f} caracteres")

# ── 2. features estructurales ─────────────────────────────────────────────────
def extract_structural_features(text: str) -> list:
    features = []
    w = int(STRUCTURAL_WEIGHT)

    if re.search(r'[\$£€]|\bprize\b|\bwon\b|\bwin\b|\bcash\b|\bfree\b|\breward\b|\bgift\b', text, re.I):
        features += ["__MONEY_OR_PRIZE__"] * w
    if re.search(r'\b\d{5,6}\b|\b\d{3}[-.\s]\d{4}\b', text):
        features += ["__PHONE_NUMBER__"] * w
    if re.search(r'http|www\.|\.com|\.net|bit\.ly|tinyurl|wap\.', text, re.I):
        features += ["__URL__"] * w
    if len(re.findall(r'[!?]{2,}', text)) >= 1:
        features += ["__EXCLAMATION__"] * w
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    if digit_ratio > 0.12:
        features += ["__DIGIT_HEAVY__"] * w
    upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if upper_ratio > 0.25:
        features += ["__ALL_CAPS__"] * w
    if re.search(r'\b(call|txt|text|click|reply|send|stop|claim|subscribe)\b', text, re.I):
        features += ["__CALL_TO_ACTION__"] * w
    if re.search(r'\b(urgent|immediately|expire|limited|hurry|now|today only|asap)\b', text, re.I):
        features += ["__URGENCY__"] * w
    if re.search(r'\b(unsubscribe|opt.?out|terms|t&c|18\+|16\+)\b', text, re.I):
        features += ["__MARKETING__"] * w

    return features

# ── 3. preprocesamiento ───────────────────────────────────────────────────────
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
punct_set  = set(string.punctuation)

def preprocess(text: str) -> list:
    structural = extract_structural_features(text)
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t not in punct_set]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    bigrams = [f"__{tokens[i]}_{tokens[i+1]}__" for i in range(len(tokens) - 1)]
    return structural + tokens + bigrams

print(f"\n{'='*60}")
print(f"  LIMPIEZA DE DATOS")
print(f"{'='*60}")
data_set["tokens"] = data_set["SMS_TEXT"].apply(preprocess)
avg_before = data_set["word_count"].mean()
avg_after  = data_set["tokens"].apply(len).mean()
print(f"\n  Promedio palabras antes : {avg_before:.1f}")
print(f"  Promedio tokens después : {avg_after:.1f}")
print(f"  Expansión por bigramas/features: +{avg_after - avg_before:.1f} tokens/msg")
print(f"  Técnicas: tokenización, lowercase, stop words, lemmatización,")
print(f"            bigramas, 9 features estructurales")

# ── 4. split 80/20 ────────────────────────────────────────────────────────────
train_set = data_set.sample(frac=0.8, random_state=RANDOM_STATE)
test_set  = data_set.drop(train_set.index)

print(f"\n{'='*60}")
print(f"  SPLIT TRAIN/TEST")
print(f"{'='*60}")
print(f"\n  Train : {len(train_set)} muestras  ({len(train_set)/len(data_set)*100:.0f}%)")
print(f"  Test  : {len(test_set)} muestras  ({len(test_set)/len(data_set)*100:.0f}%)")

# ── 5. entrenamiento ──────────────────────────────────────────────────────────
spam_train = train_set[train_set["Label"] == "spam"]
ham_train  = train_set[train_set["Label"] == "ham"]

p_spam = len(spam_train) / len(train_set)
p_ham  = len(ham_train)  / len(train_set)

spam_word_counts = Counter(w for ts in spam_train["tokens"] for w in ts)
ham_word_counts  = Counter(w for ts in ham_train["tokens"]  for w in ts)

vocab            = set(spam_word_counts) | set(ham_word_counts)
vocab_size       = len(vocab)
total_spam_words = sum(spam_word_counts.values())
total_ham_words  = sum(ham_word_counts.values())

print(f"\n{'='*60}")
print(f"  MODELO - Naive Bayes con log-odds")
print(f"{'='*60}")
print(f"\n  P(SPAM)     = {p_spam:.4f}")
print(f"  P(HAM)      = {p_ham:.4f}")
print(f"  Vocabulario = {vocab_size} tokens únicos")
print(f"  Alpha       = {LAPLACE_ALPHA} (Laplace smoothing)")
print(f"  TOP_N       = {TOP_N_WORDS} tokens más discriminativos por mensaje")

def p_word_given_spam(word):
    return (spam_word_counts.get(word, 0) + LAPLACE_ALPHA) / (total_spam_words + LAPLACE_ALPHA * vocab_size)

def p_word_given_ham(word):
    return (ham_word_counts.get(word, 0) + LAPLACE_ALPHA) / (total_ham_words + LAPLACE_ALPHA * vocab_size)

def p_spam_given_word(word):
    pws = p_word_given_spam(word)
    pwh = p_word_given_ham(word)
    num = pws * p_spam
    den = num + pwh * p_ham
    return num / den if den > 0 else 0.5

def log_odds_given_word(word):
    return math.log(p_word_given_spam(word) / p_word_given_ham(word))

def classify_message(tokens, threshold):
    known   = [w for w in tokens if w in vocab]
    unknown = [w for w in tokens if w not in vocab]

    word_log_odds = {w: log_odds_given_word(w) for w in known}
    # Tokens desconocidos = log-odds 0 (neutral, no arrastran al prior de HAM)
    for w in unknown:
        word_log_odds[w] = 0.0

    top_n = sorted(word_log_odds.items(), key=lambda x: abs(x[1]), reverse=True)[:TOP_N_WORDS]

    prior_log_odds = math.log(p_spam / p_ham)
    total_log_odds = prior_log_odds + sum(lo for _, lo in top_n)
    prob  = 1.0 / (1.0 + math.exp(-total_log_odds))
    label = "spam" if prob >= threshold else "ham"

    word_probs = {w: p_spam_given_word(w) for w in known
                  if not (w.startswith("__") and w.endswith("__"))}
    top3 = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:3]

    return prob, label, top3

# ── 6. evaluación — barrido fino de thresholds ────────────────────────────────
print(f"\n{'='*60}")
print(f"  PRUEBAS DE RENDIMIENTO")
print(f"{'='*60}")
print(f"\n  {'Threshold':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'Spam Recall':>12}")
print(f"  {'-'*60}")

y_true = test_set["Label"].apply(lambda x: 1 if x == "spam" else 0).tolist()

best_f1        = -1.0
best_threshold = 0.5
best_metrics   = {}

for threshold in [round(0.05 * i, 2) for i in range(1, 20)]:
    preds = [1 if classify_message(ts, threshold)[1] == "spam" else 0
             for ts in test_set["tokens"]]

    precision   = precision_score(y_true, preds, zero_division=0)
    recall      = recall_score(y_true, preds, zero_division=0)
    f1          = f1_score(y_true, preds, zero_division=0)
    cm          = confusion_matrix(y_true, preds)
    spam_recall = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0.0

    marker = " <-- mejor F1" if f1 > best_f1 else ""
    print(f"  {threshold:>10.2f} | {precision:>10.4f} | {recall:>8.4f} | {f1:>8.4f} | {spam_recall:>12.4f}{marker}")

    if f1 > best_f1:
        best_f1        = f1
        best_threshold = threshold
        best_metrics   = {"precision": precision, "recall": recall,
                          "f1": f1, "preds": preds, "spam_recall": spam_recall}

cm = confusion_matrix(y_true, best_metrics["preds"])
print(f"\n  Mejor threshold : {best_threshold}")
print(f"  Precision       : {best_metrics['precision']:.4f}")
print(f"  Recall          : {best_metrics['recall']:.4f}")
print(f"  F1-score        : {best_metrics['f1']:.4f}")
print(f"  Spam Recall     : {best_metrics['spam_recall']:.4f}")
print(f"\n  Matriz de confusión:")
print(f"                  HAM    SPAM")
print(f"    Real HAM  : {cm[0][0]:>5}  {cm[0][1]:>5}")
print(f"    Real SPAM : {cm[1][0]:>5}  {cm[1][1]:>5}")

print(f"\n  Top 15 palabras más indicativas de SPAM:")
spam_indicators = sorted(
    [(w, p_spam_given_word(w)) for w in vocab if not w.startswith("__")],
    key=lambda x: x[1], reverse=True
)[:15]
for word, prob in spam_indicators:
    print(f"    {word:<25} P(spam|word) = {prob:.4f}")

# ── 7. guardar model.toml ─────────────────────────────────────────────────────
model_data = {
    "priors":      {"p_spam": p_spam, "p_ham": p_ham},
    "vocab":       {"total_spam_words": total_spam_words, "total_ham_words": total_ham_words},
    "word_counts": {"spam": dict(spam_word_counts), "ham": dict(ham_word_counts)},
    "split":       {"train_size": len(train_set), "test_size": len(test_set), "random_state": RANDOM_STATE},
    "hyperparams": {"laplace_alpha": LAPLACE_ALPHA, "top_n_words": TOP_N_WORDS, "structural_weight": float(STRUCTURAL_WEIGHT)},
    "best_threshold": best_threshold,
}

with open("model.toml", "wb") as f:
    tomli_w.dump(model_data, f)

print(f"\n  ✅ Modelo guardado en model.toml")