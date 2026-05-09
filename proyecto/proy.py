"""
CC3085 - Inteligencia Artificial
Proyecto #3: Clasificador SPAM/HAM con Naive Bayes
"""

import pandas as pd
import numpy as np
import re
import math
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
CSV_PATH = "spam_ham_procesado.csv"    # <-- tu CSV ya procesado
THRESHOLD = 0.3                # umbral de clasificación — más bajo favorece detectar SPAM
TEST_SIZE = 0.2                # 20% para testing
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path, encoding="latin-1")
    df = df[["label", "text", "text_clean"]].dropna(subset=["text_clean"])
    df["label"] = df["label"].str.strip().str.lower()
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    print(f"\n✅ Dataset cargado: {len(df)} filas")
    return df


# ─────────────────────────────────────────────
# 2. EDA  (Análisis Exploratorio)
# ─────────────────────────────────────────────
def eda(df):
    print("\n" + "="*55)
    print("  EDA - ANÁLISIS EXPLORATORIO")
    print("="*55)

    counts = df["label"].value_counts()
    print(f"\nDistribución de clases:")
    for label, cnt in counts.items():
        pct = cnt / len(df) * 100
        print(f"  {label.upper():5s}: {cnt:5d} mensajes  ({pct:.1f}%)")

    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df["char_count"] = df["text"].apply(lambda x: len(str(x)))

    print(f"\nLongitud promedio de mensajes (palabras):")
    for label in ["ham", "spam"]:
        sub = df[df["label"] == label]
        print(f"  {label.upper():5s}: {sub['word_count'].mean():.1f} palabras  |  "
              f"{sub['char_count'].mean():.0f} caracteres")

    return df


# ─────────────────────────────────────────────
# 3. LIMPIEZA DE DATOS
# ─────────────────────────────────────────────
STOP_WORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each",
    "few","for","from","further","get","got","had","hadn't","has","hasn't",
    "have","haven't","having","he","he'd","he'll","he's","her","here",
    "here's","hers","herself","him","himself","his","how","how's","i","i'd",
    "i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
    "itself","let's","me","more","most","mustn't","my","myself","no","nor",
    "not","of","off","on","once","only","or","other","ought","our","ours",
    "ourselves","out","over","own","same","shan't","she","she'd","she'll",
    "she's","should","shouldn't","so","some","such","than","that","that's",
    "the","their","theirs","them","themselves","then","there","there's",
    "these","they","they'd","they'll","they're","they've","this","those",
    "through","to","too","under","until","up","very","was","wasn't","we",
    "we'd","we'll","we're","we've","were","weren't","what","what's","when",
    "when's","where","where's","which","while","who","who's","whom","why",
    "why's","will","with","won't","would","wouldn't","you","you'd","you'll",
    "you're","you've","your","yours","yourself","yourselves","u","ur","ok",
    "la","n","e","wif","oni","lar","wat","hor","c"
}

def simple_stem(word):
    """Stemming básico: quita sufijos comunes en inglés."""
    suffixes = ["ing", "tion", "ed", "er", "ly", "ies", "es", "s"]
    for suf in suffixes:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: len(word) - len(suf)]
    return word

def clean_text(text):
    """
    Pasos de limpieza:
    1. Minúsculas
    2. Eliminar caracteres no alfabéticos
    3. Tokenizar
    4. Eliminar stop words
    5. Stemming básico
    6. Descartar tokens muy cortos
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [simple_stem(t) for t in tokens]
    tokens = [t for t in tokens if len(t) >= 3]
    return tokens

def preprocess(df):
    print("\n" + "="*55)
    print("  LIMPIEZA DE DATOS (usando columna text_clean del CSV)")
    print("="*55)
    # Ya viene preprocesado: solo tokenizamos por espacio
    df["tokens"] = df["text_clean"].apply(lambda x: str(x).split())
    avg_before = df["word_count"].mean()
    avg_after  = df["tokens"].apply(len).mean()
    print(f"  Promedio palabras ANTES de limpiar : {avg_before:.1f}")
    print(f"  Promedio tokens  DESPUÉS de limpiar: {avg_after:.1f}")
    print(f"  Reducción aproximada: {(1 - avg_after/avg_before)*100:.1f}%")
    return df


# ─────────────────────────────────────────────
# 4. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────
def train_test_split(df, test_size=TEST_SIZE, seed=RANDOM_SEED):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(len(df) * (1 - test_size))
    train = df.iloc[:split].copy()
    test  = df.iloc[split:].copy()
    print(f"\n  Train: {len(train)} muestras | Test: {len(test)} muestras")
    return train, test


# ─────────────────────────────────────────────
# 5. ENTRENAMIENTO DEL MODELO NAIVE BAYES
# ─────────────────────────────────────────────
class NaiveBayesSpam:
    """
    Clasificador Naive Bayes para SPAM/HAM.

    Para cada palabra W se calcula P(S|W) con la fórmula de Bayes:
        P(S|W) = P(W|S)·P(S) / [P(W|S)·P(S) + P(W|H)·P(H)]

    Para combinar múltiples palabras W1…Wn:
        p = p1·p2·…·pn / [p1·p2·…·pn + (1-p1)·(1-p2)·…·(1-pn)]
    """

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing      # Laplace smoothing
        self.word_spam = defaultdict(int)
        self.word_ham  = defaultdict(int)
        self.p_spam = 0.0
        self.p_ham  = 0.0
        self.vocab  = set()
        self.word_p_spam = {}           # P(S|W) por palabra

    # ── Entrenamiento ──────────────────────────
    def fit(self, train_df):
        spam_msgs = train_df[train_df["label"] == "spam"]
        ham_msgs  = train_df[train_df["label"] == "ham"]

        n_total = len(train_df)
        self.p_spam = len(spam_msgs) / n_total
        self.p_ham  = len(ham_msgs)  / n_total

        # Conteo de palabras por clase
        for tokens in spam_msgs["tokens"]:
            for t in tokens:
                self.word_spam[t] += 1
                self.vocab.add(t)

        for tokens in ham_msgs["tokens"]:
            for t in tokens:
                self.word_ham[t] += 1
                self.vocab.add(t)

        total_spam_words = sum(self.word_spam.values())
        total_ham_words  = sum(self.word_ham.values())
        vocab_size = len(self.vocab)

        print(f"\n  Vocabulario único: {vocab_size} palabras")
        print(f"  P(SPAM) = {self.p_spam:.4f}  |  P(HAM) = {self.p_ham:.4f}")

        # Calcular P(S|W) para cada palabra del vocabulario
        for word in self.vocab:
            # P(W|S) con Laplace smoothing
            p_w_given_spam = (self.word_spam[word] + self.smoothing) / \
                             (total_spam_words + self.smoothing * vocab_size)
            # P(W|H) con Laplace smoothing
            p_w_given_ham  = (self.word_ham[word]  + self.smoothing) / \
                             (total_ham_words  + self.smoothing * vocab_size)

            numerator   = p_w_given_spam * self.p_spam
            denominator = numerator + p_w_given_ham * self.p_ham

            self.word_p_spam[word] = numerator / denominator if denominator > 0 else 0.5

        return self

    # ── Predicción ────────────────────────────
    def predict_proba(self, tokens):
        """
        Combina las probabilidades por palabra usando la fórmula del producto:
            prob = Π(pi) / [Π(pi) + Π(1-pi)]
        Se hace en log-space para evitar underflow numérico.

        Palabras desconocidas reciben P(S|W) = 0.5 (neutral, no penalizan).
        """
        # Palabras conocidas usan su P(S|W); desconocidas = 0.5 (neutrales)
        probs = []
        for t in tokens:
            if t in self.word_p_spam:
                probs.append(self.word_p_spam[t])
            else:
                probs.append(0.5)   # desconocida: no aporta evidencia

        if not probs:
            return self.p_spam          # sin tokens, devuelve prior

        # Clamp para evitar log(0)
        probs = [max(min(p, 1 - 1e-10), 1e-10) for p in probs]

        log_p   = sum(math.log(p)       for p in probs)
        log_1_p = sum(math.log(1 - p)   for p in probs)

        # Trick numérico: restar el máximo
        max_log = max(log_p, log_1_p)
        exp_p   = math.exp(log_p   - max_log)
        exp_1_p = math.exp(log_1_p - max_log)

        return exp_p / (exp_p + exp_1_p)

    def predict(self, tokens, threshold=THRESHOLD):
        return "spam" if self.predict_proba(tokens) >= threshold else "ham"

    def top_spam_words(self, tokens, n=3):
        """Retorna las n palabras con mayor P(S|W) presentes en el texto."""
        word_scores = [(t, self.word_p_spam[t])
                       for t in tokens if t in self.word_p_spam]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return word_scores[:n]


# ─────────────────────────────────────────────
# 6. EVALUACIÓN
# ─────────────────────────────────────────────
def evaluate(model, test_df, threshold=THRESHOLD):
    y_true, y_pred, y_prob = [], [], []

    for _, row in test_df.iterrows():
        prob = model.predict_proba(row["tokens"])
        pred = "spam" if prob >= threshold else "ham"
        y_true.append(row["label"])
        y_pred.append(pred)
        y_prob.append(prob)

    # Matriz de confusión
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "spam" and p == "spam")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "ham"  and p == "ham")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "ham"  and p == "spam")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "spam" and p == "ham")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (tp + tn) / len(y_true)

    print(f"\n  Threshold usado: {threshold}")
    print(f"\n  Matriz de Confusión:")
    print(f"              Pred HAM   Pred SPAM")
    print(f"  Real HAM  :    {tn:5d}      {fp:5d}")
    print(f"  Real SPAM :    {fn:5d}      {tp:5d}")
    print(f"\n  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def threshold_sweep(model, test_df):
    """Prueba distintos thresholds y muestra una tabla comparativa."""
    print("\n" + "="*55)
    print("  BARRIDO DE THRESHOLDS")
    print("="*55)
    print(f"\n  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}")
    print("  " + "-"*50)

    y_true = list(test_df["label"])
    y_prob = [model.predict_proba(row["tokens"]) for _, row in test_df.iterrows()]

    best_f1 = -1
    best_th = THRESHOLD

    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred = ["spam" if p >= th else "ham" for p in y_prob]
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == "spam" and p == "spam")
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == "ham"  and p == "spam")
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == "spam" and p == "ham")
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == "ham"  and p == "ham")

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc  = (tp + tn) / len(y_true)

        marker = " ← mejor F1" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

        print(f"  {th:>10.2f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {acc:>10.4f}{marker}")

    print(f"\n  ✅ Mejor threshold por F1: {best_th}")
    return best_th


# ─────────────────────────────────────────────
# 7. MÓDULO INTERACTIVO (presentación en vivo)
# ─────────────────────────────────────────────
def interactive_module(model):
    print("\n" + "="*55)
    print("  MÓDULO INTERACTIVO - Clasificador SPAM/HAM")
    print("  (escribe 'salir' para terminar)")
    print("="*55)

    while True:
        print()
        text = input("📩 Ingresa un mensaje: ").strip()
        if text.lower() in ("salir", "exit", "quit"):
            print("  Hasta luego 👋")
            break

        tokens = clean_text(text)
        prob   = model.predict_proba(tokens)
        pred   = "🚨 SPAM" if prob >= THRESHOLD else "✅ HAM"
        top3   = model.top_spam_words(tokens, n=3)

        print(f"\n  Resultado      : {pred}")
        print(f"  P(SPAM)        : {prob:.4f}  ({prob*100:.1f}%)")
        print(f"  Top 3 palabras predictoras de SPAM:")
        if top3:
            for i, (word, score) in enumerate(top3, 1):
                print(f"    {i}. '{word}'  →  P(S|W) = {score:.4f}")
        else:
            print("    (ninguna palabra reconocida en el vocabulario)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  CC3085 - Clasificador SPAM/HAM  |  Naive Bayes")
    print("="*55)

    # 1. Cargar
    df = load_data(CSV_PATH)

    # 2. EDA
    df = eda(df)

    # 3. Limpiar
    df = preprocess(df)

    # 4. Split
    print("\n" + "="*55)
    print("  SPLIT TRAIN/TEST (80/20)")
    print("="*55)
    train_df, test_df = train_test_split(df)

    # 5. Entrenar
    print("\n" + "="*55)
    print("  ENTRENAMIENTO DEL MODELO")
    print("="*55)
    model = NaiveBayesSpam(smoothing=1.0).fit(train_df)

    # 6. Evaluar con threshold por defecto
    print("\n" + "="*55)
    print("  EVALUACIÓN EN TEST SET")
    print("="*55)
    metrics = evaluate(model, test_df, threshold=THRESHOLD)

    # 7. Barrido de thresholds
    best_threshold = threshold_sweep(model, test_df)

    # 8. Re-evaluar con mejor threshold
    print("\n" + "="*55)
    print(f"  EVALUACIÓN CON MEJOR THRESHOLD ({best_threshold})")
    print("="*55)
    evaluate(model, test_df, threshold=best_threshold)

    # 9. Módulo interactivo
    interactive_module(model)
