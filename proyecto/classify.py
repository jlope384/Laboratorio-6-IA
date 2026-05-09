"""
CC3085 - Inteligencia Artificial
Proyecto #3: Módulo interactivo — carga model.toml y clasifica mensajes en vivo
Archivo: classify.py  |  Uso: python classify.py
"""

import math, string, re, tomllib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

with open("model.toml", "rb") as f:
    model = tomllib.load(f)

p_spam            = model["priors"]["p_spam"]
p_ham             = model["priors"]["p_ham"]
total_spam_words  = model["vocab"]["total_spam_words"]
total_ham_words   = model["vocab"]["total_ham_words"]
spam_word_counts  = model["word_counts"]["spam"]
ham_word_counts   = model["word_counts"]["ham"]
vocab             = set(spam_word_counts) | set(ham_word_counts)
vocab_size        = len(vocab)
threshold         = model["best_threshold"]
LAPLACE_ALPHA     = model["hyperparams"]["laplace_alpha"]
TOP_N_WORDS       = model["hyperparams"]["top_n_words"]
STRUCTURAL_WEIGHT = model["hyperparams"].get("structural_weight", 3.0)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
punct_set  = set(string.punctuation)

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

def preprocess(text: str) -> list:
    structural = extract_structural_features(text)
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t not in punct_set]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    bigrams = [f"__{tokens[i]}_{tokens[i+1]}__" for i in range(len(tokens) - 1)]
    return structural + tokens + bigrams

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

def classify(tokens: list) -> tuple:
    known   = [w for w in tokens if w in vocab]
    unknown = [w for w in tokens if w not in vocab]

    word_log_odds = {w: log_odds_given_word(w) for w in known}
    for w in unknown:
        word_log_odds[w] = 0.0  # desconocidos = neutral

    top_n = sorted(word_log_odds.items(), key=lambda x: abs(x[1]), reverse=True)[:TOP_N_WORDS]

    prior_log_odds = math.log(p_spam / p_ham)
    total_log_odds = prior_log_odds + sum(lo for _, lo in top_n)
    prob  = 1.0 / (1.0 + math.exp(-total_log_odds))
    label = "spam" if prob >= threshold else "ham"

    word_probs = {w: p_spam_given_word(w) for w in known
                  if not (w.startswith("__") and w.endswith("__"))}
    top3 = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:3]

    return prob, label, top3

# ── barra de probabilidad visual ─────────────────────────────────────────────
def prob_bar(prob, width=30):
    filled = int(prob * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {prob*100:.1f}%"

# ── loop interactivo ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Clasificador SPAM/HAM  |  CC3085 IA")
print(f"{'='*60}")
print(f"  Threshold : {threshold}  |  P(spam) prior = {p_spam:.4f}")
print(f"  Vocab     : {vocab_size} tokens  |  TOP_N = {TOP_N_WORDS}")
print(f"  Escribe 'salir' para terminar\n")

while True:
    try:
        text = input("Mensaje: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not text or text.lower() in ("salir", "exit", "quit"):
        break

    tokens         = preprocess(text)
    prob, label, top3 = classify(tokens)
    known_count    = sum(1 for t in tokens if t in vocab)
    unknown_count  = len(tokens) - known_count

    print(f"\n {label.upper()}")
    print(f"  {prob_bar(prob)}")
    print(f"  Tokens: {len(tokens)} total  ({known_count} conocidos, {unknown_count} nuevos)")
    if top3:
        print(f"  Top 3 palabras predictoras de SPAM:")
        for i, (word, score) in enumerate(top3, 1):
            print(f"    {i}. '{word}'  →  P(S|W) = {score:.4f}")
    print()