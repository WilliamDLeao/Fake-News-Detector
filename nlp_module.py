import pickle
import numpy as np
from fingerprintgenerator import FingerprintGenerator
from news_normalizer import NewsNormalizer
from scipy.spatial.distance import hamming
import warnings

# Suprimir warnings de versionamento do scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Carregar classificador treinado
try:
    with open("trained_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    print("✅ Modelo carregado com sucesso")
except FileNotFoundError:
    classifier = None
    print("⚠️ Modelo não encontrado. Execute train_classifier.py primeiro.")
except Exception as e:
    classifier = None
    print(f"⚠️ Erro ao carregar modelo: {e}")

normalizer = NewsNormalizer()
fg = FingerprintGenerator(hash_sizes=[128])
BITS = 128  # tamanho esperado

def _is_bits_vector(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            if len(x) == BITS:
                return True
        except Exception:
            return False
    return False

def _to_bits_from_code(text):
    code = normalizer.normalize_news(text)
    fp = fg.generate_simhash(code, BITS)
    bits = list(map(int, bin(fp)[2:].zfill(BITS)))
    return bits

def run_nlp(text_or_bits):
    if classifier is None:
        # Fallback se o modelo não estiver treinado
        bits = _to_bits_from_code(text_or_bits) if not _is_bits_vector(text_or_bits) else text_or_bits
        return 0, 0.5, True  # Retorna como verdadeira com confiança neutra
    
    if _is_bits_vector(text_or_bits):
        bits = np.array(text_or_bits, dtype=int)
    else:
        bits = np.array(_to_bits_from_code(text_or_bits), dtype=int)

    # Predição pela MLP
    try:
        proba = classifier.predict_proba([bits])[0]
        pred = int(classifier.predict([bits])[0])
        confidence = float(max(proba))
        used_hamming = False

        # Fallback se confiança baixa
        if confidence < 0.69 and hasattr(classifier, "_centroids"):
            c_true = classifier._centroids["true"]
            c_fake = classifier._centroids["fake"]

            d_true = hamming(bits, c_true)
            d_fake = hamming(bits, c_fake)

            pred = 0 if d_true < d_fake else 1
            confidence = float(max(proba))  # mantém confiança calculada pela MLP
            used_hamming = True

        return pred, confidence, used_hamming
    except Exception as e:
        print(f"⚠️ Erro na classificação: {e}")
        return 0, 0.5, True

# Utilitários para inspeção
def get_model_info():
    if classifier is None:
        return {"error": "Modelo não carregado"}
        
    info = {}
    info["type"] = type(classifier).__name__
    try:
        params = classifier.get_params()
        info["params"] = params
    except Exception:
        info["params"] = None

    if hasattr(classifier, "hidden_layer_sizes"):
        info["hidden_layer_sizes"] = classifier.hidden_layer_sizes
    if hasattr(classifier, "coefs_"):
        n_layers = len(classifier.coefs_)
        info["n_layers_including_io"] = n_layers + 1
        total_params = sum(w.size for w in classifier.coefs_) + sum(b.size for b in getattr(classifier, "intercepts_", []))
        info["total_parameters"] = int(total_params)

    info["training_info"] = getattr(classifier, "_training_info", None)
    info["centroids"] = getattr(classifier, "_centroids", None)
    return info

def print_model_summary():
    info = get_model_info()
    print("=== Modelo salvo ===")
    if "error" in info:
        print(info["error"])
        return
        
    print("Tipo:", info.get("type"))
    if info.get("hidden_layer_sizes") is not None:
        print("Hidden layer sizes:", info["hidden_layer_sizes"])
    if info.get("params") is not None:
        useful = {k: info["params"].get(k) for k in ("activation", "solver", "max_iter", "random_state")}
        print("Params (subset):", useful)
    if info.get("total_parameters") is not None:
        print("Total params:", info["total_parameters"])
    if info.get("training_info") is not None:
        print("Training info:", info["training_info"])
    if info.get("centroids") is not None:
        print("Centroids salvos:", {k: v[:10] for k, v in info["centroids"].items()})  # mostra só 10 bits iniciais