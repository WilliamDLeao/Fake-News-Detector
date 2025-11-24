import os
import pickle
import numpy as np
import pandas as pd
from fingerprintgenerator import FingerprintGenerator
from news_normalizer import NewsNormalizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from constant import PORCENTAGEM_ANALISADA 

def load_news_from_csv(csv_path, label, sample_frac=PORCENTAGEM_ANALISADA):
    """Carrega apenas uma fração dos dados para maior velocidade"""
    try:
        print(f"📖 Lendo {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"   Encontradas {len(df)} notícias no total")
        
        sampled_df = df.sample(frac=sample_frac, random_state=42)
        print(f"   Amostrando {len(sampled_df)} notícias ({sample_frac*100}%)")
        
        texts = []
        for _, row in sampled_df.iterrows():
            # Combina título e texto para análise mais robusta
            combined_text = f"{row['title']} {row['text']}"
            texts.append(combined_text)
        
        labels = [label] * len(texts)
        return texts, labels
    except Exception as e:
        print(f"⚠️ Erro ao carregar {csv_path}: {e}")
        return [], []

def main():
    true_path = "True.csv"
    fake_path = "Fake.csv"

    # Verificar se os arquivos existem
    print("🔍 Verificando arquivos...")
    if not os.path.exists(true_path):
        print(f"❌ Arquivo {true_path} não encontrado!")
        print("   Certifique-se de que True.csv está na pasta raiz")
        return
    if not os.path.exists(fake_path):
        print(f"❌ Arquivo {fake_path} não encontrado!")
        print("   Certifique-se de que Fake.csv está na pasta raiz")
        return

    print("✅ Arquivos encontrados!")

    # Carregar apenas X% de cada dataset
    print("\n📂 Carregando dados...")
    true_texts, true_labels = load_news_from_csv(true_path, 0, PORCENTAGEM_ANALISADA )
    fake_texts, fake_labels = load_news_from_csv(fake_path, 1, PORCENTAGEM_ANALISADA)

    texts = true_texts + fake_texts
    labels = true_labels + fake_labels

    print(f"\n📊 Estatísticas dos dados:")
    print(f"   Total de notícias: {len(texts)}")
    print(f"   ✅ Verdadeiras: {len(true_texts)}")
    print(f"   ❌ Falsas: {len(fake_texts)}")

    if len(texts) == 0:
        print("❌ Nenhuma notícia encontrada. Verifique:")
        print("   - Os arquivos CSV existem")
        print("   - Os arquivos não estão vazios")
        print("   - Os arquivos têm colunas 'title' e 'text'")
        return

    # Verificar estrutura dos dados
    print("\n🔍 Verificando amostra dos dados...")
    if len(texts) > 0:
        print(f"   Primeira notícia (pré-normalização): {texts[0][:100]}...")

    # Normalização
    print("\n🔄 Normalizando textos...")
    normalizer = NewsNormalizer()
    normalized = [normalizer.normalize_news(t) for t in texts]

    if len(normalized) > 0:
        print(f"   Primeira notícia (pós-normalização): {normalized[0][:100]}...")

    # Fingerprints (128 bits)
    print("\n🔑 Gerando fingerprints...")
    fg = FingerprintGenerator(hash_sizes=[128])
    fingerprints = [fg.generate_simhash(code, 128) for code in normalized]

    print(f"   ✅ Fingerprints geradas: {len(fingerprints)}")

    # Converter para vetor de bits (float)
    X = np.array([list(map(int, bin(fp)[2:].zfill(128))) for fp in fingerprints], dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print(f"   Shape dos dados: {X.shape}")

    # Divisão treino/teste (80/20)
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    print(f"\n📈 Divisão treino/teste:")
    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Teste: {X_test.shape[0]} amostras")

    # Verificar se temos dados suficientes
    if X_train.shape[0] < 10:
        print("❌ Dados insuficientes para treinamento.")
        return

    # ----- Configuração da MLP
    print("\n🧠 Configurando rede neural...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True
    )

    # Treinar
    print("⏳ Treinando MLPClassifier...")
    clf.fit(X_train, y_train)
    print("✅ Treinamento concluído.")

    # Avaliação
    print("\n📊 Avaliando modelo...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy (test): {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Guardar info de treino
    clf._training_info = {
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "train_fraction": float(X_train.shape[0] / (X_train.shape[0] + X_test.shape[0])),
        "test_accuracy": float(acc),
        "n_true_news": len(true_texts),
        "n_fake_news": len(fake_texts)
    }

    # ---- Calcular centróides por classe (para fallback Hamming)
    print("\n📐 Calculando centróides...")
    X_true = X[y == 0]
    X_fake = X[y == 1]

    centroid_true = np.round(X_true.mean(axis=0)).astype(int) if len(X_true) > 0 else np.zeros(X.shape[1], dtype=int)
    centroid_fake = np.round(X_fake.mean(axis=0)).astype(int) if len(X_fake) > 0 else np.ones(X.shape[1], dtype=int)

    clf._centroids = {
        "true": centroid_true,
        "fake": centroid_fake
    }

    # Salvar modelo
    print("\n💾 Salvando modelo...")
    with open("trained_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("✅ Classificador (MLP) + centróides salvos em trained_classifier.pkl")
    
    # Mostrar resumo do modelo
    print("\n📋 Resumo do modelo treinado:")
    from nlp_module import print_model_summary
    print_model_summary()

if __name__ == "__main__":
    main()