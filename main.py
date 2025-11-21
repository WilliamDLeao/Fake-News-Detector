from fingerprintgenerator import FingerprintGenerator
from nlp_module import run_nlp
from news_normalizer import NewsNormalizer
from graph_exporter import GraphExporter
from collections import defaultdict
from itertools import combinations  # ADICIONE ESTA IMPORTACAO
import os
import pandas as pd
import random  # ADICIONE TAMBÉM O RANDOM

def load_news_from_csv(csv_path, sample_frac=0.2):
    """Carrega notícias de um CSV"""
    try:
        print(f"📖 Lendo {csv_path}...")
        df = pd.read_csv(csv_path)
        sampled_df = df.sample(frac=sample_frac, random_state=42)
        
        texts = []
        filenames = []
        labels = []
        for idx, row in sampled_df.iterrows():
            combined_text = f"{row['title']} {row['text']}"
            texts.append(combined_text)
            filenames.append(f"{os.path.basename(csv_path)}_{idx}")
            labels.append(1 if "Fake" in csv_path else 0)
        
        print(f"   ✅ Carregadas {len(texts)} notícias de {csv_path}")
        return texts, filenames, labels
    except Exception as e:
        print(f"⚠️ Erro ao carregar {csv_path}: {e}")
        return [], [], []

def main():
    true_path = "True.csv"
    fake_path = "Fake.csv"

    # Verificar se os arquivos existem
    print("🔍 Verificando arquivos...")
    if not os.path.exists(true_path):
        print(f"❌ Arquivo {true_path} não encontrado!")
        return
    if not os.path.exists(fake_path):
        print(f"❌ Arquivo {fake_path} não encontrado!")
        return

    print("✅ Arquivos encontrados!")

    # Carregar dados
    print("\n📂 Carregando notícias...")
    true_texts, true_filenames, true_labels = load_news_from_csv(true_path, 0.2)
    fake_texts, fake_filenames, fake_labels = load_news_from_csv(fake_path, 0.2)

    texts = true_texts + fake_texts
    filenames = true_filenames + fake_filenames
    labels = true_labels + fake_labels

    print(f"\n📊 Total de arquivos carregados: {len(texts)}")

    if len(texts) == 0:
        print("❌ Nenhuma notícia carregada.")
        return

    # Normalização
    print("\n🔄 Normalizando textos...")
    normalizer = NewsNormalizer()
    normalized_texts = [normalizer.normalize_news(t) for t in texts]
    
    # Gerar fingerprints
    print("🔑 Gerando fingerprints...")
    fp_gen = FingerprintGenerator(hash_sizes=[64])
    fingerprints = [fp_gen.generate_simhash(t, 64) for t in normalized_texts]
    
    # --- SEÇÃO OTIMIZADA: CRIAR GRAFOS ---
    print("\n🕸️ Criando grafo de similaridade (OTIMIZADO)...")
    graph_exporter = GraphExporter()
    
    # Opção 1: K-NN (MAIS RÁPIDA E EFICIENTE) - RECOMENDADA
    edges_count = graph_exporter.create_similarity_graph_knn(
        texts=texts,
        fingerprints=fingerprints,
        labels=labels,
        filenames=filenames,
        k_neighbors=15,  # Conexões por nó
        threshold=20     # Distância máxima
    )
    
    # Exportar para Gephi
    nodes_count, edges_count = graph_exporter.export_for_gephi("gephi_news_network")
    
    print(f"📊 Estatísticas do Grafo:")
    print(f"   - Nós (notícias): {nodes_count}")
    print(f"   - Arestas (similaridades): {edges_count}")
    if nodes_count > 1:
        density = edges_count / (nodes_count * (nodes_count - 1) / 2)
        print(f"   - Densidade: {density:.6f}")
    else:
        print(f"   - Densidade: N/A (apenas 1 nó)")
    
    # --- FIM DA SEÇÃO DE GRAFOS ---
    
    # Cálculo de distâncias (opcional, mas mais leve)
    print("\n📏 Calculando distâncias de Hamming (amostra)...")
    sample_size = min(100, len(fingerprints))
    if len(fingerprints) > 1:
        # Correção: usar combinations diretamente já que foi importado
        sample_pairs = random.sample(list(combinations(range(len(fingerprints)), 2)), sample_size)
        distances = [bin(fingerprints[i] ^ fingerprints[j]).count("1") for i, j in sample_pairs]
        avg_distance = sum(distances) / len(distances)
        print(f"   Distância média de Hamming (amostra de {sample_size} pares): {avg_distance:.2f}")
    else:
        print("   Não há fingerprints suficientes para calcular distâncias")

    # Classificação
    print("\n🔍 Classificando notícias...")
    nlp_results = [run_nlp(t) for t in texts]

    # Agrupar resultados por tipo (True/Fake)
    results_by_type = defaultdict(list)
    for filename, (pred, conf, used_hamming) in zip(filenames, nlp_results):
        file_type = "True" if "True" in filename else "Fake"
        results_by_type[file_type].append((filename, pred, conf, used_hamming))

    # Determinar tamanho máximo do nome do arquivo
    max_len = max((len(file) for file in filenames), default=0)

    # Exibir resultados
    print("\n📊 RESULTADOS DA CLASSIFICAÇÃO:")
    for file_type, files in results_by_type.items():
        print(f"\n📂 {file_type} News")
        for file, pred, conf, used_hamming in files:
            symbol = "❌" if pred == 1 else "✅"
            pred_text = "FAKE" if pred == 1 else "TRUE"
            
            if used_hamming:
                print(f"\t📄 {file:<{max_len}} --> {symbol} {pred_text} *H")
            else:
                print(f"\t📄 {file:<{max_len}} --> {symbol} {pred_text} (Confiança: {conf:6.2%})")

    # Estatísticas gerais
    total_files = len(texts)
    fake_count = sum(1 for pred, _, _ in nlp_results if pred == 1)
    true_count = total_files - fake_count
    
    print(f"\n📈 ESTATÍSTICAS GERAIS:")
    print(f"Total de notícias analisadas: {total_files}")
    print(f"Notícias classificadas como TRUE: {true_count} ({true_count/total_files*100:.2f}%)")
    print(f"Notícias classificadas como FAKE: {fake_count} ({fake_count/total_files*100:.2f}%)")

    # Verificar acurácia básica
    correct_predictions = 0
    for filename, (pred, _, _) in zip(filenames, nlp_results):
        true_label = 1 if "Fake" in filename else 0
        if pred == true_label:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_files if total_files > 0 else 0
    print(f"🎯 Acurácia geral: {accuracy:.2%}")

if __name__ == "__main__":
    main()