from fingerprintgenerator import FingerprintGenerator
from news_normalizer import NewsNormalizer
from graph_exporter import GraphExporter
from collections import defaultdict
from itertools import combinations 
import os
import pandas as pd
import random 
import time
import tracemalloc
import psutil
import gc
from constant import PORCENTAGEM_ANALISADA 

def remove_duplicates(texts, filenames, labels, dates):
    """Remove notícias duplicadas baseadas no conteúdo do texto"""
    
    unique_texts = {}
    duplicates_count = 0
    
    for i, text in enumerate(texts):
        normalized = ' '.join(text.strip().split())
        
        if normalized in unique_texts:
            duplicates_count += 1
        else:
            unique_texts[normalized] = filenames[i]
    
    if duplicates_count > 0:
        unique_indices = {}
        for i, text in enumerate(texts):
            normalized = ' '.join(text.strip().split())
            if normalized not in unique_indices:
                unique_indices[normalized] = i
        
        unique_texts_list = [texts[i] for i in unique_indices.values()]
        unique_filenames = [filenames[i] for i in unique_indices.values()]
        unique_labels = [labels[i] for i in unique_indices.values()]
        unique_dates = [dates[i] for i in unique_indices.values()]
        
       
        return unique_texts_list, unique_filenames, unique_labels, unique_dates
    
    return texts, filenames, labels, dates

def load_news_from_csv(csv_path, sample_frac=PORCENTAGEM_ANALISADA):
    """Carrega notícias de um CSV"""
    try:
        if not os.path.exists(csv_path):
            return [], [], [], []
        
        df = pd.read_csv(csv_path)
        original_size = len(df)
        sampled_df = df.sample(frac=sample_frac, random_state=42)
        sampled_size = len(sampled_df)
        
        texts = []
        filenames = []
        labels = []
        dates = []
        
        error_count = 0
        for idx, row in sampled_df.iterrows():
            try:
                title = str(row['title']) if pd.notna(row.get('title')) else ""
                text_content = str(row['text']) if pd.notna(row.get('text')) else ""
                date_value = str(row['date']) if pd.notna(row.get('date')) else ""
                
                combined_text = f"{title} {text_content}".strip()
                
                if not combined_text or combined_text.isspace():
                    error_count += 1
                    continue
                
                texts.append(combined_text)
                filenames.append(f"{os.path.basename(csv_path)}_{idx}")
                labels.append(1 if "Fake" in csv_path else 0)
                dates.append(date_value)
                
            except Exception:
                error_count += 1
                continue
        
        return texts, filenames, labels, dates
        
    except Exception:
        return [], [], [], []

def main():
    """Função principal"""
    print("🚀 INICIANDO ANÁLISE DE NOTÍCIAS")
    start_time = time.time()
    tracemalloc.start()
    
    try:
        true_path = "True.csv"
        fake_path = "Fake.csv"

        # Verificar se os arquivos existem
        if not os.path.exists(true_path):
            print(f"❌ Arquivo {true_path} não encontrado!")
            return
        if not os.path.exists(fake_path):
            print(f"❌ Arquivo {fake_path} não encontrado!")
            return

        # Carregar dados
        print("\n📂 Carregando notícias...")
        load_start = time.time()
        
        true_texts, true_filenames, true_labels, true_dates = load_news_from_csv(true_path, PORCENTAGEM_ANALISADA)
        fake_texts, fake_filenames, fake_labels, fake_dates = load_news_from_csv(fake_path, PORCENTAGEM_ANALISADA)

        # Combinar todos os dados
        all_texts = true_texts + fake_texts
        all_filenames = true_filenames + fake_filenames
        all_labels = true_labels + fake_labels
        all_dates = true_dates + fake_dates

        print(f"📊 Total de arquivos carregados: {len(all_texts)}")

        # REMOVER DUPLICATAS
        print("\n🧹 Removendo duplicatas...")
        unique_texts, unique_filenames, unique_labels, unique_dates = remove_duplicates(
            all_texts, all_filenames, all_labels, all_dates
        )

        texts = unique_texts
        filenames = unique_filenames
        labels = unique_labels
        dates = unique_dates

        load_time = time.time() - load_start
        print(f"✅ Dados preparados.")

        if len(texts) == 0:
            print("❌ Nenhuma notícia carregada.")
            return

        # Normalização
        print("\n🔄 Normalizando textos...")
        normalize_start = time.time()
        
        normalizer = NewsNormalizer()
        normalized_texts = []

        for i, (text, label) in enumerate(zip(texts, labels)):
            try:
                is_true_news = (label == 0)
                normalized_text = normalizer.normalize_news(text, is_true_news)
                normalized_texts.append(normalized_text)
            except Exception:
                normalized_texts.append("")
        
        normalize_time = time.time() - normalize_start
        print(f"✅ Textos normalizados.")

        # Gerar fingerprints
        print("\n🔑 Gerando fingerprints...")
        fingerprint_start = time.time()
        
        fp_gen = FingerprintGenerator(hash_sizes=[64])
        fingerprints = []
        for i, text in enumerate(normalized_texts):
            try:
                fingerprint = fp_gen.generate_simhash(text, 64)
                fingerprints.append(fingerprint)
            except Exception:
                fingerprints.append(0)
        
        fingerprint_time = time.time() - fingerprint_start
        print(f"✅ Fingerprints geradas.")

        # --- SEÇÃO DO GRAFO ---
        print("\n🕸️ Criando grafo de similaridade...")
        graph_start = time.time()
        
        graph_exporter = GraphExporter()
        
        edges_count = graph_exporter.create_similarity_graph_knn(
            texts=texts,
            fingerprints=fingerprints,
            labels=labels,
            filenames=filenames,
            dates=dates,
            k_neighbors=15,
            threshold=20
        )
        
        graph_time = time.time() - graph_start

        # Exportar para Gephi
        print("💾 Exportando para Gephi...")
        export_start = time.time()
        
        nodes_count, edges_count = graph_exporter.export_for_gephi("gephi_news_network")
        
        export_time = time.time() - export_start

        # --- ESTATÍSTICAS FINAIS ---
        total_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        
        print(f"\n🎉 PROCESSAMENTO CONCLUÍDO")
           
        total_files = len(texts)
        fake_count = sum(labels)
        true_count = total_files - fake_count
             
        if nodes_count > 1:
            density = edges_count / (nodes_count * (nodes_count - 1) / 2)
        
        print(f"\n💾 Dados exportados para: gephi_news_network/")

    except Exception as e:
        print(f"💥 ERRO CRÍTICO no programa principal: {e}")
        raise
        
    finally:
        tracemalloc.stop()
        gc.collect()
        print("✅ Programa finalizado")

if __name__ == "__main__":
    main()
