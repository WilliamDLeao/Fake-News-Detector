from fingerprintgenerator import FingerprintGenerator
from news_normalizer import NewsNormalizer
from graph_exporter import GraphExporter
from collections import defaultdict
from itertools import combinations 
import os
import pandas as pd
import random 
import logging
import time
import tracemalloc
import psutil
import gc
from constant import PORCENTAGEM_ANALISADA 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_analysis_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_memory_usage(stage_name):
    """Log do uso de memória"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logger.info(f"🧠 {stage_name} - Memória: {memory_mb:.2f} MB")
    return memory_mb

def remove_duplicates(texts, filenames, labels, dates):
    """Remove notícias duplicadas baseadas no conteúdo do texto"""
    logger.info("🔍 Verificando duplicatas...")
    
    # Usar um dicionário para rastrear textos únicos
    unique_texts = {}
    duplicates_count = 0
    
    for i, text in enumerate(texts):
        # Normalizar o texto para comparação (remover espaços extras, etc)
        normalized = ' '.join(text.strip().split())
        
        if normalized in unique_texts:
            duplicates_count += 1
            logger.debug(f"   Duplicata encontrada: {filenames[i]} → igual a {unique_texts[normalized]}")
        else:
            unique_texts[normalized] = filenames[i]
    
    if duplicates_count > 0:
        logger.info(f"🚫 Encontradas {duplicates_count} notícias duplicadas")
        
        # Reconstruir as listas sem duplicatas
        unique_indices = {}
        for i, text in enumerate(texts):
            normalized = ' '.join(text.strip().split())
            if normalized not in unique_indices:
                unique_indices[normalized] = i
        
        # Criar novas listas apenas com itens únicos
        unique_texts_list = [texts[i] for i in unique_indices.values()]
        unique_filenames = [filenames[i] for i in unique_indices.values()]
        unique_labels = [labels[i] for i in unique_indices.values()]
        unique_dates = [dates[i] for i in unique_indices.values()]
        
        logger.info(f"✅ Mantidas {len(unique_texts_list)} notícias únicas de {len(texts)} originais")
        return unique_texts_list, unique_filenames, unique_labels, unique_dates
    else:
        logger.info("✅ Nenhuma duplicata encontrada")
        return texts, filenames, labels, dates

def load_news_from_csv(csv_path, sample_frac=PORCENTAGEM_ANALISADA):
    """Carrega notícias de um CSV com logging detalhado"""
    try:
        logger.info(f"📖 Lendo {csv_path}...")
        
        if not os.path.exists(csv_path):
            logger.error(f"❌ Arquivo não encontrado: {csv_path}")
            return [], [], [], []
        
        file_size = os.path.getsize(csv_path) / 1024 / 1024
        logger.info(f"   Tamanho do arquivo: {file_size:.2f} MB")
        
        df = pd.read_csv(csv_path)
        logger.info(f"   DataFrame carregado: {len(df)} linhas, {len(df.columns)} colunas")
        logger.info(f"   Colunas disponíveis: {list(df.columns)}")
        
        # VERIFICAR DUPLICATAS NO CSV ORIGINAL
        original_duplicates = df.duplicated(subset=['text']).sum()
        if original_duplicates > 0:
            logger.warning(f"⚠️ CSV contém {original_duplicates} textos duplicados originalmente")
        
        original_size = len(df)
        sampled_df = df.sample(frac=sample_frac, random_state=42)
        sampled_size = len(sampled_df)
        logger.info(f"   Amostra: {sampled_size}/{original_size} ({sampled_size/original_size*100:.1f}%)")
        
        texts = []
        filenames = []
        labels = []
        dates = []
        
        error_count = 0
        for idx, row in sampled_df.iterrows():
            try:
                # Extrair dados com tratamento robusto
                title = str(row['title']) if pd.notna(row.get('title')) else ""
                text_content = str(row['text']) if pd.notna(row.get('text')) else ""
                date_value = str(row['date']) if pd.notna(row.get('date')) else ""
                
                combined_text = f"{title} {text_content}".strip()
                
                # Verificar se temos conteúdo válido
                if not combined_text or combined_text.isspace():
                    logger.warning(f"⚠️ Linha {idx} sem conteúdo textual")
                    error_count += 1
                    continue
                
                texts.append(combined_text)
                filenames.append(f"{os.path.basename(csv_path)}_{idx}")
                labels.append(1 if "Fake" in csv_path else 0)
                dates.append(date_value)
                
            except Exception as e:
                error_count += 1
                logger.warning(f"⚠️ Erro ao processar linha {idx}: {e}")
                continue
        
        logger.info(f"   ✅ Carregadas {len(texts)} notícias de {csv_path}")
        logger.info(f"   ❌ Erros no processamento: {error_count}")
        return texts, filenames, labels, dates
        
    except Exception as e:
        logger.error(f"❌ Erro crítico ao carregar {csv_path}: {e}", exc_info=True)
        return [], [], [], []

def main():
    """Função principal com logging completo"""
    start_time = time.time()
    tracemalloc.start()
    
    try:
        logger.info("🚀 INICIANDO ANÁLISE DE NOTÍCIAS")
        initial_memory = log_memory_usage("Início do programa")
        
        true_path = "True.csv"
        fake_path = "Fake.csv"

        # Verificar se os arquivos existem
        logger.info("🔍 Verificando arquivos...")
        if not os.path.exists(true_path):
            logger.error(f"❌ Arquivo {true_path} não encontrado!")
            return
        if not os.path.exists(fake_path):
            logger.error(f"❌ Arquivo {fake_path} não encontrado!")
            return

        logger.info("✅ Arquivos encontrados!")

        # Carregar dados
        logger.info("\n📂 Carregando notícias...")
        load_start = time.time()
        
        true_texts, true_filenames, true_labels, true_dates = load_news_from_csv(true_path, PORCENTAGEM_ANALISADA)
        fake_texts, fake_filenames, fake_labels, fake_dates = load_news_from_csv(fake_path, PORCENTAGEM_ANALISADA)

        # Combinar todos os dados
        all_texts = true_texts + fake_texts
        all_filenames = true_filenames + fake_filenames
        all_labels = true_labels + fake_labels
        all_dates = true_dates + fake_dates

        logger.info(f"📊 Total de arquivos carregados: {len(all_texts)}")
        logger.info(f"   - Notícias TRUE: {len(true_texts)}")
        logger.info(f"   - Notícias FAKE: {len(fake_texts)}")

        # REMOVER DUPLICATAS
        logger.info("\n🧹 Removendo duplicatas...")
        unique_texts, unique_filenames, unique_labels, unique_dates = remove_duplicates(
            all_texts, all_filenames, all_labels, all_dates
        )

        texts = unique_texts
        filenames = unique_filenames
        labels = unique_labels
        dates = unique_dates

        load_time = time.time() - load_start
        logger.info(f"✅ Dados únicos preparados em {load_time:.2f}s")
        logger.info(f"📈 Estatísticas após remoção de duplicatas:")
        logger.info(f"   - Total original: {len(all_texts)}")
        logger.info(f"   - Total único: {len(texts)}")
        logger.info(f"   - Duplicatas removidas: {len(all_texts) - len(texts)}")
        
        log_memory_usage("Após carregar e limpar dados")

        if len(texts) == 0:
            logger.error("❌ Nenhuma notícia carregada.")
            return

        # Normalização
        logger.info("\n🔄 Normalizando textos...")
        normalize_start = time.time()
        
        normalizer = NewsNormalizer()
        normalized_texts = []

        for i, (text, label) in enumerate(zip(texts, labels)):
            if i % 1000 == 0 and i > 0:
                logger.info(f"   Normalizados {i}/{len(texts)} textos...")
            try:
                # Usar label para determinar se é true news (label == 0)
                is_true_news = (label == 0)
                normalized_text = normalizer.normalize_news(text, is_true_news)
                normalized_texts.append(normalized_text)
            except Exception as e:
                logger.warning(f"⚠️ Erro ao normalizar texto {i}: {e}")
                normalized_texts.append("")  # Fallback
        
        normalize_time = time.time() - normalize_start
        logger.info(f"✅ Textos normalizados em {normalize_time:.2f}s")
        log_memory_usage("Após normalização")

        # Gerar fingerprints
        logger.info("🔑 Gerando fingerprints...")
        fingerprint_start = time.time()
        
        fp_gen = FingerprintGenerator(hash_sizes=[64])
        fingerprints = []
        for i, text in enumerate(normalized_texts):
            if i % 1000 == 0 and i > 0:
                logger.info(f"   Geradas {i}/{len(normalized_texts)} fingerprints...")
            try:
                fingerprint = fp_gen.generate_simhash(text, 64)
                fingerprints.append(fingerprint)
            except Exception as e:
                logger.warning(f"⚠️ Erro ao gerar fingerprint {i}: {e}")
                fingerprints.append(0)  # Fallback
        
        fingerprint_time = time.time() - fingerprint_start
        logger.info(f"✅ Fingerprints geradas em {fingerprint_time:.2f}s")
        log_memory_usage("Após gerar fingerprints")

        # --- SEÇÃO DO GRAFO ---
        logger.info("\n🕸️ Criando grafo de similaridade...")
        graph_start = time.time()
        
        graph_exporter = GraphExporter(log_level=logging.INFO)
        
        # Opção K-NN (mais rápida e eficiente)
        edges_count = graph_exporter.create_similarity_graph_knn(
            texts=texts,
            fingerprints=fingerprints,
            labels=labels,
            filenames=filenames,
            dates=dates,
            k_neighbors=15,  # Conexões por nó
            threshold=20     # Distância máxima
        )
        
        graph_time = time.time() - graph_start
        logger.info(f"✅ Grafo criado em {graph_time:.2f}s")
        log_memory_usage("Após criar grafo")

        # Exportar para Gephi
        logger.info("💾 Exportando para Gephi...")
        export_start = time.time()
        
        nodes_count, edges_count = graph_exporter.export_for_gephi("gephi_news_network")
        
        export_time = time.time() - export_start
        logger.info(f"✅ Exportação concluída em {export_time:.2f}s")

       
        # --- ESTATÍSTICAS FINAIS ---
        total_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        
        logger.info(f"\n🎉 PROCESSAMENTO CONCLUÍDO")
        logger.info(f"📊 ESTATÍSTICAS GERAIS:")
        logger.info(f"   Tempo total: {total_time:.2f}s")
        logger.info(f"   Memória pico: {peak / 1024 / 1024:.2f} MB")
        
        total_files = len(texts)
        fake_count = sum(labels)
        true_count = total_files - fake_count
        
        logger.info(f"   Notícias analisadas: {total_files}")
        logger.info(f"   Notícias TRUE: {true_count} ({true_count/total_files*100:.2f}%)")
        logger.info(f"   Notícias FAKE: {fake_count} ({fake_count/total_files*100:.2f}%)")
        logger.info(f"   Nós no grafo: {nodes_count}")
        logger.info(f"   Arestas no grafo: {edges_count}")
        
        if nodes_count > 1:
            density = edges_count / (nodes_count * (nodes_count - 1) / 2)
            logger.info(f"   Densidade do grafo: {density:.6f}")
        
        logger.info(f"\n⏱️ TEMPOS DAS ETAPAS:")
        logger.info(f"   Carregamento: {load_time:.2f}s")
        logger.info(f"   Normalização: {normalize_time:.2f}s")
        logger.info(f"   Fingerprints: {fingerprint_time:.2f}s")
        logger.info(f"   Criação do grafo: {graph_time:.2f}s")
        logger.info(f"   Exportação: {export_time:.2f}s")
        
        logger.info(f"\n💾 Dados exportados para: gephi_news_network/")
        logger.info(f"📋 Logs salvos em: news_analysis_debug.log")

    except Exception as e:
        logger.critical(f"💥 ERRO CRÍTICO no programa principal: {e}", exc_info=True)
        raise
        
    finally:
        tracemalloc.stop()
        # Limpeza final
        gc.collect()
        final_memory = log_memory_usage("Final do programa")
        logger.info("✅ Programa finalizado")

if __name__ == "__main__":
    main()