import networkx as nx
import pandas as pd
import numpy as np
import os
import logging
import time
import tracemalloc
from collections import defaultdict
from itertools import combinations
import math
import psutil
import gc

class GraphExporter:
    def __init__(self, log_level=logging.INFO):
        self.graph = nx.Graph()
        self.setup_logging(log_level)
        
    def setup_logging(self, log_level):
        """Configura sistema de logging detalhado"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('graph_exporter_debug.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_memory_usage(self, stage_name):
        """Log detalhado do uso de memória"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        self.logger.info(f"🧠 {stage_name} - Memória: {memory_mb:.2f} MB")
        return memory_mb
    
    def log_system_resources(self):
        """Log de recursos do sistema"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        self.logger.debug(f"📊 Sistema - CPU: {cpu_percent}% | Memória: {memory.percent}%")
    
    def create_similarity_graph_optimized(self, texts, fingerprints, labels, filenames, dates, threshold=10, max_edges_per_node=20):
        """
        Versão otimizada que evita comparação O(n²) com logging detalhado
        """
        start_time = time.time()
        tracemalloc.start()
        
        try:
            n = len(fingerprints)
            self.logger.info(f"🚀 INICIANDO create_similarity_graph_optimized")
            self.logger.info(f"📦 Parâmetros - Nós: {n}, Threshold: {threshold}, Max edges por nó: {max_edges_per_node}")
            
            initial_memory = self.log_memory_usage("Início")
            self.log_system_resources()
            
            # Verificação de dados de entrada
            if len(dates) != n:
                self.logger.error(f"❌ Inconsistência: dates={len(dates)} != {n}")
                raise ValueError("Array dates deve ter o mesmo tamanho")
            self._validate_input_data(texts, fingerprints, labels, filenames, n)
            
            # 1. Estratégia: Agrupar por prefixos para reduzir comparações
            prefix_length = 8
            self.logger.info(f"🔍 Agrupando {n} fingerprints por prefixo (length: {prefix_length})")
            
            groups = defaultdict(list)
            grouping_start = time.time()
            
            for i, fp in enumerate(fingerprints):
                if i % 10000 == 0 and i > 0:
                    self.logger.info(f"  Processados {i}/{n} fingerprints...")
                
                try:
                    fp_bin = bin(fp)[2:].zfill(64)
                    prefix = fp_bin[:prefix_length]
                    groups[prefix].append(i)
                except Exception as e:
                    self.logger.error(f"❌ Erro ao processar fingerprint {i}: {e}")
                    continue
            
            grouping_time = time.time() - grouping_start
            self.logger.info(f"✅ Agrupamento concluído: {len(groups)} grupos em {grouping_time:.2f}s")
            self.log_memory_usage("Após agrupamento")
            
            # 2. Adicionar nós primeiro
            self.logger.info("🆕 Adicionando nós ao grafo...")
            node_start = time.time()
            
            nodes_added = 0
            for i, (text, fingerprint, label, filename, date) in enumerate(zip(texts, fingerprints, labels, filenames, dates)):
                try:
                    self.graph.add_node(i, 
                                      label=f"News_{i}",
                                      fingerprint=fingerprint,
                                      text_preview=text[:50] + "..." if len(text) > 50 else text,
                                      type="Fake" if label == 1 else "True",
                                      filename=filename,
                                      date=date,
                                      degree=0)
                    nodes_added += 1
                    
                    if nodes_added % 5000 == 0:
                        self.log_memory_usage(f"Adicionados {nodes_added} nós")
                        
                except Exception as e:
                    self.logger.error(f"❌ Erro ao adicionar nó {i}: {e}")
                    continue
            
            node_time = time.time() - node_start
            self.logger.info(f"✅ {nodes_added} nós adicionados em {node_time:.2f}s")
            
            # 3. Comparação inteligente apenas dentro dos grupos e entre grupos vizinhos
            self.logger.info("🔗 Criando arestas...")
            edge_start = time.time()
            
            edges_added = 0
            group_keys = list(groups.keys())
            total_groups = len(group_keys)
            
            for idx, key in enumerate(group_keys):
                if idx % 100 == 0:
                    self.logger.info(f"📊 Processando grupo {idx}/{total_groups} - Arestas totais: {edges_added}")
                    self.log_memory_usage(f"Grupo {idx}")
                
                indices = groups[key]
                
                # Comparação dentro do grupo
                if len(indices) > 1:
                    for i, j in combinations(indices, 2):
                        if edges_added >= n * max_edges_per_node:
                            self.logger.warning(f"⚠️ Limite de arestas atingido: {edges_added}")
                            break
                            
                        try:
                            distance = bin(fingerprints[i] ^ fingerprints[j]).count("1")
                            if distance <= threshold:
                                self.graph.add_edge(i, j, weight=1-distance/threshold, distance=distance)
                                edges_added += 1
                                
                                if edges_added % 1000 == 0:
                                    self.logger.debug(f"  Arestas criadas: {edges_added}")
                                    
                        except Exception as e:
                            self.logger.error(f"❌ Erro ao criar aresta ({i},{j}): {e}")
                            continue
                
                # Comparação com grupos vizinhos
                if idx < len(group_keys) - 1:
                    next_key = group_keys[idx + 1]
                    next_indices = groups[next_key]
                    
                    # Amostrar do grupo atual e próximo grupo
                    sample_current = indices[:min(5, len(indices))]
                    sample_next = next_indices[:min(5, len(next_indices))]
                    
                    for i in sample_current:
                        for j in sample_next:
                            if edges_added >= n * max_edges_per_node:
                                break
                                
                            try:
                                distance = bin(fingerprints[i] ^ fingerprints[j]).count("1")
                                if distance <= threshold:
                                    self.graph.add_edge(i, j, weight=1-distance/threshold, distance=distance)
                                    edges_added += 1
                            except Exception as e:
                                self.logger.error(f"❌ Erro ao criar aresta entre grupos ({i},{j}): {e}")
                                continue
            
            edge_time = time.time() - edge_start
            total_time = time.time() - start_time
            
            # Log final detalhado
            current, peak = tracemalloc.get_traced_memory()
            self.logger.info(f"✅ OPERAÇÃO CONCLUÍDA")
            self.logger.info(f"📊 Estatísticas Finais:")
            self.logger.info(f"   - Tempo total: {total_time:.2f}s")
            self.logger.info(f"   - Nós: {len(self.graph.nodes)}")
            self.logger.info(f"   - Arestas: {edges_added}")
            self.logger.info(f"   - Memória pico: {peak / 1024 / 1024:.2f} MB")
            self.logger.info(f"   - Tempo agrupamento: {grouping_time:.2f}s")
            self.logger.info(f"   - Tempo nós: {node_time:.2f}s")
            self.logger.info(f"   - Tempo arestas: {edge_time:.2f}s")
            
            return edges_added
            
        except Exception as e:
            self.logger.critical(f"💥 ERRO CRÍTICO em create_similarity_graph_optimized: {e}", exc_info=True)
            raise
        finally:
            tracemalloc.stop()
            # Limpeza de memória
            gc.collect()
    
    def _validate_input_data(self, texts, fingerprints, labels, filenames, n):
        """Valida os dados de entrada"""
        self.logger.info("🔍 Validando dados de entrada...")
        
        if len(texts) != n or len(labels) != n or len(filenames) != n:
            self.logger.error(f"❌ Inconsistência nos dados: texts={len(texts)}, labels={len(labels)}, filenames={len(filenames)}")
            raise ValueError("Todos os arrays de entrada devem ter o mesmo tamanho")
        
        # Verificar tipos de dados
        for i, fp in enumerate(fingerprints[:min(10, n)]):  # Amostra
            if not isinstance(fp, (int, np.integer)):
                self.logger.warning(f"⚠️ Fingerprint {i} não é inteiro: {type(fp)}")
        
        self.logger.info("✅ Validação de dados concluída")
    
    def create_similarity_graph_knn(self, texts, fingerprints, labels, filenames, dates, k_neighbors=10, threshold=15):
        """
        Versão K-NN com logging detalhado
        """
        start_time = time.time()
        tracemalloc.start()
        
        try:
            n = len(fingerprints)
            self.logger.info(f"🚀 INICIANDO create_similarity_graph_knn")
            self.logger.info(f"📦 Parâmetros - Nós: {n}, k_neighbors: {k_neighbors}, threshold: {threshold}")
            
            if len(dates) != n:
                self.logger.error(f"❌ Inconsistência: dates={len(dates)} != {n}")
                raise ValueError("Array dates deve ter o mesmo tamanho")
            self._validate_input_data(texts, fingerprints, labels, filenames, n)
            
            # Adicionar todos os nós
            self.logger.info("🆕 Adicionando nós...")
            for i, (text, fingerprint, label, filename, date) in enumerate(zip(texts, fingerprints, labels, filenames, dates)):
                self.graph.add_node(i, 
                                  label=f"News_{i}",
                                  fingerprint=fingerprint,
                                  text_preview=text[:50] + "..." if len(text) > 50 else text,
                                  type="Fake" if label == 1 else "True",
                                  filename=filename,
                                  date=date)
                
                if i % 5000 == 0 and i > 0:
                    self.log_memory_usage(f"Adicionados {i} nós")
            
            edges_added = 0
            self.logger.info("🔍 Calculando vizinhos K-NN...")
            
            for i in range(n):
                if i % 1000 == 0:
                    self.logger.info(f"📊 Processando nó {i}/{n} - Arestas: {edges_added}")
                    self.log_memory_usage(f"Nó {i}")
                
                distances = []
                
                # Calcular distâncias apenas para um subconjunto (amostragem)
                sample_size = min(100, n)
                if n > 100:
                    indices = np.random.choice([x for x in range(n) if x != i], sample_size, replace=False)
                else:
                    indices = [x for x in range(n) if x != i]
                
                for j in indices:
                    try:
                        distance = bin(fingerprints[i] ^ fingerprints[j]).count("1")
                        distances.append((j, distance))
                    except Exception as e:
                        self.logger.error(f"❌ Erro ao calcular distância ({i},{j}): {e}")
                        continue
                
                # Ordenar por distância e pegar os k mais próximos
                distances.sort(key=lambda x: x[1])
                for j, dist in distances[:k_neighbors]:
                    if dist <= threshold:
                        try:
                            self.graph.add_edge(i, j, weight=1-dist/threshold, distance=dist)
                            edges_added += 1
                        except Exception as e:
                            self.logger.error(f"❌ Erro ao adicionar aresta ({i},{j}): {e}")
                            continue
            
            total_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            
            self.logger.info(f"✅ K-NN CONCLUÍDO")
            self.logger.info(f"📊 Estatísticas:")
            self.logger.info(f"   - Tempo: {total_time:.2f}s")
            self.logger.info(f"   - Arestas: {edges_added}")
            self.logger.info(f"   - Memória pico: {peak / 1024 / 1024:.2f} MB")
            
            return edges_added
            
        except Exception as e:
            self.logger.critical(f"💥 ERRO CRÍTICO em create_similarity_graph_knn: {e}", exc_info=True)
            raise
        finally:
            tracemalloc.stop()
            gc.collect()

    def export_for_gephi(self, output_path="gephi_data"):
        """Exporta nodes e edges para CSV compatível com Gephi com logging"""
        start_time = time.time()
        
        try:
            self.logger.info(f"💾 Exportando dados para Gephi em {output_path}")
            os.makedirs(output_path, exist_ok=True)
            
            # Nodes CSV
            self.logger.info("📝 Exportando nós...")
            nodes_data = []
            for node_id, attrs in self.graph.nodes(data=True):
                nodes_data.append({
                    'Id': node_id,
                    'Label': attrs.get('label', ''),
                    'Type': attrs.get('type', ''),
                    'TextPreview': attrs.get('text_preview', ''),
                    'Filename': attrs.get('filename', ''),
                    'Date': attrs.get('date', ''),
                    'Degree': self.graph.degree(node_id)
                })
            
            nodes_df = pd.DataFrame(nodes_data)
            nodes_path = f"{output_path}/nodes.csv"
            nodes_df.to_csv(nodes_path, index=False, encoding='utf-8')
            self.logger.info(f"✅ Nós exportados: {len(nodes_data)}")
            
            # Edges CSV
            self.logger.info("📝 Exportando arestas...")
            edges_data = []
            for edge in self.graph.edges(data=True):
                edges_data.append({
                    'Source': edge[0],
                    'Target': edge[1],
                    'Weight': edge[2].get('weight', 0),
                    'Distance': edge[2].get('distance', 0),
                    'Type': 'Undirected'
                })
            
            edges_df = pd.DataFrame(edges_data)
            edges_path = f"{output_path}/edges.csv"
            edges_df.to_csv(edges_path, index=False)
            self.logger.info(f"✅ Arestas exportadas: {len(edges_data)}")
            
            # Estatísticas
            stats = {
                'total_nodes': len(nodes_data),
                'total_edges': len(edges_data),
                'density': len(edges_data) / (len(nodes_data) * (len(nodes_data) - 1) / 2) if len(nodes_data) > 1 else 0,
                'avg_degree': sum(dict(self.graph.degree()).values()) / len(nodes_data) if len(nodes_data) > 0 else 0
            }
            
            stats_df = pd.DataFrame([stats])
            stats_path = f"{output_path}/graph_stats.csv"
            stats_df.to_csv(stats_path, index=False)
            
            total_time = time.time() - start_time
            
            self.logger.info(f"✅ EXPORTAÇÃO CONCLUÍDA em {total_time:.2f}s")
            self.logger.info(f"📊 Estatísticas do Grafo:")
            self.logger.info(f"   - Nós: {stats['total_nodes']}")
            self.logger.info(f"   - Arestas: {stats['total_edges']}")
            self.logger.info(f"   - Densidade: {stats['density']:.6f}")
            self.logger.info(f"   - Grau médio: {stats['avg_degree']:.2f}")
            
            return len(nodes_data), len(edges_data)
            
        except Exception as e:
            self.logger.critical(f"💥 ERRO na exportação para Gephi: {e}", exc_info=True)
            raise