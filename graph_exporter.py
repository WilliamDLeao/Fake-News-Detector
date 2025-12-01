import networkx as nx
import pandas as pd
import numpy as np
import os
import time
import tracemalloc
from collections import defaultdict
from itertools import combinations
import psutil
import gc

class GraphExporter:
    def __init__(self):
        self.graph = nx.Graph()
    
    def create_similarity_graph_optimized(self, texts, fingerprints, labels, filenames, dates, threshold=10, max_edges_per_node=20):
        """
        Versão otimizada que evita comparação O(n²)
        """
        print("🚀 Criando grafo otimizado...")
        start_time = time.time()
        tracemalloc.start()
        
        try:
            n = len(fingerprints)
            
            if len(dates) != n:
                raise ValueError("Array dates deve ter o mesmo tamanho")
            
            # 1. Estratégia: Agrupar por prefixos
            prefix_length = 8
            
            groups = defaultdict(list)
            
            for i, fp in enumerate(fingerprints):
                try:
                    fp_bin = bin(fp)[2:].zfill(64)
                    prefix = fp_bin[:prefix_length]
                    groups[prefix].append(i)
                except Exception:
                    continue
            
            # 2. Adicionar nós primeiro
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
                except Exception:
                    continue
            
            # 3. Comparação inteligente
            edges_added = 0
            group_keys = list(groups.keys())
            
            for idx, key in enumerate(group_keys):
                indices = groups[key]
                
                # Comparação dentro do grupo
                if len(indices) > 1:
                    for i, j in combinations(indices, 2):
                        if edges_added >= n * max_edges_per_node:
                            break
                            
                        try:
                            distance = bin(fingerprints[i] ^ fingerprints[j]).count("1")
                            if distance <= threshold:
                                self.graph.add_edge(i, j, weight=1-distance/threshold, distance=distance)
                                edges_added += 1
                        except Exception:
                            continue
                
                # Comparação com grupos vizinhos
                if idx < len(group_keys) - 1:
                    next_key = group_keys[idx + 1]
                    next_indices = groups[next_key]
                    
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
                            except Exception:
                                continue
            
            total_time = time.time() - start_time
            print(f"✅ Grafo otimizado criado com {edges_added} arestas em {total_time:.2f}s")
            return edges_added
            
        except Exception as e:
            print(f"❌ Erro ao criar grafo otimizado: {e}")
            raise
        finally:
            tracemalloc.stop()
            gc.collect()
    
    def create_similarity_graph_knn(self, texts, fingerprints, labels, filenames, dates, k_neighbors=10, threshold=15):
        """
        Versão K-NN
        """
        start_time = time.time()
        tracemalloc.start()
        
        try:
            n = len(fingerprints)
            
            if len(dates) != n:
                raise ValueError("Array dates deve ter o mesmo tamanho")
            
            # Adicionar todos os nós
            for i, (text, fingerprint, label, filename, date) in enumerate(zip(texts, fingerprints, labels, filenames, dates)):
                self.graph.add_node(i, 
                                  label=f"News_{i}",
                                  fingerprint=fingerprint,
                                  text_preview=text[:50] + "..." if len(text) > 50 else text,
                                  type="Fake" if label == 1 else "True",
                                  filename=filename,
                                  date=date)
            
            edges_added = 0
            
            for i in range(n):
                distances = []
                
                # Calcular distâncias para um subconjunto
                sample_size = min(100, n)
                if n > 100:
                    indices = np.random.choice([x for x in range(n) if x != i], sample_size, replace=False)
                else:
                    indices = [x for x in range(n) if x != i]
                
                for j in indices:
                    try:
                        distance = bin(fingerprints[i] ^ fingerprints[j]).count("1")
                        distances.append((j, distance))
                    except Exception:
                        continue
                
                # Ordenar por distância e pegar os k mais próximos
                distances.sort(key=lambda x: x[1])
                for j, dist in distances[:k_neighbors]:
                    if dist <= threshold:
                        try:
                            self.graph.add_edge(i, j, weight=1-dist/threshold, distance=dist)
                            edges_added += 1
                        except Exception:
                            continue
            
            total_time = time.time() - start_time
            return edges_added
            
        except Exception as e:
            print(f"❌ Erro ao criar grafo K-NN: {e}")
            raise
        finally:
            tracemalloc.stop()
            gc.collect()

    def export_for_gephi(self, output_path="gephi_data"):
        """Exporta nodes e edges para CSV compatível com Gephi"""
        start_time = time.time()
        
        try:
            os.makedirs(output_path, exist_ok=True)
            
            # Nodes CSV
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
            
            # Edges CSV
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
            
            
            return len(nodes_data), len(edges_data)
            
        except Exception as e:
            print(f"❌ Erro na exportação para Gephi: {e}")
            raise

