import networkx as nx
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from itertools import combinations
import math

class GraphExporter:
    def __init__(self):
        self.graph = nx.Graph()
    
    def create_similarity_graph_optimized(self, texts, fingerprints, labels, filenames, threshold=10, max_edges_per_node=20):
        """
        Versão otimizada que evita comparação O(n²)
        """
        n = len(fingerprints)
        print(f"🕸️ Criando grafo com {n} nós...")
        
        # 1. Estratégia: Agrupar por prefixos para reduzir comparações
        prefix_length = 8  # Primeiros 8 bits para agrupamento
        groups = defaultdict(list)
        
        for i, fp in enumerate(fingerprints):
            prefix = bin(fp)[2:].zfill(64)[:prefix_length]
            groups[prefix].append(i)
        
        print(f"📊 Criados {len(groups)} grupos por prefixo")
        
        # 2. Adicionar nós primeiro
        for i, (text, fingerprint, label, filename) in enumerate(zip(texts, fingerprints, labels, filenames)):
            self.graph.add_node(i, 
                              label=f"News_{i}",
                              fingerprint=fingerprint,
                              text_preview=text[:50] + "..." if len(text) > 50 else text,
                              type="Fake" if label == 1 else "True",
                              filename=filename,
                              degree=0)  # Para controle
        
        # 3. Comparação inteligente apenas dentro dos grupos e entre grupos vizinhos
        edges_added = 0
        group_keys = list(groups.keys())
        
        for idx, key in enumerate(group_keys):
            indices = groups[key]
            
            # Comparação dentro do grupo
            if len(indices) > 1:
                for i, j in combinations(indices, 2):
                    if edges_added >= n * max_edges_per_node:
                        break
                        
                    distance = bin(fingerprints[i] ^ fingerprints[j]).count("1")
                    if distance <= threshold:
                        self.graph.add_edge(i, j, weight=1-distance/threshold, distance=distance)
                        edges_added += 1
            
            # Comparação com grupos vizinhos (prefixos similares)
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
                            
                        distance = bin(fingerprints[i] ^ fingerprints[j]).count("1")
                        if distance <= threshold:
                            self.graph.add_edge(i, j, weight=1-distance/threshold, distance=distance)
                            edges_added += 1
        
        print(f"✅ Grafo criado com {len(self.graph.nodes)} nós e {edges_added} arestas")
        return edges_added

    def create_similarity_graph_sampling(self, texts, fingerprints, labels, filenames, threshold=10, sample_rate=0.1):
        """
        Versão com amostragem: compara apenas uma fração dos pares
        """
        n = len(fingerprints)
        print(f"🕸️ Criando grafo com {n} nós (amostragem: {sample_rate*100}%)...")
        
        # Adicionar todos os nós
        for i, (text, fingerprint, label, filename) in enumerate(zip(texts, fingerprints, labels, filenames)):
            self.graph.add_node(i, 
                              label=f"News_{i}",
                              fingerprint=fingerprint,
                              text_preview=text[:50] + "..." if len(text) > 50 else text,
                              type="Fake" if label == 1 else "True",
                              filename=filename)
        
        # Amostragem inteligente: comparar cada nó com uma amostra aleatória
        edges_added = 0
        rng = np.random.default_rng(42)  # Para reproducibilidade
        
        for i in range(n):
            # Número de comparações por nó baseado no sample_rate
            k = max(1, int(n * sample_rate))
            
            # Amostrar k índices aleatórios (excluindo o próprio i)
            other_indices = rng.choice([x for x in range(n) if x != i], 
                                     size=min(k, n-1), replace=False)
            
            for j in other_indices:
                distance = bin(fingerprints[i] ^ fingerprints[j]).count("1")
                if distance <= threshold:
                    self.graph.add_edge(i, j, weight=1-distance/threshold, distance=distance)
                    edges_added += 1
        
        print(f"✅ Grafo criado com {len(self.graph.nodes)} nós e {edges_added} arestas")
        return edges_added

    def create_similarity_graph_knn(self, texts, fingerprints, labels, filenames, k_neighbors=10, threshold=15):
        """
        Versão K-NN: cada nó conecta apenas aos k vizinhos mais próximos
        """
        n = len(fingerprints)
        print(f"🕸️ Criando grafo K-NN com {n} nós (k={k_neighbors})...")
        
        # Converter fingerprints para array numpy para eficiência
        fp_array = np.array([list(map(int, bin(fp)[2:].zfill(64))) for fp in fingerprints])
        
        # Adicionar todos os nós
        for i, (text, fingerprint, label, filename) in enumerate(zip(texts, fingerprints, labels, filenames)):
            self.graph.add_node(i, 
                              label=f"News_{i}",
                              fingerprint=fingerprint,
                              text_preview=text[:50] + "..." if len(text) > 50 else text,
                              type="Fake" if label == 1 else "True",
                              filename=filename)
        
        edges_added = 0
        
        for i in range(n):
            distances = []
            
            # Calcular distâncias apenas para um subconjunto (amostragem)
            sample_size = min(100, n)  # Comparar com no máximo 100 outros nós
            if n > 100:
                indices = np.random.choice([x for x in range(n) if x != i], sample_size, replace=False)
            else:
                indices = [x for x in range(n) if x != i]
            
            for j in indices:
                distance = bin(fingerprints[i] ^ fingerprints[j]).count("1")
                distances.append((j, distance))
            
            # Ordenar por distância e pegar os k mais próximos
            distances.sort(key=lambda x: x[1])
            for j, dist in distances[:k_neighbors]:
                if dist <= threshold:
                    self.graph.add_edge(i, j, weight=1-dist/threshold, distance=dist)
                    edges_added += 1
        
        print(f"✅ Grafo K-NN criado com {len(self.graph.nodes)} nós e {edges_added} arestas")
        return edges_added

    def create_similarity_graph_batch(self, texts, fingerprints, labels, filenames, threshold=10, batch_size=100):
        """
        Processamento em lotes para economizar memória
        """
        n = len(fingerprints)
        print(f"🕸️ Criando grafo em lotes ({batch_size} nós por lote)...")
        
        # Adicionar todos os nós primeiro
        for i, (text, fingerprint, label, filename) in enumerate(zip(texts, fingerprints, labels, filenames)):
            self.graph.add_node(i, 
                              label=f"News_{i}",
                              fingerprint=fingerprint,
                              text_preview=text[:50] + "..." if len(text) > 50 else text,
                              type="Fake" if label == 1 else "True",
                              filename=filename)
        
        edges_added = 0
        num_batches = math.ceil(n / batch_size)
        
        for batch_idx in range(num_batches):
            print(f"   Processando lote {batch_idx + 1}/{num_batches}...")
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, n)
            
            # Comparar nós deste lote com todos os nós anteriores
            for i in range(start, end):
                for j in range(i):  # Apenas com nós anteriores para evitar duplicatas
                    distance = bin(fingerprints[i] ^ fingerprints[j]).count("1")
                    if distance <= threshold:
                        self.graph.add_edge(i, j, weight=1-distance/threshold, distance=distance)
                        edges_added += 1
            
            # Liberar memória
            if batch_idx % 10 == 0:
                import gc
                gc.collect()
        
        print(f"✅ Grafo criado com {len(self.graph.nodes)} nós e {edges_added} arestas")
        return edges_added

    def export_for_gephi(self, output_path="gephi_data"):
        """Exporta nodes e edges para CSV compatível com Gephi"""
        os.makedirs(output_path, exist_ok=True)
        
        # Nodes CSV
        nodes_data = []
        for node_id, attrs in self.graph.nodes(data=True):
            nodes_data.append({
                'Id': node_id,
                'Label': attrs['label'],
                'Type': attrs['type'],
                'TextPreview': attrs['text_preview'],
                'Filename': attrs['filename'],
                'Degree': self.graph.degree(node_id)
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(f"{output_path}/nodes.csv", index=False, encoding='utf-8')
        
        # Edges CSV
        edges_data = []
        for edge in self.graph.edges(data=True):
            edges_data.append({
                'Source': edge[0],
                'Target': edge[1],
                'Weight': edge[2]['weight'],
                'Distance': edge[2]['distance'],
                'Type': 'Undirected'
            })
        
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(f"{output_path}/edges.csv", index=False)
        
        # Estatísticas
        stats = {
            'total_nodes': len(nodes_data),
            'total_edges': len(edges_data),
            'density': len(edges_data) / (len(nodes_data) * (len(nodes_data) - 1) / 2) if len(nodes_data) > 1 else 0,
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(nodes_data) if len(nodes_data) > 0 else 0
        }
        
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(f"{output_path}/graph_stats.csv", index=False)
        
        print(f"💾 Dados exportados para {output_path}/")
        print(f"   - nodes.csv: {len(nodes_data)} nós")
        print(f"   - edges.csv: {len(edges_data)} arestas")
        print(f"   - graph_stats.csv: estatísticas do grafo")
        print(f"   - Densidade: {stats['density']:.6f}")
        print(f"   - Grau médio: {stats['avg_degree']:.2f}")
        
        return len(nodes_data), len(edges_data)