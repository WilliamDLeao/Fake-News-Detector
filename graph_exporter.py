import networkx as nx
import pandas as pd
import os

class GraphExporter:
    def __init__(self):
        self.graph = nx.Graph()
    
    def create_similarity_graph(self, texts, fingerprints, labels, filenames, threshold=10):
        """
        Cria grafo onde nós são notícias e arestas representam similaridade
        threshold: distância máxima de Hamming para considerar similar
        """
        
        # Adicionar nós
        for i, (text, fingerprint, label, filename) in enumerate(zip(texts, fingerprints, labels, filenames)):
            self.graph.add_node(i, 
                              label=f"News_{i}",
                              fingerprint=fingerprint,
                              text_preview=text[:100] + "..." if len(text) > 100 else text,
                              type="Fake" if label == 1 else "True",
                              filename=filename)
        
        # Adicionar arestas baseadas na similaridade
        edges_added = 0
        for i in range(len(fingerprints)):
            for j in range(i+1, len(fingerprints)):
                distance = bin(fingerprints[i] ^ fingerprints[j]).count("1")
                if distance <= threshold:
                    self.graph.add_edge(i, j, weight=1-distance/threshold, distance=distance)
                    edges_added += 1
        
        print(f"✅ Grafo criado com {len(self.graph.nodes)} nós e {edges_added} arestas")
    
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
                'Filename': attrs['filename']
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
        
        print(f"💾 Dados exportados para {output_path}/")
        print(f"   - nodes.csv: {len(nodes_data)} nós")
        print(f"   - edges.csv: {len(edges_data)} arestas")
        
        return len(nodes_data), len(edges_data)