# Sistema de Fingerprint de Notícias e Exportação em Grafos

Um sistema para analisar e classificar notícias usando técnicas de fingerprint e grafos.

## Funcionalidades

- **Normalização de Textos**: Limpeza e preparação de textos
- **Geração de Fingerprints**: Cria códigos únicos para cada texto usando SimHash
- **Análise de Similaridade**: Cria grafos para visualizar relações entre notícias
- **Exportação para Gephi**: Gera arquivos para análise em software de redes

## Módulos Principais

1. **NewsNormalizer** - Limpa e normaliza textos
2. **FingerprintGenerator** - Gera códigos hash para comparação
3. **nlp_module** - Classifica notícias usando MLP
4. **GraphExporter** - Cria e exporta grafos de similaridade

## Fluxo do Sistema

```
Texto Original → Normalização → SimHash → Grafo
```

## Pré-requisitos

```bash
pip install networkx pandas numpy scikit-learn
```

Versões usadas:
- networkx 3.5
- numpy 2.3.4
- pandas 2.3.3
- scikit-learn 1.7.0

## Como Usar

### 1. Preparar os Dados

Coloque os arquivos CSV na pasta principal:
- `True.csv` - Notícias verdadeiras
- `Fake.csv` - Notícias falsas

### 2. Executar 

```bash
python main.py
```

## Estrutura de Arquivos

```
.
├── main.py                 # Programa principal
├── fingerprintgenerator.py # Gera fingerprints
├── news_normalizer.py      # Normaliza textos
├── graph_exporter.py       # Trabalha com grafos
├── constant.py            # Configurações
```

## Configurações

No arquivo `constant.py`:
```python
PORCENTAGEM_ANALISADA = 0.2  # Percentual de dados para análise
```

No `FingerprintGenerator`:
```python
hash_sizes = [64, 128]  # Tamanhos de hash suportados
```

## Otimizações para Grandes Volumes de Dados

- **Amostragem**: Usa apenas parte dos dados
- **Agrupamento por Prefixos**: Compara apenas textos similares
- **Processamento em Lotes**: Economiza memória
- **Algoritmo K-NN**: Cria grafos de forma eficiente

## Resultados Gerados

### 1. Fingerprints
- Arquivos com hashes SimHash
- Tempos de execução e distâncias de Hamming

### 2. Grafos de Similaridade
- Arquivos CSV para usar no Gephi
- Nós (notícias) e arestas (similaridades)
- Estatísticas de conexão

## Algoritmos Usados

### SimHash
- Divide texto em pedaços de 4 caracteres
- Combina múltiplos hashes SHA512
- Gera vetor binário final

### Distância de Hamming
- Compara diferenças bit a bit entre fingerprints
- Limite configurável para similaridade

## Métricas do Sistema

- Acurácia de classificação
- Distâncias médias de Hamming
- Densidade dos grafos
- Tempos de execução
- Relatórios detalhados
