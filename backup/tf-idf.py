import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFComparator:
    def __init__(self, fingerprint_dir="outputs/fingerprints_test"):
        self.fingerprint_dir = fingerprint_dir

    def load_fingerprints(self, bits):
        path = os.path.join(self.fingerprint_dir, f"fp_{bits}.txt")
        if not os.path.exists(path):
            print(f"Arquivo não encontrado para {bits} bits.")
            return None, None

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        try:
            h_line = [line for line in lines if "HUMANO" in line][0].split(":")[1].strip()
            ia_line = [line for line in lines if "IA" in line][0].split(":")[1].strip()
            return h_line, ia_line
        except:
            print(f"Erro ao processar arquivo {bits} bits.")
            return None, None

    def compare_all(self, bit_sizes):
        print("\n🎯 Comparando fingerprints com TF-IDF + Cosine Similarity...\n")
        for bits in bit_sizes:
            human_fp, ia_fp = self.load_fingerprints(bits)
            if human_fp is None or ia_fp is None:
                continue

            tfidf = TfidfVectorizer(analyzer='char')
            vectors = tfidf.fit_transform([human_fp, ia_fp])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

            print(f"[{bits} bits] Similaridade TF-IDF: {similarity:.4f} ➜ {'🔴 ALTA' if similarity > 0.7 else '🟢 BAIXA'}")


# Execução direta
if __name__ == "__main__":
    sizes = [64, 128, 256, 512, 1024, 2048]
    comparator = TFIDFComparator()
    comparator.compare_all(sizes)
