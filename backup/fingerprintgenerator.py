import time
import hashlib
import os

class FingerprintGenerator:
    def __init__(self, hash_sizes, output_dir="outputs/fingerprints_test"):
        self.hash_sizes = hash_sizes
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_simhash(self, code, bits):
        tokens = self._tokenize(code)
        vector = [0] * bits

        for token in tokens:
            hash_bits = self._hash_token_to_bits(token, bits)
            for i in range(bits):
                vector[i] += 1 if hash_bits[i] == '1' else -1

        final_bits = ''.join(['1' if v > 0 else '0' for v in vector])
        return int(final_bits, 2)

    def _tokenize(self, code):
        n = 4  # n-gramas de 4 caracteres
        return [code[i:i+n] for i in range(len(code)-n+1)]

    def _hash_token_to_bits(self, token, bits):
        # Gera hash SHA-512 (512 bits)
        combined_hash = b''
        i = 0
        while len(combined_hash) * 8 < bits:
            h = hashlib.sha512((token + str(i)).encode('utf-8')).digest()
            combined_hash += h
            i += 1

        # Converte para binário e trunca até o tamanho desejado
        binary = ''.join(f'{byte:08b}' for byte in combined_hash)
        return binary[:bits]

    def generate_and_compare(self, code_human, code_ia):
        print("Iniciando geração de fingerprints com múltiplos tamanhos...\n")
        for bits in self.hash_sizes:
            print(f"[{bits} bits]")
            start = time.time()

            hash_h = self.generate_simhash(code_human, bits)
            hash_ia = self.generate_simhash(code_ia, bits)

            end = time.time()
            exec_time = end - start

            hash_h_bin = bin(hash_h)
            hash_ia_bin = bin(hash_ia)

            print(f"Fingerprint HUMANO : {hash_h_bin[:80]}...")  # Truncar para visualização
            print(f"Fingerprint IA     : {hash_ia_bin[:80]}...")
            print(f"Tempo de execução: {exec_time:.4f} segundos\n")

            # Salvar em arquivo
            output_path = os.path.join(self.output_dir, f"fp_{bits}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Fingerprint HUMANO: {hash_h_bin}\n")
                f.write(f"Fingerprint IA    : {hash_ia_bin}\n")
                f.write(f"Tempo de execução: {exec_time:.4f} segundos\n")
                print("H:", hash_h_bin[:80])
                print("I:", hash_ia_bin[:80])


# Teste direto
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))

    from news_normalizer import CodeNormalizer

    normalizer = CodeNormalizer()
    code_h, code_ia = normalizer.normalize_both_codes()

    sizes = [64, 128, 256, 512, 1024, 2048]
    generator = FingerprintGenerator(hash_sizes=sizes)
    generator.generate_and_compare(code_h, code_ia)

