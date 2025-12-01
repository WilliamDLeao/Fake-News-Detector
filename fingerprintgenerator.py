
import time
import hashlib
import os

class FingerprintGenerator:
    def __init__(self, hash_sizes, output_dir="outputs/fingerprints"):
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
        combined_hash = b''
        i = 0
        while len(combined_hash) * 8 < bits:
            h = hashlib.sha512((token + str(i)).encode('utf-8')).digest()
            combined_hash += h
            i += 1

        binary = ''.join(f'{byte:08b}' for byte in combined_hash)
        return binary[:bits]

    def hamming_distance(self, hash1, hash2):
        return bin(hash1 ^ hash2).count("1")

    def generate_and_compare(self, code_true, code_fake, project_name="proj"):
        print(f"\n=== Projeto {project_name} ===")
        for bits in self.hash_sizes:
            start = time.time()

            hash_true = self.generate_simhash(code_true, bits)
            hash_fake = self.generate_simhash(code_fake, bits)
            distance = self.hamming_distance(hash_true, hash_fake)

            end = time.time()
            exec_time = end - start

            hash_true_bin = bin(hash_true)
            hash_fake_bin = bin(hash_fake)

            print(f"[{bits} bits]")
            print(f"gerar fingerprint - texto true : {hash_true_bin[:80]}...")
            print(f"gerar fingerprint - texto fake : {hash_fake_bin[:80]}...")
            print(f"Distância Hamming  : {distance}")
            print(f"Tempo de execução  : {exec_time:.4f} segundos\n")

            # Salvar em arquivo separado por projeto
            safe_name = project_name.replace("/", "_")
            output_path = os.path.join(self.output_dir, f"{safe_name}_fp_{bits}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"gerar fingerprint - texto true : {hash_true_bin}\n")
                f.write(f"gerar fingerprint - texto fake : {hash_fake_bin}\n")
                f.write(f"Distância Hamming : {distance}\n")
                f.write(f"Tempo de execução : {exec_time:.4f} segundos\n")

