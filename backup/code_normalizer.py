import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class CodeNormalizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=500)
        self.types = ['int', 'float', 'double', 'char', 'bool', 'string', 'long', 'short', 'auto']
        self.keywords = ['for', 'while', 'if', 'else', 'switch', 'case', 'return', 'break', 'continue']

    def normalize_code(self, code: str) -> str:
        # Remove comentários (mesmo estilo do antigo)
        code = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.DOTALL | re.MULTILINE)
        
        # Normaliza tipos (mantém "var" como no antigo)
        for t in self.types:
            code = re.sub(rf'\b{t}\s+\w+\b', f'{t} var', code)
        
        # Normaliza atribuições (mantém "var" como no antigo)
        code = re.sub(r'\b\w+\s*=', 'var =', code)
        
        # Normaliza keywords (prefixa com KEYWORD_)
        for kw in self.keywords:
            code = re.sub(rf'\b{kw}\b', f'KEYWORD_{kw}', code)
        
        # Remove excesso de espaços
        code = re.sub(r'\s+', ' ', code)
        return code.strip()

    def get_signature(self, code: str) -> np.ndarray:
        normalized = self.normalize_code(code)
        return self.vectorizer.transform([normalized]).toarray()[0]

    def fit_vectorizer(self, code_list: list):
        normalized_list = [self.normalize_code(code) for code in code_list]
        self.vectorizer.fit(normalized_list)

    # Método adicional para comparar dois códigos como no novo
    def normalize_both_codes(self):
        with open("dataBase/authors/human/h_p2_code1.cpp", "r", encoding="utf-8") as f:
            human_code = f.read()
        with open("dataBase/authors/ia/ia_p2_code1.cpp", "r", encoding="utf-8") as f:
            ia_code = f.read()

        norm_human = self.normalize_code(human_code)
        norm_ia = self.normalize_code(ia_code)

        return norm_human, norm_ia