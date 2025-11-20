import re

class NewsNormalizer:
    def __init__(self):
        self.patterns = {
            'url': r'https?://\S+',
            'mention': r'@\w+',
            'hashtag': r'#\w+',
            'number': r'\b\d+[\.,]?\d*\b',
            'date': r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
            'name': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'currency': r'R\$\s?\d+[\.,]?\d*',  # Para valores monetários
            'organization': r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b',  # Nomes próprios e organizações
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?55\s?)?(?:\(?\d{2}\)?[\s-]?)?\d{4,5}[\s-]?\d{4}\b'
        }
    
    def normalize_news(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
            
        # Aplica todas as normalizações
        text = re.sub(self.patterns['url'], 'URL', text)
        text = re.sub(self.patterns['mention'], 'MENCAO', text)
        text = re.sub(self.patterns['hashtag'], 'HASHTAG', text)
        text = re.sub(self.patterns['number'], 'NUMERO', text)
        text = re.sub(self.patterns['date'], 'DATA', text)
        text = re.sub(self.patterns['name'], 'NOME_PESSOA', text)
        text = re.sub(self.patterns['currency'], 'VALOR_MONETARIO', text)
        text = re.sub(self.patterns['organization'], 'ORGANIZACAO', text)
        text = re.sub(self.patterns['email'], 'EMAIL', text)
        text = re.sub(self.patterns['phone'], 'TELEFONE', text)
        
        # Limpeza final - remove caracteres especiais e múltiplos espaços
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()