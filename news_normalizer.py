import re

class NewsNormalizer:
    def __init__(self):
        self.patterns = {
            'news_agency_prefix': r'^[A-Z][A-Za-z\s]+\([A-Za-z]+\)\s*-\s*',
            'url': r'https?://\S+',
            'mention': r'@\w+',
            'hashtag': r'#\w+',
            'number': r'\b\d+[\.,]?\d*\b',
            'date': r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
            'name': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'currency': r'R\$\s?\d+[\.,]?\d*',
            'organization': r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?55\s?)?(?:\(?\d{2}\)?[\s-]?)?\d{4,5}[\s-]?\d{4}\b'
        }
        
        # Padrões para padrões de chamada editoriais editoriais
        self.editorial_patterns = [
            # Padrões entre parênteses, remove qualquer coisa entre parênteses no início
            r'^\s*\([^)]*\)\s*',
            r'\s*\([^)]*\)\s*$',
            # Padrões entre colchetes, remove qualquer coisa entre parênteses no início
            r'^\s*\[[^\]]*\]\s*', 
            r'\s*\[[^\]]*\]\s*$',
            # Palavras características da gringa
            r'^\s*(VIDEO|AUDIO|PHOTOS?|PICTURES?|WATCH|LISTEN|READ|LIVE|EXCLUSIVE|BREAKING?|BREAKING|JUST IN|JUSTIN)\s*[:\-\s]*',
            r'\s*(VIDEO|AUDIO|PHOTOS?|PICTURES?|WATCH|LISTEN|READ|LIVE|EXCLUSIVE|BREAKING?|BREAKING|JUST IN|JUSTIN)\s*$',
            # Combinações com traços/pontuação
            r'\s*-\s*(VIDEO|AUDIO|PHOTOS?|PICTURES?|WATCH|LISTEN|READ|LIVE|EXCLUSIVE|BREAKING?|BREAKING)',
            r'\s*:\s*(VIDEO|AUDIO|PHOTOS?|PICTURES?|WATCH|LISTEN|READ|LIVE|EXCLUSIVE|BREAKING?|BREAKING)'
        ]
    
    def normalize_news(self, text: str, is_true_news: bool = False) -> str:
        if not isinstance(text, str):
            return ""
        
        original_text = text
        
        # ETAPA 1: Remover prefixos de agências de notícias (apenas para true news)
        if is_true_news:
            text = re.sub(self.patterns['news_agency_prefix'], '', text)
        
        # ETAPA 2: Remoção de traços editoriais
        # Primeiro: remover QUALQUER coisa entre parênteses ou colchetes no início/fim
        text = re.sub(r'^\s*[\(\[].*?[\)\]]\s*', '', text)
        text = re.sub(r'\s*[\(\[].*?[\)\]]\s*$', '', text)
        
        # Segundo: aplicar padrões específicos
        for pattern in self.editorial_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # Terceiro: remover palavras específicas em qualquer posição
        editorial_words = ['video', 'audio', 'photos', 'pictures', 'watch', 'listen', 
                         'read', 'live', 'exclusive', 'breaking', 'just in', 'justin']
        for word in editorial_words:
            # Remover a palavra exata (case insensitive)
            text = re.sub(r'\b' + re.escape(word) + r'\b', '', text, flags=re.IGNORECASE)
        
        
        # ETAPA 3: Aplica todas as normalizações padrão
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
        
        # ETAPA 4: Limpeza final 
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove pontuação
        text = re.sub(r'\s+', ' ', text)      # Espaços múltiplos 
        text = text.strip().lower()
        
        return text