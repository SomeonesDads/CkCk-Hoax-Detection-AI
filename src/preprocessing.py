"""
Text Preprocessing
====================
Indonesian text cleaning and normalization pipeline.
"""

import re
import string
from typing import Optional


class TextPreprocessor:
    """
    Indonesian text preprocessing pipeline.
    
    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove mentions and hashtags
        4. Remove HTML tags
        5. Normalize whitespace
        6. Remove excessive punctuation
        7. (Optional) Stemming with Sastrawi
    
    Usage:
        preprocessor = TextPreprocessor(use_stemmer=False)
        clean = preprocessor.clean("Ini tEks <b>HTML</b> https://example.com !!!")
    """

    def __init__(self, use_stemmer: bool = False):
        self.use_stemmer = use_stemmer
        self.stemmer = None

        if use_stemmer:
            try:
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                factory = StemmerFactory()
                self.stemmer = factory.createStemmer()
            except ImportError:
                print("[WARN] Sastrawi not installed. Stemming disabled.")
                print("       Install with: pip install Sastrawi")
                self.use_stemmer = False

    def lowercase(self, text: str) -> str:
        """Convert to lowercase."""
        return text.lower()

    def remove_urls(self, text: str) -> str:
        """Remove URLs."""
        return re.sub(r'https?://\S+|www\.\S+', '', text)

    def remove_mentions_hashtags(self, text: str) -> str:
        """Remove @mentions and #hashtags."""
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        return text

    def remove_html(self, text: str) -> str:
        """Remove HTML tags."""
        return re.sub(r'<[^>]+>', '', text)

    def remove_emojis(self, text: str) -> str:
        """Remove emojis and special unicode characters."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols
            "\U0001F680-\U0001F6FF"  # transport
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub('', text)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (collapse multiple spaces, strip)."""
        return re.sub(r'\s+', ' ', text).strip()

    def remove_excessive_punctuation(self, text: str) -> str:
        """Reduce repeated punctuation (e.g., '!!!' → '!')."""
        return re.sub(r'([!?.,:;])\1+', r'\1', text)

    def normalize_indonesian_slang(self, text: str) -> str:
        """
        Normalize common Indonesian internet slang.
        Add more mappings as needed for your domain.
        """
        slang_map = {
            r'\bgak\b': 'tidak',
            r'\bgk\b': 'tidak',
            r'\bga\b': 'tidak',
            r'\bgpp\b': 'tidak apa-apa',
            r'\byg\b': 'yang',
            r'\bdgn\b': 'dengan',
            r'\bsdh\b': 'sudah',
            r'\budh\b': 'sudah',
            r'\bblm\b': 'belum',
            r'\bjgn\b': 'jangan',
            r'\bjg\b': 'juga',
            r'\bkrn\b': 'karena',
            r'\bkpd\b': 'kepada',
            r'\bdr\b': 'dari',
            r'\btdk\b': 'tidak',
            r'\btp\b': 'tapi',
            r'\bspt\b': 'seperti',
            r'\btsb\b': 'tersebut',
            r'\bdll\b': 'dan lain-lain',
        }
        for pattern, replacement in slang_map.items():
            text = re.sub(pattern, replacement, text)
        return text

    def stem(self, text: str) -> str:
        """Apply Sastrawi stemmer if enabled."""
        if self.stemmer:
            return self.stemmer.stem(text)
        return text

    def clean(self, text: str, normalize_slang: bool = False) -> str:
        """
        Run the full preprocessing pipeline.
        
        Args:
            text: Raw input text.
            normalize_slang: Whether to normalize Indonesian slang.
            
        Returns:
            Cleaned text.
        """
        if not text or not isinstance(text, str):
            return ""

        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_mentions_hashtags(text)
        text = self.remove_emojis(text)
        text = self.lowercase(text)
        text = self.remove_excessive_punctuation(text)

        if normalize_slang:
            text = self.normalize_indonesian_slang(text)

        if self.use_stemmer:
            text = self.stem(text)

        text = self.normalize_whitespace(text)
        return text

    def clean_batch(self, texts: list[str], normalize_slang: bool = False) -> list[str]:
        """Clean a batch of texts."""
        return [self.clean(t, normalize_slang) for t in texts]


if __name__ == "__main__":
    preprocessor = TextPreprocessor(use_stemmer=False)

    samples = [
        "BREAKING!!! Vaksin COVID berbahaya!! Cek di https://hoax.com #antivax @beritahoax",
        "<p>Pemerintah <b>mengumumkan</b> kebijakan baru.</p>",
        "Gak percaya gw dgn berita ini!!! HOAX!!!",
    ]

    for text in samples:
        cleaned = preprocessor.clean(text, normalize_slang=True)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print()
