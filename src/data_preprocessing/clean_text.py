import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import emoji
from langdetect import detect
from spellchecker import SpellChecker

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class TextCleaner():
    def __init__(self):
        self.stop_words=set(stopwords.words('english'))
        self.lemmatizer=WordNetLemmatizer()
        self.spell_checker=SpellChecker()

    def remove_emojis(self,text):
        return emoji.replace_emoji(text,replace='')
    
    def language_detect(self, text):
        try:
            if not text.strip():
                raise ValueError("No features in text.")
            language=detect(text)
            return language
        except Exception as e:
            print(f"Exception occurred in Language_detect function: {e}")
            return 'Unknown'

    def preprocess_text(self,text,spell_check=False):
        # language=self.language_detect(text=text)
        # if language != 'en':
        #     return ""
        text=self.remove_emojis(text=text)
        text=text.lower()
        text=re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
        text=''.join([char for char in text if char not in string.punctuation])
        tokens=word_tokenize(text=text,language='english')
        tokens=[word for word in tokens if word not in self.stop_words]
        if spell_check:
            tokens=[self.spell_checker.correction(word) for word in tokens ]
            tokens=[word for word in tokens if word is not None]
        tokens=[self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    



