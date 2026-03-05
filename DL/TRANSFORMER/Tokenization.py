import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
    def build_vocab(self, texts: List[str]) -> None:

        special_tokens = [
            self.pad_token,   
            self.unk_token,   
            self.bos_token,   
            self.eos_token,   
        ]

        for token in special_tokens:
            idx = self.vocab_size               
            self.word_to_id[token] = idx        
            self.id_to_word[idx] = token        
            self.vocab_size += 1                

        for sentence in texts:                  
            for word in sentence.split():       
                if word not in self.word_to_id: 
                    idx = self.vocab_size
                    self.word_to_id[word] = idx
                    self.id_to_word[idx] = word
                    self.vocab_size += 1

    
    def encode(self, text: str) -> List[int]:
        unk_id = self.word_to_id[self.unk_token] 
        return [self.word_to_id.get(word, unk_id)for word in text.split()] 
    
    
    def decode(self, ids: List[int]) -> str:
        return " ".join(self.id_to_word.get(i, self.unk_token)for i in ids)
