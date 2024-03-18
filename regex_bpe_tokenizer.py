import os
import shutil
from typing import Optional
import pickle
import numpy as np
from tqdm.auto import tqdm
import regex as re

class ConfigError(Exception):
    """for invalid user configuration of tokenizer class"""
    pass

class regex_tokenizer:
    """
        This is the base class for tokenizer, it provde basic functionalities such as produce pair counts and merge new tokens

    """
    def __init__(self):
        # initialize base vocabulary dictionary which is the character encoding based on UTF-8
        self._base_vocab = {i: bytes([i]) for i in range(256)}
        self._base_vocab_size = 256
        self.pattern = re.compile(r""" ?<\|[a-z]+\|>|'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
        
        
    
    def _get_file_paths(self,title,vocab_size,tokenizer_folder_path=os.getcwd()):
        self.folder = title + "_regex_tok_folder"
        self.vocab_path = os.path.join(tokenizer_folder_path,self.folder, title + "_vocab_dict_size"+str(vocab_size)+".pkl")
        self.merges_path = os.path.join(tokenizer_folder_path,self.folder, title + "_merge_history_size"+str(vocab_size)+".pkl")
        self.tokens_path = os.path.join(tokenizer_folder_path,self.folder, title + "_tokens_size"+str(vocab_size)+".txt")

    def _get_pair_counts(self,tokens, count_dict={}):
        """
            treverse through the entire encoded text, produce a dictionary with paired occurrences of adjacent tokens
                key: token pairs, e.g., (106, 32)
                value: counts of occurrence of key, e.g., 300
                meaning: token pair (106, 32) occurred 300 times in the text
        """
        for (c1, c2) in zip(tokens[:-1],tokens[1:]):
            count_dict[(c1,c2)] = count_dict.get((c1,c2),0) + 1
        return count_dict
    
    def _merge_pair(self,tokens,pair,new_token):
        """
            Replace all occurrences of pair in tokens by new_token
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i],tokens[i+1]) == pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def _recover_chunks(self,text):
        """
            recovers converted nested list of integers back to nested list of integers
        """
        recovered_chunks = [[int(num) for num in block.strip().split(" ")] for block in text.split("\n")]
        return recovered_chunks
    
    def _retrieve_training_history(self,title,vocab_size,tokenizer_folder_path=os.getcwd(),is_train=True):
        """
            retrieve the dictionaries
        """
        self._get_file_paths(title,vocab_size,tokenizer_folder_path)
        try:
            with open(self.vocab_path,"rb") as f:
                vocab = pickle.load(f)
            with open(self.merges_path,"rb") as f:
                merges = pickle.load(f)
            if is_train:
                with open(self.tokens_path,"r") as f:
                    chunks = f.read()
                token_chunks = self._recover_chunks(chunks)
            else: token_chunks = None
        except:
            m = f"Dictionary files do not exit, tokenizer requires training with {title} dataset. \nOr provided with inconsistent vocab_size, use os.listdir to inspect dictionary files."
            m_more = " Or past_tokens cannot be retreived, if this is the case, encode text first"
            raise FileNotFoundError(m+m_more)
        else:
            return vocab, merges, token_chunks

    def _initialize_special_tokens(self,token_list):
        self.special_token_list = ['<|startofchapter|>']
        if token_list is not None:
            self.special_token_list.extend(token_list)
        special_tokens_start_index = 100000
        self.special_tokens = {}
        for i in range(len(self.special_token_list)):
            self.special_tokens[self.special_token_list[i]] = special_tokens_start_index+i
        self.inverse_special_tokens = {v:k for k,v in self.special_tokens.items()}
        self._get_file_paths(self.title,self.final_vocab_size)
        self._save_special_tokens()
    
    def _save_special_tokens(self,tokenizer_folder_path=os.getcwd(), mode = "training"):
        if mode == "training":
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
        
        with open(os.path.join(tokenizer_folder_path,self.folder,self.title+"_special_tokens.pkl"), "wb") as f:
            pickle.dump(self.special_tokens,f)
    
    def _read_update_special_tokens(self, token_list,tokenizer_folder_path=os.getcwd(), mode="training"):
        with open(os.path.join(tokenizer_folder_path,self.folder,self.title+"_special_tokens.pkl"), "rb") as f:
            self.special_tokens = pickle.load(f)
        if token_list is not None:
            special_token_list = list(self.special_tokens.keys())
            difference = list(set(token_list) - set(special_token_list))
            special_token_list.extend(difference)
            self.special_token_list = special_token_list
            max_idx = max(self.special_tokens.items(),key=lambda x:x[1])[1]
            for i in range(len(difference)):
                self.special_tokens[difference[i]] = max_idx + 1 + i
        else:
            self.special_token_list = list(self.special_tokens.keys())
        self.inverse_special_tokens = {v:k for k,v in self.special_tokens.items()}
        self._save_special_tokens(tokenizer_folder_path,mode)

class TrainTokenizer(regex_tokenizer):
    """
        This class implement compression algorithm described in 
            https://en.wikipedia.org/wiki/Byte_pair_encoding#:~:text=Byte%20pair%20encoding%20(also%20known,for%20use%20in%20downstream%20modeling
            with the addition of separating text into chunks before merging, avoid merging elements across categories: character with punctuation
        It takes text and title, train a tokenizer and store the files in a directory
        Args:
            text: str, actual text
            title: str, name of the mateirals that the tokenizer is training on
            speical_token_list: list of special tokens besides <|startofchapter|>
                must be in the form <|xxxxxx|>
            fresh_start: bool, whether to train from scratch or continue training/compressing, default=True
            final_vocab_size: int, final vocabulary size after compression - determines how many merges to perform, in thousands
            last_vocab_size: int, if continue training, what is the last final_vocab_size in thousands: 10 -> 10,000
        
        Folder/Title Naming Convention:
            "book_title_base_tok_folder"
        
        Sub-files (in the _tok_folder) Naming Convetion:
            "book_title_vocab_dict_size10.pkl" stores encoding dictionary, where 10 means 10,000 vocabulary size
            "book_title_merge_history_size10.pkl" stores merging history, where 10 means 10,000 vocabulary size
            "book_title_tokens_size10.npy" stores tokens from last compression, with tokenization with size 10,000
            "book_title_special_tokens.pkl" stores dictionary for special tokens

        
    """

    def __init__(self, text: str, title: str, special_token_list=None, final_vocab_size: int =6000, fresh_start: bool =True, last_vocab_size: Optional[int] =None):
        super(TrainTokenizer,self).__init__()
        self.title = title
        self.final_vocab_size = final_vocab_size
        # initialize training vocabulary and merge history dictionaries
        if fresh_start:
            self._initialize_special_tokens(special_token_list)
            self.vocab = self._base_vocab
            self.merge_history = {}
            for token in self.special_token_list:
                text = text.replace(token,"")
            text_chunks = re.findall(self.pattern,text)
            self.token_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        else:
            if last_vocab_size is None: raise ConfigError("for continue training (fresh_start == False), last final_vocab_size must be provided")
            if final_vocab_size <= last_vocab_size: raise ConfigError("unable to perform tokenizer training, because new vocabulary size must be larger than previous vocabulary size")
            self.vocab, self.merge_history, self.token_chunks = self._retrieve_training_history(self.title,last_vocab_size)
            self._read_update_special_tokens(special_token_list)
        assert len(self.vocab) == (len(self.merge_history) + self._base_vocab_size), "dictionary lengths not matching - the following should be true: voca = merge_hist + 256"
        assert final_vocab_size > len(self.vocab), f"final vocabulary size specified is too small, must be larger than {len(self.vocab)}"

        
    
    def _process_token_chunks(self,token_chunks):
        """
            make [[1,2,3], [4,5,6,7], [8,9,10]] into string: 
                1 2 3
                 4 5 6 7
                 8 9 10
        """
        chunks_str = str(token_chunks)
        chunks_str = chunks_str[1:-2] # removing leading "[" and trailing "]]"
        chunks_str = chunks_str.replace("[", "") # remove all "["
        chunks_str = chunks_str.replace("],", "\n") # replace end of each list to be a new line character
        chunks_str = chunks_str.replace(",", "") # remove all comma
        return chunks_str

    
    def _perform_merge(self,verbose):
        """
            Training loop compression process:
                1. identify top pair
                2. swap the occurrences of top pair in each token chunk
                3. update merge_history and vocab
            the training loop ignores special characters
            after training, vocab, merge_history are saved as pickle files and final tokens are saved as npy file
        """
        vocab_size = len(self.merge_history) + self._base_vocab_size
        num_merges = self.final_vocab_size - vocab_size
        progress_bar = tqdm(range(num_merges))

        for i in range(num_merges):
            progress_bar.update(1)
            pair_counts = {}
            for chunk in self.token_chunks:
                pair_counts = self._get_pair_counts(tokens=chunk,count_dict=pair_counts)
            top_pair = max(pair_counts,key=pair_counts.get)
            self.token_chunks = [self._merge_pair(tokens=chunk,pair=top_pair,new_token=vocab_size + i) for chunk in self.token_chunks]
            self.merge_history[top_pair] = vocab_size+i
            self.vocab[vocab_size+i] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            if verbose: print(f"merged {top_pair} as {vocab_size+i}")
        
        self._get_file_paths(self.title,self.final_vocab_size)

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        
        # save files to folder
        with open(self.vocab_path,"wb") as f:
            pickle.dump(self.vocab,f)
        with open(self.merges_path,"wb") as f:
            pickle.dump(self.merge_history, f)
        processed_token_chunks = self._process_token_chunks(self.token_chunks)
        with open(self.tokens_path,"w") as f:
            f.write(processed_token_chunks)
    
    def run(self, verbose=False):
        self._perform_merge(verbose)

class ApplyTokenizer(regex_tokenizer):
    """
        This subclass can encode and decode text based on trained tokenizer
        Arg:
            mode: str, "encode" or "decode"
            title: str, tokenizer trained on which texts user wish to apply
            vocab_size: int, which version of the tokenizer user wish to apply, usually in thousands
            tokenizer_folder_path: file path to the tokenizer folder, to access the dictionaries
            speical_token_list: list of special tokens besides <|startofchapter|>
                must be in the form <|xxxxxx|>

    """
    def __init__(self, title, vocab_size, tokenizer_folder_path, special_token_list = None):
        super(ApplyTokenizer,self).__init__()
        self.title = title
        self.vocab, self.merge_history, _ = self._retrieve_training_history(title=title,vocab_size=vocab_size,tokenizer_folder_path=tokenizer_folder_path,is_train=False)
        self._read_update_special_tokens(special_token_list,tokenizer_folder_path,"infer")
        self._reconcile_special_token()

    def _encode_chunk(self,text_chunk):
        tokens = list(text_chunk.encode("utf-8"))
        # print(f"tokens: {tokens}")
        while len(tokens) >= 2:
            # print(f"input tokens {tokens}")
            pairs = self._get_pair_counts(tokens=tokens,count_dict={}) # dictionary of (101, 102) 30000
            # print(f"pairs: {pairs}")
            pair_replace = min(pairs,key=lambda x: self.merge_history.get(x,float('inf'))) # if no match, returns itself
            # print(f"pair to replace: {pair_replace}")
            if pair_replace not in self.merge_history:
                break
            new_token = self.merge_history[pair_replace]
            # print(f"new tokens: {new_token}")
            tokens = self._merge_pair(tokens=tokens,pair=pair_replace,new_token=new_token)
            # print(f"updated tokens: {tokens}")
        return tokens
    
    def encode(self,text):
        assert type(text) == str, "input for encoding not string"
        text_chunks = re.findall(self.pattern,text)
        encoded_tokens = []
        for chunk in text_chunks:
            if chunk.strip() in self.special_token_list:
                encoded_tokens.append(self.special_tokens[chunk.strip()])
            else:
                encoded_tokens.extend(self._encode_chunk(chunk))
        return encoded_tokens

    def decode(self,tokens):
        parts_decoded = []
        for token in tokens:
            if token in self.vocab:
                parts_decoded.append(self.vocab[token])
            elif token in self.inverse_special_tokens:
                parts_decoded.append(self.inverse_special_tokens[token].encode("utf-8"))
            else:
                raise ValueError(f"{token} is not part of vocabulary or special characters")
        text = b"".join(parts_decoded)
        text_decoded = text.decode("utf-8",errors="replace")
        return text_decoded
    
    def _reconcile_special_token(self):
        """
            This function assign index values for special tokens with index directly after vocabulary count
            For example: previously <|title|> is 1000000, say vocabulary size is 8000, then after reconcile, <|title|> is 8000 or 8001
        """
        max_idx = len(self.vocab) - 1
        for  i in range(len(self.special_token_list)):
            self.special_tokens[self.special_token_list[i]] = max_idx + 1 + i
        self.inverse_special_tokens = {v:k for k, v in self.special_tokens.items()}

        


    

