import numpy as np
import torch
class HelperFunctionsClass:

    # train_test_split function identifies one book from full text - multiple books, then return train and validation set for that book
    def train_test_split(self,text,ratio,starting_idx=0, book_start_token=False,ending=None,token="<|startofbook|>"):
        if book_start_token:
            ending_idx = text.find(token,10+starting_idx) -1
            if ending_idx == -2: ending_idx = -1
        else:
            assert ending is not None, "Have to provide ending of the book unless there is book_start_token"
            ending_idx = text.find(ending)
            ending_idx += len(ending)
        book = text[starting_idx:ending_idx]
        n = int(ratio*len(book))
        n = self.locate_next_break(book,n)
        return book[:n], book[n:], ending_idx
    
    # this function takes in text and an index, it finds the most immediate setence end from the given index
    def locate_next_break(self,text,n):
        ending = ['.','!','?']
        while text[n] not in ending:
            n += 1
        return n + 1
    
    # this function convert [1, 2, 3] into 1 2 3
    def int_array_to_str(self, array):
        array_str = list(map(str,array))
        array_str = " ".join(array_str)
        return array_str
    
    # this function reads converted-str-list file then return int list
    def convert_str_file_to_int_array(self, file_path,convert_to_torch = True):
        if convert_to_torch: return torch.tensor(np.loadtxt(file_path,dtype=int),dtype=torch.long)
        else: return np.loadtxt(file_path,dtype=int)