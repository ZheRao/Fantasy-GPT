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