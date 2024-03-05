import os
import shutil

class update_utilities_class:
    def __init__(self,file_name:str, current_path:str, utils_path:str = r'..\helper_functions'):
        self.file_name = file_name
        self.utilities_path = utils_path
        self.destination_path = current_path
        self.destination_file_path = self.destination_path + r"/"+self.file_name
        self.utilities_file_path = self.utilities_path+r"/"+self.file_name
        
    
    # if the file already exist in the destination path, delete it
    def check_n_delete(self):
        if os.path.exists(self.destination_file_path):
            os.remove(self.destination_file_path)
            print(f"File already exist in destination folder, it is now removed")
    
    # copy the new file from utilities folder to destination path
    def copy_file(self):
        shutil.copyfile(self.utilities_file_path,self.destination_file_path)
        print(f"File copied, now the file is available to import from the destinated path")
    
    # run the script
    def run(self):
        self.check_n_delete()
        self.copy_file()