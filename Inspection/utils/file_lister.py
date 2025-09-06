from pathlib import Path

class FileLister:
    def __init__(self, path = None, file_type = None , not_include = None):
        self.path = path
        self.file_type = file_type
        self.not_include = not_include
        self.isdir = (file_type == 'dir')
        self._file_list = self.__getlist()  # 用私有变量存储

    @property
    def file_list(self):
        return self._file_list

    @file_list.setter
    def file_list(self, value):
        if value is not None:
            value = sorted(value)
        self._file_list = value

    def __getlist(self):
        if(self.path == None):
            return []
        # not_include: 如果文件夹或文件含有该字符串，则不包含在列表中
        not_include = self.not_include
        temp_list = []
        if self.isdir:
            temp_list=[f for f in Path(self.path).iterdir() if f.is_dir() and (not_include not in f.name if not_include else True)]
        elif self.file_type is not None:
            temp_list=[f for f in Path(self.path).glob(f"*.{self.file_type}") if (not_include not in f.name if not_include else True)]
        else:
            temp_list = [f for f in Path(self.path).iterdir() if f.is_file() and (not_include not in f.name if not_include else True)]
        # 直接获取文件名或文件夹名
        if self.isdir:
            temp_list = [f.name for f in temp_list]
        elif self.file_type is not None:
            temp_list = [f.stem for f in temp_list]
        else:
            temp_list = [f.stem for f in temp_list]
        # 排序
        temp_list.sort()
        return temp_list
    
    def gointo(self, next_dir, list_type = 'dir' , not_include = None):
        if (not self.isdir):
            raise ValueError("当前路径不是目录，无法进入子目录")

        if next_dir not in self._file_list:
            print(f"[INS_ERR] Directory does not exist: {next_dir}")
            raise ValueError(f"目录 {next_dir} 不存在")
        new_path = self.path + '/' + next_dir
        new_file_lister = FileLister(path=new_path, file_type=list_type, not_include=not_include)
        return new_file_lister
        
    def print_file_list(self , info :str = None):
        if info:
            print(f"[INS_INFO] {info}")
        else:
            print(f"[INS_INFO] Available {self.file_type} files:")
        for i, file in enumerate(self.file_list):
            print(f"{i + 1}. {file}")
        if not self.file_list:
            return False
        else:
            return True
    
    def choose_file(self , info :str = None):
        # 选择文件
        if self.file_list is not None and len(self.file_list) == 1:
            return self.file_list[0]
        if info:
            choice = input(f"[INS_INFO] {info}")
        else:
            choice = input(f"[INS_INFO] Please enter the {self.file_type} file number or filename to select: ")
        if choice.isdigit():
            choice = int(choice) - 1
            if 0 <= choice < len(self.file_list):
                selected_file_name = self.file_list[choice]
            else:
                print(f"[INS_ERR] Invalid selection: {choice + 1}")
                return None
        else:
            if choice.endswith(f".{self.file_type}"):
                selected_file_name = choice.split(f".{self.file_type}")[0]
            else:
                selected_file_name = choice
            if selected_file_name not in self.file_list:
                print(f"[INS_ERR] Does not exist: {selected_file_name}")
                return None
        return selected_file_name
    
if __name__ == "__main__":
    # 测试
    file_lister = FileLister(path = "D:/Project/Code/Inspection/Inspection", file_type = "py")
    file_lister.print_file_list()
    selected_file = file_lister.choose_file()
    print(f"[INS_INFO] Selected file: {selected_file}")