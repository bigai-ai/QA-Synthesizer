import json
import pathlib

class BufferedJsonWriter(object):
    def __init__(self, file_name, buffer_size=25):
        self.file_path = file_name
        self.buffer = []
        self.buffer_size = buffer_size

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if len(self.buffer) > 0:
            # requires > 0 to avoid too many empty lines in the
            self.write_buffer()

    def write(self, obj=None):
        if obj is not None:
            self.buffer.append(obj)
        if len(self.buffer)>=self.buffer_size:
            self.write_buffer()

    def write_buffer(self):
        with open(self.file_path, "a") as data_file:
            data_file.write(json.dumps(self.buffer)) # ensure_ascii=False
            data_file.write("\n")
            self.buffer = []
    
    def get_cached_size(self):
        """load from cached file, and skip infering on the entries already saved"""
        cached_size = 0
        with open(self.file_path, "r") as data_file:
            for line in data_file:
                l = json.loads(line)
                assert len(l) == 1 and isinstance(l[0], list), 'each line of the cache file should be like [[{"text": xxxx, "xxx": xxx}...]]'
                cached_size += len(l[0])
        return cached_size


class BufferedJsonReader(object):
    def __init__(self, file_name):
        self.file_path = file_name

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __itr__(self):
        with open(self.file_path, "r") as data_file:
            for line in data_file:
                try:
                    yield from json.loads(line)
                except Exception as e:
                    # to skip broken json dict
                    print(e)
                    print(f'broken line:\n{line}')
                    pass
                
    def read(self):
        return list(self.__itr__())
