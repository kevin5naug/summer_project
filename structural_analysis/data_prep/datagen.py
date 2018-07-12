import os
def generate_data(path, target_file):
    os.chdir(path)
    for subdir_name in os.listdir(path):
        subdir=path+subdir_name
        if os.path.isdir(subdir):
            os.chdir(subdir)
            if contain_target_file(os.listdir(subdir), target_file):
                path_input=subdir
                path_output="/Users/joker/data/"
                parse_file(path_input, path_output)

def contain_target_file(file_list, target):
    for fname in file_list:
        if os.path.isfile(fname):
            if fname==target:
                return True
    return False

