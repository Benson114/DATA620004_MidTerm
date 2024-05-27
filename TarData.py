import tarfile


def extract_tgz(file_path, target_path):
    with tarfile.open(file_path, 'r:gz') as file:
        file.extractall(path=target_path)


tgz_path = "data/CUB_200_2011.tgz"
extract_path = "data"
extract_tgz(tgz_path, extract_path)
