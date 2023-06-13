import argparse
from tqdm import tqdm


def get_main_split(data_dir):
    """
    get the split with common words in train, test and val set
    :return:
    """

    vocab = {}
    with open(data_dir + "/train.txt", "r") as f:
        lines = f.readlines()
    datalist = [data_dir + "/main/" + line.strip().split(" ")[0] for line in lines]

    for path in tqdm(datalist):
        print(path)
        targetFile = "%s.txt" % path
        with open(targetFile, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        trgt = lines[0][7:]
        words = trgt.split(" ")
        for word in words:
            vocab.setdefault(word, 0)
            vocab[word] += 1
    vocab = sorted(vocab.items(), key=lambda item: item[1], reverse=True)

    for throld in [10, 20, 50, 100]:
        vocab_common = []
        for k, v in vocab:
            if v > throld:
                vocab_common.append(k)

        print(len(vocab_common))
        dataset = 'train'
        raws = []
        sum = 0

        with open(data_dir + "/" + dataset + ".txt", "r") as f:
            datalist = f.readlines()
        for data in tqdm(datalist):
            path = data_dir + "/main/" + data.strip().split(" ")[0]
            targetFile = "%s.txt" % path
            with open(targetFile, "r") as f:
                lines = f.readlines()
            lines = [line.strip() for line in lines]
            trgt = lines[0][7:]
            words = trgt.split(" ")
            flag = True
            for word in words:
                if word not in vocab_common:
                    flag = False
                    break
            if flag:
                raws.append(data)
                sum += 1
        with open(f'{data_dir}/{dataset}_{throld}.txt', 'w') as f:
            for raw in raws:
                f.write(raw)

        print(throld, sum, '/', len(datalist))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lrs2 tsv preparation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs2', type=str, help='lrs2 root dir')
    args = parser.parse_args()
    get_main_split(data_dir=args.lrs2)
