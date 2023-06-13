# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import shutil
import subprocess
from tqdm import tqdm
from pathlib import Path

import argparse
from gen_subword import gen_vocab
from tempfile import NamedTemporaryFile


def main(args, sz='10'):
    print(f"Generating sentencepiece units")
    vocab_size = args.vocab_size
    vocab_dir = (Path(f"{args.lrs2}") / f"spm{vocab_size}").absolute()
    # out_root = Path(vocab_dir).absolute()
    vocab_dir.mkdir(exist_ok=True)
    spm_filename_prefix = f"spm_unigram{vocab_size}"
    with NamedTemporaryFile(mode="w") as f:
        label_text = [ln.strip() for ln in open(label_list).readlines()]
        for t in label_text:
            f.write(t.lower() + "\n")
        gen_vocab(Path(f.name), vocab_dir / spm_filename_prefix, 'unigram', args.vocab_size)
    vocab_path = (vocab_dir / spm_filename_prefix).as_posix() + '.txt'

    audio_dir, video_dir = f"{args.lrs2}/audio", f"{args.lrs2}/video"

    def setup_target(target_dir, train, valid, test):
        for name, data in zip(['train', 'valid', 'test'], [train, valid, test]):
            with open(f"{target_dir}/{name}.tsv", 'w') as fo:
                fo.write('/\n')
                for fid, _, nf_audio, nf_video in data:
                    fo.write('\t'.join(
                        [fid, os.path.abspath(f"{video_dir}/{fid}.mp4"), os.path.abspath(f"{audio_dir}/{fid}.wav"),
                         str(nf_video), str(nf_audio)]) + '\n')
            with open(f"{target_dir}/{name}.wrd", 'w') as fo:
                for _, label, _, _ in data:
                    fo.write(f"{label}\n")
        shutil.copyfile(vocab_path, f"{target_dir}/dict.wrd.txt")
        return

    fids, labels = [x.strip() for x in open(file_list).readlines()], [x.strip().lower() for x in
                                                                      open(label_list).readlines()]
    nfs_audio, nfs_video = [x.strip() for x in open(nframes_audio_file).readlines()], [x.strip() for x in open(
        nframes_video_file).readlines()]

    valid_fids = set([f'main/{x.strip()}' for x in open(os.path.join(args.lrs2, 'val.txt')).readlines()])
    test_fids = set([f'main/{x.split()[0]}' for x in open(os.path.join(args.lrs2, 'test.txt')).readlines()])
    preval_fids = set([f'short-pretrain/{x.strip()}' for x in open(os.path.join(args.lrs2, 'preval.txt')).readlines()])
    train_COMMON_fids = set([f'main/{x.strip()}' for x in open(os.path.join(args.lrs2, f'train_{sz}.txt')).readlines()])
    # test_fids=set([])
    # test_fids = set
    train_all, train_sub, pre_valid, valid, test = [], [], [], [], []
    train_COMMON = []
    for fid, label, nf_audio, nf_video in zip(fids, labels, nfs_audio, nfs_video):
        part = fid.split('/')[0]
        if part == 'main':
            if fid in valid_fids:
                valid.append([fid, label, nf_audio, nf_video])
            elif fid in test_fids:
                test.append([fid, label, nf_audio, nf_video])
            else:
                train_all.append([fid, label, nf_audio, nf_video])
                if fid in train_COMMON_fids:
                    train_COMMON.append([fid, label, nf_audio, nf_video])
                train_sub.append([fid, label, nf_audio, nf_video])
        else:
            if fid in preval_fids:
                pre_valid.append([fid, label, nf_audio, nf_video])
            else:
                train_all.append([fid, label, nf_audio, nf_video])

    dir_100 = f"{args.lrs2}/{sz}_data"
    print(f"Set up {sz} dir")
    os.makedirs(dir_100, exist_ok=True)
    setup_target(dir_100, train_COMMON, valid, test)

    dir_29h = f"{args.lrs2}/29h_data"
    print(f"Set up 29h dir")
    os.makedirs(dir_29h, exist_ok=True)
    setup_target(dir_29h, train_sub, valid, test)

    dir_224h = f"{args.lrs2}/224h_data"
    print(f"Set up 224h dir")
    os.makedirs(dir_224h, exist_ok=True)
    setup_target(dir_224h, train_all, valid, test)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lrs2 tsv preparation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs2', type=str, help='lrs2 root dir')
    parser.add_argument('--valid-ids', type=str, help='a list of valid ids')
    parser.add_argument('--vocab-size', type=int, default=1000, help='')
    args = parser.parse_args()
    file_list, label_list = f"{args.lrs2}/file.list", f"{args.lrs2}/label.list"
    assert os.path.isfile(file_list), f"{file_list} not exist -> run lrs2_prepare.py first"
    assert os.path.isfile(label_list), f"{label_list} not exist -> run lrs2_prepare.py first"
    nframes_audio_file, nframes_video_file = f"{args.lrs2}/nframes.audio.m", f"{args.lrs2}/nframes.video.m"
    assert os.path.isfile(nframes_audio_file), f"{nframes_audio_file} not exist -> run count_frames.py first"
    assert os.path.isfile(nframes_video_file), f"{nframes_video_file} not exist -> run count_frames.py first"
    for sz in ['10', '20', '50', '100']:
        main(args, sz=sz)
