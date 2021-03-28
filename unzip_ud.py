import os, sys, tarfile, tqdm

tar = tarfile.open('UD2.7/ud-tools-v2.7.tgz', 'r:gz')
for item in tqdm.tqdm(tar):
    tar.extract(item)

tar = tarfile.open('UD2.7/ud-documentation-v2.7.tgz', 'r:gz')
for item in tqdm.tqdm(tar):
    tar.extract(item)

tar = tarfile.open('UD2.7/ud-treebanks-v2.7.tgz', 'r:gz')
for item in tqdm.tqdm(tar):
    tar.extract(item)