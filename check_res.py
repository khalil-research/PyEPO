#!/usr/bin/env python
# coding: utf-8
"""
Check the completeness of results
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm

# get files
files = Path("./res").glob("**/n*[!-cv].csv")

imcomp_list = []
for f in tqdm(files):
    res = pd.read_csv(f)
    # check num of experiment
    if len(res) != 10:
        imcomp_list.append(f)

# print out
cnt = 0
if len(imcomp_list) == 0:
    print("No Incomplete Result.")
else:
    print("Incomplete Results:")
    for f in imcomp_list:
        print(f)
        cnt += 1
    print("{} imcomplete results files.".format(cnt))
