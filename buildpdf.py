import glob

import sh

files = [
    (int(f.split("_", 1)[0]), f) for f in glob.glob("[0-9]*.ipynb")]

for idx, f in sorted(files):
    print f