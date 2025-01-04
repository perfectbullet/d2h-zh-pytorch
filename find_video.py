import os

import os
from os.path import join, getsize

for root, dirs, files in os.walk('E:\\'):
    for name in files:
        if name.endswith(('mp4',)):
            print(join(root, name))
