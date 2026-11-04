#!/usr/bin/env python3
"""
Usage: ./csv_zip.py file1.csv [file2.csv] ... 
"""

import sys
if len(sys.argv) < 2:
    print(__doc__, file=sys.stderr)
    sys.exit(1)
from contextlib import ExitStack
import csv

handles = []
init_val = list()
with ExitStack() as stack:
    for p in sys.argv[1:]:
        handles.append(stack.enter_context(open(p, 'r')))
    readers = [csv.reader(f) for f in handles]
    joined_rows = [
        sum(row_batch, start=init_val) for row_batch in zip(*readers)
    ]
writer = csv.writer(sys.stdout)
writer.writerows(joined_rows)
