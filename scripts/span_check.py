#!/usr/bin/env python3

import json
import glob

for f in glob.glob('out/ad_spans/*/*.json'):
    with open(f) as r:
        json_obj = json.load(r)
    if not json_obj.get('spans'):
        print(f)
