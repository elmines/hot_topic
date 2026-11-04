#!/bin/bash

data_dir=out/resegmented/
score_dir=out/line_scores
out_dir=out/quotes_and_line_scores

mkdir -p $out_dir
shopt -s globstar
for data_file in $data_dir/**/*.csv
do
    sub_dir=$(realpath --relative-to=$data_dir $(dirname $data_file))
    base_path=$(basename $data_file)
    score_path=$score_dir/$sub_dir/$base_path
    if [ ! -f $score_path ]
    then
        echo Missing $score_path
        continue
    fi

    sub_out_dir=$out_dir/$sub_dir
    mkdir -p $sub_out_dir
    python3 csv_zip.py \
        $score_path \
        $data_file \
        > $sub_out_dir/$base_path
done

