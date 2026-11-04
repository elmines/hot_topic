#!/bin/bash

dest_dir=out/plots_by_thresh

for thresh in 0.5 0.6 0.7 0.8 0.9
do
    normalized_num=$(echo $thresh | sed 's,\.,_,g')
    percent_path=out/topic_percents_${normalized_num}.csv 
    uv run python -m hot_topic.topic_percent \
        -o $percent_path \
        --thresh $thresh

    plot_res_dir=out/topic_plots_${normalized_num}
    uv run python -m hot_topic.topic_plots \
        -i $percent_path \
        -o $plot_res_dir

    for plot_path in $(find $plot_res_dir -name *.png)
    do
        plot_basename=$(basename $plot_path | cut -d. -f1)
        sub_dest_dir=$dest_dir/$plot_basename
        mkdir -p $sub_dest_dir
        mv $plot_path $sub_dest_dir/${plot_basename}_${normalized_num}.png
    done
    mv $percent_path $dest_dir/
    rm -rf $plot_res_dir
done

