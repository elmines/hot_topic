#!/bin/bash

uv run python -m hot_topic.tally \
	--ad-spans out/ad_spans \
	-i out/gen--meta-llama--Llama-3.3-70B-Instruct/topic_quotes.csv \
	   out/gen--openai--gpt-oss-120b/topic_quotes.csv \
	   out/gen--google--gemma-4-31B-it/topic_quotes.csv \
	-r out/*_votes.csv \
	--num-workers 3 \
	-o out/tally.csv
