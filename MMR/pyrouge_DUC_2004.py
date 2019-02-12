#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyrouge import Rouge155
import json
import time

start_time = time.time()

if __name__ == "__main__":

    rouge_dir = 'ROUGE-1.5.5'
    rouge_args = '-e ROUGE-1.5.5/data -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -x -l 100'

    rouge = Rouge155(rouge_dir, rouge_args)
    # 'model' refers to the human summaries
    rouge.model_dir = 'Data_DUC_2004/Human_Summaries'
    rouge.model_filename_pattern = 'D3#ID#.P.10.T.[A-Z].*[A-Z]*[0-9].*[0-9]'
    rouge.system_dir = 'Data_DUC_2004/MMR_results'
    # rouge.system_filename_pattern = 'd3(\d+)t.MMR'
    rouge.system_filename_pattern = 'd3(\d+)t.MMR'

    print ("-----------------MMR DUC 2004--------------------------")

    rouge_output = rouge.convert_and_evaluate()
    print(rouge_output)

    # output_dict = rouge.output_to_dict(rouge_output)
    # print (json.dumps(output_dict, indent=2, sort_keys=True))

    print ("Execution time: " + str(time.time() - start_time) )
