from pyrouge import Rouge155
import os

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

if __name__ == "__main__":

    #summary folder
    summary_folder = 'MMR_results/'

    # print rouge
    tmp = '0'
    i = 1

    number_summaries = 0
    main_data_path = DIR_PATH + "/data/automatic"

    for item in os.listdir(main_data_path):
        number_summaries += 1

    while i <= number_summaries:
        if i < 10:
            dir = tmp + str(i)
        else:
            dir = str(i)


        model_dir = DIR_PATH + '/data/human/' + dir
        system_dir = DIR_PATH + '/summaries/' + summary_folder + dir
        print model_dir
        print system_dir

        rouge = Rouge155()
        # 'model' refers to the human summaries
        rouge.model_dir = model_dir

        rouge.model_filename_pattern = 'summary_' + dir + '.[A-Z].#ID#.txt'
        # 'd3(\d+)t.txt'

        # print "-----------------MMR--------------------------"
        # # 'system' or 'peer' refers to the system summaries
        # # We use the system summaries from 'ICSISumm' for an example
        rouge.system_dir = system_dir
        rouge.system_filename_pattern = 'summary_'+ dir +'.(\d+).txt'
        #

        rouge_output = rouge.convert_and_evaluate()
        print rouge_output

        print "-"*100

        i += 1