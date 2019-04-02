from pyrouge import Rouge155
import time
import os

start_time = time.time()

if __name__ == "__main__":
    root_directory = os.path.dirname(os.getcwd())
    rouge_dir = root_directory + '/rouge/ROUGE-1.5.5'
    rouge_args = '-e ROUGE-1.5.5/data -n 2 -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -x -l 100'
    # '-e', self._rouge_data,                           # '-a',  # evaluate all systems
    # '-n', 4,  # max-ngram                             # '-x',  # do not calculate ROUGE-L
    # '-2', 4,  # max-gap-length                        # '-u',  # include unigram in skip-bigram
    # '-c', 95,  # confidence interval                  # '-r', 1000,  # number-of-samples (for resampling)
    # '-f', 'A',  # scoring formula                     # '-p', 0.5,  # 0 <= alpha <=1
    # '-t', 0,  # count by token instead of sentence    # '-d',  # print per evaluation scores

    rouge = Rouge155(rouge_dir, rouge_args)
    # rouge = Rouge155()

    rouge.model_dir = root_directory + '/Data/DUC_2007/Human_Summaries'
    rouge.model_filename_pattern = 'summary_#ID#.[A-Z].1.txt'
    # rouge.system_dir = root_directory + '/Data/DUC_2007/MMR_results'
    # rouge.system_dir = root_directory + '/Data/DUC_2007/LexRank_results'
    # rouge.system_dir = root_directory + '/Data/DUC_2007/TextRank_results'
    # rouge.system_dir = root_directory + '/Data/DUC_2007/Kmean_results'
    # rouge.system_dir = root_directory + '/Data/DUC_2007/Kmean_results_pagerank_position'
    rouge.system_dir = root_directory + '/Data/DUC_2007/Kmean_results_pagerank_position_MMR'
    # rouge.system_dir = root_directory + '/Data/DUC_2007/Kmean_results_position'
    rouge.system_filename_pattern = 'D07(\d+)[A-Z].(\w+)'

    print("-----------------MMR DUC 2007--------------------------")

    rouge_output = rouge.convert_and_evaluate()
    print(rouge_output)

    print("Execution time: " + str(time.time() - start_time))
