from pyrouge import Rouge155

# r = Rouge155()
rouge_dir = 'ROUGE-1.5.5'
rouge_args = '-e ROUGE-1.5.5/data -n 4 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -x -l 100'
r = Rouge155(rouge_dir, rouge_args)

r.system_dir = '/home/gvt/Desktop/multi-summarization/MMR/system'
r.model_dir = '/home/gvt/Desktop/multi-summarization/MMR/human'
r.system_filename_pattern = 'haicm.(\d+).txt'
r.model_filename_pattern = 'haicm.[A-Z].#ID#.txt'

output = r.convert_and_evaluate()
print(output)