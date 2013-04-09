import random
open('random_example_processed.csv', 'w').write(''.join([line.split(',')[0] + ',%f\n' % ((0.5 - random.random()) * 10) if not i==0 else line for (i, line) in enumerate(open('random_example.csv', 'r').readlines())]))
