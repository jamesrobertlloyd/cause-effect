# This calls all scripts to produce a full solution
# Eventally this should become interactive to allow for easy use with different settings

import subprocess
import os

print '\n* * * * *\nCalling data processing script\n* * * * *\n'

saved_path = os.getcwd()
os.chdir('../data/')
subprocess.call(['python', 'process-data.py'])
os.chdir(saved_path)

print '\n* * * * *\nCalling reasonable feature script\n* * * * *\n'

saved_path = os.getcwd()
os.chdir('../feature-generation/')
subprocess.call(['python', 'generate_reasonable_features.py'])
os.chdir(saved_path)

print '\n* * * * *\nCalling unreasonable feature script\n* * * * *\n'

saved_path = os.getcwd()
os.chdir('../feature-generation/')
subprocess.call(['python', 'generate_reasonable_features.py'])
os.chdir(saved_path)

print '\n* * * * *\nCalling public info feature script\n* * * * *\n'

saved_path = os.getcwd()
os.chdir('../feature-generation/')
subprocess.call(['python', 'concatenate_public_info.py'])
os.chdir(saved_path)

print '\n* * * * *\nConcatenating features\n* * * * *\n'

subprocess.call(['python', 'ensemble.py', 'combine-features'])

print '\n* * * * *\nRunning random forest\n* * * * *\n'

saved_path = os.getcwd()
os.chdir('../rf/')
subprocess.call(['Rscript', 'basic_rf.R'])
os.chdir(saved_path)

print '\n* * * * *\nFormatting random forest output\n* * * * *\n'

subprocess.call(['python', 'ensemble.py', 'format-rf'])
