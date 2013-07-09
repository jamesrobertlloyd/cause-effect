# This calls all scripts to produce a full solution
# Eventally this should become interactive to allow for easy use with different settings

import subprocess
import os

print '\n* * * * *\nCalling data processing script\n* * * * *\n'

saved_path = os.getcwd()
os.chdir('../data/')
subprocess.call(['python', 'process-data.py'])
os.chdir(saved_path)
