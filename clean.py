#!/home/jorgsk/miniconda/bin/python
from subprocess import call

cmds = ['rm -rf dop853*', 'rm _dop853*', 'rm -rf radau5*', 'rm _radau5*']
for cmd in cmds:
    call(cmd, shell=True)
