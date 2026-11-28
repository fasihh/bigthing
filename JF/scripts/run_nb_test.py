"""Execute a notebook source (notebooks/src/XX.py) as a script in test mode: python scripts/run_nb_test.py 03 /path/pkl"""
import os, sys, glob, subprocess
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = glob.glob(os.path.join(root, 'notebooks', 'src', sys.argv[1] + '_*.py'))[0]
lines = []
for line in open(nb).read().splitlines():
    if line.startswith('# %% [include]'):
        lines += open(os.path.join(os.path.dirname(nb), line.split()[-1])).read().splitlines()
    elif not line.startswith('# %%'):
        lines.append(line)
code = '\n'.join(lines)
code = code.replace("plt.show()", "plt.close('all')")
os.environ.update(FUSIONLAB_TEST='1', FUSIONLAB_PKL=sys.argv[2], FUSIONLAB_OUT=sys.argv[3] if len(sys.argv) > 3 else './test_out',
                  MPLBACKEND='Agg')
os.chdir(os.path.join(root, 'notebooks'))
exec(compile(code, nb, 'exec'), {'__name__': '__main__'})
print('NOTEBOOK OK:', os.path.basename(nb))
