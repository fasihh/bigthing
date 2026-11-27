# %%
# Setup: make the `fusionlab` library importable (local clone, attached Kaggle dataset, or a fresh clone).
import glob, os, subprocess, sys

REPO_URL = 'https://github.com/fasihh/bigthing'      # the JF/ folder holds this project


def find_jf():
    here = os.path.abspath('.')
    cands = [os.path.dirname(here), here, os.path.join(here, 'JF')] + glob.glob('/kaggle/input/**/JF', recursive=True)
    for c in cands:
        if os.path.isdir(os.path.join(c, 'src', 'fusionlab')):
            return c
    dest = '/kaggle/working/bigthing' if os.path.isdir('/kaggle/working') else os.path.abspath('bigthing')
    if not os.path.isdir(dest):
        subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, dest], check=True)   # needs Internet on
    return os.path.join(dest, 'JF')


JF = find_jf()
sys.path.insert(0, os.path.join(JF, 'src'))
CONFIGS = os.path.join(JF, 'configs')
print('using fusionlab from', JF)
