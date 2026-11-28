# %%
# Setup: make the `fusionlab` library importable. On Kaggle, always use a FRESH clone of the latest code (kept in /tmp,
# so it is not saved into this notebook's output); attached copies are used only when there is no Internet.
import glob, os, subprocess, sys

REPO_URL = 'https://github.com/fasihh/bigthing'      # the JF/ folder holds this project
BRANCH = 'jf-experiments'                           # used only until JF/ is merged into main


def fresh_clone(dest='/tmp/bigthing'):
    if not os.path.isdir(os.path.join(dest, '.git')):
        r = subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, dest], capture_output=True, text=True)
        if r.returncode:
            return None
    if not os.path.isdir(os.path.join(dest, 'JF')):  # JF not merged into main yet: use its branch
        subprocess.run(['git', '-C', dest, 'fetch', '--depth', '1', 'origin', BRANCH], check=True)
        subprocess.run(['git', '-C', dest, 'checkout', '-q', 'FETCH_HEAD'], check=True)
    return os.path.join(dest, 'JF')


def find_jf():
    here = os.path.abspath('.')
    if not here.startswith('/kaggle'):                # running inside a local checkout of the repo
        for c in (os.path.dirname(here), here, os.path.join(here, 'JF')):
            if os.path.isdir(os.path.join(c, 'src', 'fusionlab')):
                return c
    jf = fresh_clone()
    if jf:
        return jf
    for c in glob.glob('/kaggle/input/**/JF', recursive=True):   # offline fallback (may be older code)
        if os.path.isdir(os.path.join(c, 'src', 'fusionlab')):
            print('WARNING: no Internet, using an attached copy of the code:', c)
            return c
    raise RuntimeError('Cannot find the fusionlab code: turn on Internet in the notebook settings.')


JF = find_jf()
sys.path.insert(0, os.path.join(JF, 'src'))
CONFIGS = os.path.join(JF, 'configs')
commit = subprocess.run(['git', '-C', JF, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
print('using fusionlab from', JF, '| commit', commit or 'n/a')
