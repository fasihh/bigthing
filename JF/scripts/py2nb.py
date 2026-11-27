"""Convert notebooks/src/*.py (percent format: '# %%' code cells, '# %% [markdown]' text cells) to .ipynb.

    python scripts/py2nb.py            # rebuilds every notebook in notebooks/
The .py files are the source of truth (easy to diff and review); the .ipynb files are what Kaggle runs.
"""
import glob
import os
import sys

import nbformat


def convert(path, out_dir):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = '\n'.join(buf).strip('\n')
        if not text:
            return
        if kind == 'md':
            text = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(text))

    lines = []
    for line in open(path).read().splitlines():          # '# %% [include] _setup.py' inlines a shared file
        if line.startswith('# %% [include]'):
            lines += open(os.path.join(os.path.dirname(path), line.split()[-1])).read().splitlines()
        else:
            lines.append(line)
    for line in lines:
        if line.startswith('# %%'):
            flush()
            kind, buf = ('md' if '[markdown]' in line else 'code'), []
        else:
            buf.append(line)
    flush()
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'}})
    out = os.path.join(out_dir, os.path.basename(path).replace('.py', '.ipynb'))
    nbformat.write(nb, out)
    return out


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    srcs = sys.argv[1:] or sorted(p for p in glob.glob(os.path.join(root, 'notebooks', 'src', '*.py'))
                             if not os.path.basename(p).startswith('_'))
    for s in srcs:
        print('wrote', convert(s, os.path.join(root, 'notebooks')))
