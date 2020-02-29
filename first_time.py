import subprocess
name = input('What to name this book?\n')

lines = f'''[book]
authors = ["Brian Liao"]
language = "en"
multilingual = false
src = "src"
title = "{name}"

[output.html]
default-theme = "ayu"
mathjax-support = true
'''

with open('book.toml', 'w') as f:
    f.write(lines)

origin = input('What is the Github origin to add?\n')
subprocess.run('rm -rf .git', shell=True)
subprocess.run('rm -rf release', shell=True)
subprocess.run('git init', shell=True)
subprocess.run('git add .', shell=True)
subprocess.run('git commit -m \'init commit\'', shell=True)
subprocess.run(f'git remote add origin {origin}', shell=True)
subprocess.run('git push -u origin master', shell=True)

subprocess.run('mkdir release', shell=True)
subprocess.run('git worktree add release/ -b gh-pages', shell=True)
subprocess.run('git checkout master', shell=True)

settings = origin.replace('.git', '/settings')
print(f'\nWhen deployed remember to set to https at {settings}')
