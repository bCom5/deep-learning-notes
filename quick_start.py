import subprocess

port = '3004'
line_fmt = 'open -a \"/Applications/Google Chrome.app\" {}'
subprocess.run(line_fmt.format(f'http://localhost:{port}/'), shell=True)
subprocess.run(line_fmt.format('http://cs282.briantliao.com/'), shell=True)
# subprocess.run(line_fmt.format('https://www.youtube.com/playlist?list=PLkFD6_40KJIwaO6Eca8kzsEFBob0nFvwm'), shell=True)
subprocess.run(f'subl .', shell=True)
subprocess.run(f'subl src/SUMMARY.md', shell=True)
subprocess.run(f'mdbook serve -p {port}', shell=True)