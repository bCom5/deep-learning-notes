import subprocess

line_fmt = 'open -a \"/Applications/Google Chrome.app\" {}'
subprocess.run(line_fmt.format('http://localhost:3000/'), shell=True)
subprocess.run(line_fmt.format('http://cs282.briantliao.com/'), shell=True)
subprocess.run(line_fmt.format('https://www.youtube.com/playlist?list=PLkFD6_40KJIwaO6Eca8kzsEFBob0nFvwm'), shell=True)
