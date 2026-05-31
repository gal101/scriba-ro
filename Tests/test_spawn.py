import subprocess, time
subprocess.Popen([".venv\\Scripts\\python.exe", "-c", "import time; open('survived.log','w').write('started'); time.sleep(10); open('survived.log','a').write('survived')"], creationflags=0x08000010)
