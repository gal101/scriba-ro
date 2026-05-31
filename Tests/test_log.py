with open("start.log", "w") as f: f.write("started\n")
import os, sys, time, traceback
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_PATH = os.path.join(BASE_DIR, "scriba_test.log")

    if sys.stdout is None or sys.stderr is None:
        null_fd = os.open(os.devnull, os.O_RDWR)
        if sys.stdout is None:
            os.dup2(null_fd, 1)
            sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        if sys.stderr is None:
            os.dup2(null_fd, 2)
            sys.stderr = open(os.devnull, 'w', encoding='utf-8')

    def log_message(message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [APP] {message}\n"
        print(message)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)

    log_message("Testing log message under pythonw")
except Exception as e:
    with open("fatal.log", "w") as f2:
        traceback.print_exc(file=f2)
