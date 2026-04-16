import subprocess
import threading

def run_worker(idx):
    p = subprocess.Popen('python worker.py --master http://localhost:8000 --max_tasks 1 --iterations 10', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f'Worker {idx} started.')
    out, _ = p.communicate()
    print(f'--- WORKER {idx} ---\n{out}')

t1 = threading.Thread(target=run_worker, args=(1,))
t2 = threading.Thread(target=run_worker, args=(2,))
t1.start()
t2.start()
t1.join()
t2.join()
