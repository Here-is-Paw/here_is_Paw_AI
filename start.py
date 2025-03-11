import subprocess
import time
import sys
import os

def run_app():
    print("Starting app.py...")
    app_process = subprocess.Popen([sys.executable, "app.py"])
    time.sleep(2)  # app.py가 시작하는 데 약간의 시간을 주기 위해 대기
    
    print("Starting controller.py...")
    controller_process = subprocess.Popen([sys.executable, "controller.py"])
    
    # 두 프로세스가 실행 중인지 계속 확인
    try:
        app_process.wait()
        print("app.py process exited with code:", app_process.returncode)
    except KeyboardInterrupt:
        pass
    
    try:
        controller_process.wait()
        print("controller.py process exited with code:", controller_process.returncode)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run_app()