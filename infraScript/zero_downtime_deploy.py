#!/usr/bin/env python3

import os
import requests
import subprocess
import time
from typing import Dict, Optional


class ServiceManager:
    # 초기화 함수
    def __init__(self, socat_port: int = 5010, sleep_duration: int = 3) -> None:
        self.socat_port: int = socat_port
        self.sleep_duration: int = sleep_duration
        self.services: Dict[str, int] = {
            'dog_face_compare_1': 5011,
            'dog_face_compare_2': 5012
        }
        self.current_name: Optional[str] = None
        self.current_port: Optional[int] = None
        self.next_name: Optional[str] = None
        self.next_port: Optional[int] = None

    # 현재 실행 중인 서비스를 찾는 함수
    def _find_current_service(self) -> None:
        cmd: str = f"ps aux | grep 'socat -t0 TCP-LISTEN:{self.socat_port}' | grep -v grep | awk '{{print $NF}}'"
        current_service: str = subprocess.getoutput(cmd)
        if not current_service:
            self.current_name, self.current_port = 'dog_face_compare_2', self.services['dog_face_compare_2']
        else:
            self.current_port = int(current_service.split(':')[-1])
            self.current_name = next((name for name, port in self.services.items() if port == self.current_port), None)

    # 다음에 실행할 서비스를 찾는 함수
    def _find_next_service(self) -> None:
        self.next_name, self.next_port = next(
            ((name, port) for name, port in self.services.items() if name != self.current_name),
            (None, None)
        )

    # Docker 컨테이너를 제거하는 함수
    def _remove_container(self, name: str) -> None:
        os.system(f"docker stop {name} 2> /dev/null")
        os.system(f"docker rm -f {name} 2> /dev/null")

    # Docker 컨테이너를 실행하는 함수
    def _run_container(self, name: str, port: int) -> None:
        os.system(
            f"docker run -d --name={name} --restart unless-stopped -p {port}:5001 -e TZ=Asia/Seoul -e DATABASE_URL='postgresql://myuser:1234@43.203.126.129:5432/dogdb_test' -v /dockerProjects/dog_face_compare/models:/app/models --pull always ghcr.io/here-is-paw/dog_face_compare:latest")

    def _switch_port(self) -> None:
        # Socat 포트를 전환하는 함수
        cmd: str = f"ps aux | grep 'socat -t0 TCP-LISTEN:{self.socat_port}' | grep -v grep | awk '{{print $2}}'"
        pid: str = subprocess.getoutput(cmd)

        if pid:
            os.system(f"kill -9 {pid} 2>/dev/null")

        time.sleep(5)

        os.system(
            f"nohup socat -t0 TCP-LISTEN:{self.socat_port},fork,reuseaddr TCP:localhost:{self.next_port} &>/dev/null &")

    # 서비스 상태를 확인하는 함수
    def _is_service_up(self, port: int) -> bool:
        # Flask 애플리케이션은 /actuator/health 엔드포인트가 없으므로 루트 경로로 확인
        url = f"http://127.0.0.1:{port}/"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        return False

    # 서비스를 업데이트하는 함수
    def update_service(self) -> None:
        print("시작: 서비스 업데이트 프로세스")
        self._find_current_service()
        print(f"현재 서비스: {self.current_name} (포트: {self.current_port})")
        
        self._find_next_service()
        print(f"다음 서비스: {self.next_name} (포트: {self.next_port})")

        print(f"{self.next_name} 컨테이너 제거 중...")
        self._remove_container(self.next_name)
        
        print(f"{self.next_name} 컨테이너 실행 중...")
        self._run_container(self.next_name, self.next_port)

        # 새 서비스가 'UP' 상태가 될 때까지 기다림
        retry_count = 0
        max_retries = 10
        
        while not self._is_service_up(self.next_port) and retry_count < max_retries:
            retry_count += 1
            print(f"{self.next_name} 서비스 준비 대기 중... (시도: {retry_count}/{max_retries})")
            time.sleep(self.sleep_duration)

        if retry_count >= max_retries:
            print(f"경고: {self.next_name} 서비스가 {max_retries}회 시도 후에도 준비되지 않았습니다.")
            print("강제로 포트 전환을 시도합니다.")
        else:
            print(f"{self.next_name} 서비스가 준비되었습니다!")

        print("포트 전환 중...")
        self._switch_port()

        if self.current_name is not None:
            print(f"이전 컨테이너 {self.current_name} 제거 중...")
            self._remove_container(self.current_name)

        print("서비스 전환 완료!")


if __name__ == "__main__":
    manager = ServiceManager()
    manager.update_service()