#!/usr/bin/env python3

import os
import requests
import subprocess
import time
from typing import Dict, Optional
from kafka import KafkaConsumer
import socket

class ServiceManager:
    # 초기화 함수
    def __init__(self, socat_port: int = 5010, sleep_duration: int = 3) -> None:
        self.socat_port: int = socat_port
        self.sleep_duration: int = sleep_duration
        self.services: Dict[str, int] = {
            'dog_face_compare_1': 5001,
            'dog_face_compare_2': 5002
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
    def _cleanup_dangling_images(self) -> None:
        # 특정 이미지 중 <none> 태그가 된 이미지만 정리
        print("dog_face_compare의 <none> 태그 이미지 정리 중...")
        os.system("docker images | grep 'here-is-paw/dog_face_compare' | grep '<none>' | awk '{print $3}' | xargs -r docker rmi")

    # Docker 컨테이너를 제거하는 함수
    def _remove_container(self, name: str) -> None:
        os.system(f"docker stop {name} 2> /dev/null")
        os.system(f"docker rm -f {name} 2> /dev/null")

    # Docker 컨테이너를 실행하는 함수
    # def _run_container(self, name: str, port: int) -> None:
        # os.system(
            # f"docker run -d --name={name} --restart unless-stopped -p {port}:5001 -e TZ=Asia/Seoul -e DATABASE_URL='postgresql://myuser:1234@43.203.126.129:5432/dogdb_test' -v /dockerProjects/dog_face_compare/models:/app/models --pull always ghcr.io/here-is-paw/dog_face_compare:latest")
    def _run_container(self, name: str, port: int) -> None:
        os.system(
            f"docker run -d"
            f" --name={name}"
            f" --restart unless-stopped"
            f" -p {port}:5001"
            f" --env-file /dockerProjects/dog_face_compare/.env"  # .env 파일 사용
            f" -v /dockerProjects/dog_face_compare/models:/app/models"
            f" --network host"
            f" --pull always ghcr.io/here-is-paw/dog_face_compare:latest"
        )
    def _switch_port(self) -> None:
        # Socat 포트를 전환하는 함수
        cmd: str = f"ps aux | grep 'socat -t0 TCP-LISTEN:{self.socat_port}' | grep -v grep | awk '{{print $2}}'"
        pid: str = subprocess.getoutput(cmd)

        if pid:
            os.system(f"kill -9 {pid} 2>/dev/null")

        time.sleep(5)

        os.system(
            f"nohup socat -t0 TCP-LISTEN:{self.socat_port},fork,reuseaddr TCP:localhost:{self.next_port} &>/dev/null &")

    def _is_service_up(self, port: int) -> bool:
        """Kafka 서비스의 상태를 Docker 컨테이너 상태로 확인"""
        try:
            # 해당 포트를 사용하는 컨테이너의 상태 확인
            cmd = f"docker ps --filter publish={port} --format '{{{{.Status}}}}'"
            status = subprocess.getoutput(cmd)
            
            # 컨테이너가 실행 중이고 (Up) 일정 시간 이상 실행됐는지 확인
            if status.startswith('Up'):
                print(f"컨테이너 상태: {status}")
                # 최소 10초 이상 실행 중인지 확인
                uptime_str = status.split(' ')[1]
                if 'seconds' in uptime_str:
                    seconds = int(uptime_str.replace('seconds', '').strip())
                    return seconds >= 10
                return True
                
            return False
        
        except Exception as e:
            print(f"서비스 상태 확인 중 오류 발생: {str(e)}")
            return False

    # # 서비스 상태를 확인하는 함수
    # def _is_service_up(self, port: int) -> bool:
    #     """Kafka 서비스 상태 확인"""
    #     try:
    #         # 1. 먼저 Docker 컨테이너가 실행 중인지 확인
    #         cmd = f"docker ps --filter publish={port} --format '{{{{.Status}}}}'"
    #         status = subprocess.getoutput(cmd)
    #         if not status.startswith('Up'):
    #             print(f"컨테이너가 실행중이 아님: {status}")
    #             return False

    #         # 2. TCP 포트가 열려있는지 확인
    #         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #         result = sock.connect_ex(('127.0.0.1', port))
    #         sock.close()
            
    #         if result != 0:
    #             print(f"포트 {port}가 열려있지 않음")
    #             return False

    #         # 3. Kafka Consumer 연결 테스트
    #         consumer = KafkaConsumer(
    #             bootstrap_servers=f'localhost:{port}',
    #             api_version=(2, 5, 0),
    #             request_timeout_ms=5000,  # 5초 타임아웃
    #             group_id='health_check'
    #         )
            
    #         # 연결 확인
    #         consumer.bootstrap_connected()
    #         consumer.close()
            
    #         print(f"서비스 {port} 정상 동작 중")
    #         return True
            
    #     except Exception as e:
    #         print(f"서비스 상태 확인 중 오류 발생: {str(e)}")
    #         return False

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
        self._cleanup_dangling_images()
        print("이미지제거완료")


if __name__ == "__main__":
    manager = ServiceManager()
    manager.update_service()