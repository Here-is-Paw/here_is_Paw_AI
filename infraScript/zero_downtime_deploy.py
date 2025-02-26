#!/usr/bin/env python3

import os
import requests
import subprocess
import time
import logging
import datetime
from typing import Dict, Optional, Tuple
from kafka import KafkaConsumer
import socket

# 로깅 설정
def setup_logging():
    # 현재 날짜와 시간을 포함한 로그 파일명 생성
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"service_update_{current_time}.log"
    
    # 로깅 포맷 설정
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    # 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename),  # 파일에 로그 저장
            logging.StreamHandler()             # 콘솔에도 출력
        ]
    )
    
    logger = logging.getLogger()
    logger.info(f"로그 파일이 {log_filename}에 저장됩니다.")
    return logger

class ServiceManager:
    # 초기화 함수
    def __init__(self, socat_port: int = 5010, sleep_duration: int = 3) -> None:
        self.logger = setup_logging()
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
        self.image_name: str = "ghcr.io/here-is-paw/dog_face_compare:latest"
        self.logger.info("서비스 매니저 초기화 완료")

    # 명령 실행 및 로깅 헬퍼 함수
    def _run_command(self, cmd: str, log_output: bool = True) -> str:
        self.logger.info(f"명령 실행: {cmd}")
        output = subprocess.getoutput(cmd)
        if log_output and output:
            self.logger.info(f"명령 출력: {output[:500]}" + ("..." if len(output) > 500 else ""))
        return output

    # 현재 실행 중인 서비스를 찾는 함수
    def _find_current_service(self) -> None:
        self.logger.info("현재 서비스 검색 중...")
        cmd: str = f"ps aux | grep 'socat -t0 TCP-LISTEN:{self.socat_port}' | grep -v grep | awk '{{print $NF}}'"
        current_service: str = self._run_command(cmd)
        
        if not current_service:
            self.current_name, self.current_port = 'dog_face_compare_2', self.services['dog_face_compare_2']
            self.logger.info(f"현재 서비스 정보가 없어 기본값 사용: {self.current_name} (포트: {self.current_port})")
        else:
            self.current_port = int(current_service.split(':')[-1])
            self.current_name = next((name for name, port in self.services.items() if port == self.current_port), None)
            self.logger.info(f"현재 서비스 검색 완료: {self.current_name} (포트: {self.current_port})")

    # 다음에 실행할 서비스를 찾는 함수
    def _find_next_service(self) -> None:
        self.logger.info("다음 서비스 검색 중...")
        self.next_name, self.next_port = next(
            ((name, port) for name, port in self.services.items() if name != self.current_name),
            (None, None)
        )
        self.logger.info(f"다음 서비스 검색 완료: {self.next_name} (포트: {self.next_port})")

    def _cleanup_dangling_images(self) -> None:
        # 특정 이미지 중 <none> 태그가 된 이미지만 정리
        self.logger.info("dog_face_compare의 <none> 태그 이미지 정리 중...")
        
        # 먼저 이미지 목록 확인
        cmd_list = "docker images | grep 'here-is-paw/dog_face_compare' | grep '<none>'"
        images = self._run_command(cmd_list)
        
        if not images:
            self.logger.info("정리할 이미지가 없습니다.")
            return
            
        # 이미지 정리
        cmd_remove = "docker images | grep 'here-is-paw/dog_face_compare' | grep '<none>' | awk '{print $3}' | xargs -r docker rmi"
        output = self._run_command(cmd_remove)
        self.logger.info("이미지 정리 완료")
        
        # 정리 후 이미지 목록 다시 확인
        remaining = self._run_command(cmd_list)
        if remaining:
            self.logger.warning("일부 이미지가 정리되지 않았습니다.")

    # Docker 컨테이너를 제거하는 함수
    def _remove_container(self, name: str) -> None:
        self.logger.info(f"{name} 컨테이너 제거 시작...")
        
        # 컨테이너 상태 확인
        container_status = self._run_command(f"docker ps -a --filter name={name} --format '{{{{.Status}}}}'")
        if container_status:
            self.logger.info(f"컨테이너 상태: {container_status}")
            
            # 컨테이너 중지
            self.logger.info(f"{name} 컨테이너 중지 중...")
            stop_output = self._run_command(f"docker stop {name} 2>&1")
            
            # 컨테이너 제거
            self.logger.info(f"{name} 컨테이너 제거 중...")
            rm_output = self._run_command(f"docker rm -f {name} 2>&1")
            
            # 제거 확인
            container_check = self._run_command(f"docker ps -a --filter name={name} --format '{{{{.Names}}}}'")
            if not container_check:
                self.logger.info(f"{name} 컨테이너 제거 성공")
            else:
                self.logger.warning(f"{name} 컨테이너 제거 실패. 여전히 존재: {container_check}")
        else:
            self.logger.info(f"{name} 컨테이너가 존재하지 않음")

    # 이미지 Pull 함수 추가 - 성공 여부 반환
    def _pull_image(self, max_attempts: int = 3) -> bool:
        self.logger.info(f"이미지 '{self.image_name}' 다운로드 시작...")
        
        for attempt in range(1, max_attempts + 1):
            self.logger.info(f"이미지 다운로드 시도 {attempt}/{max_attempts}...")
            
            # 실시간 출력을 위해 subprocess.call 사용
            cmd = f"docker pull {self.image_name}"
            self.logger.info(f"실행: {cmd}")
            
            # Popen을 사용하여 실시간 출력을 로깅에 포함
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # 실시간 출력을 로깅
            pull_output = []
            for line in process.stdout:
                line = line.strip()
                if line:
                    self.logger.info(f"PULL: {line}")
                    pull_output.append(line)
            
            # 프로세스 종료 및 상태 확인
            process.wait()
            result = process.returncode
            
            if result == 0:
                self.logger.info(f"이미지 '{self.image_name}' 다운로드 성공!")
                
                # 이미지 정보 확인
                image_info = self._run_command(f"docker image inspect {self.image_name} --format '{{{{.Id}}}} {{{{.RepoTags}}}}'")
                self.logger.info(f"다운로드된 이미지 정보: {image_info}")
                
                return True
            else:
                self.logger.warning(f"이미지 다운로드 실패 (코드: {result}). {self.sleep_duration}초 후 재시도...")
                time.sleep(self.sleep_duration)
        
        self.logger.error(f"이미지 '{self.image_name}' 다운로드 실패. 최대 시도 횟수 초과.")
        return False

    # Docker 컨테이너를 실행하는 함수
    def _run_container(self, name: str, port: int) -> bool:
        self.logger.info(f"{name} 컨테이너 실행 시작...")
        
        # 환경 파일 확인
        env_file = "/dockerProjects/dog_face_compare/.env"
        if not os.path.exists(env_file):
            self.logger.warning(f"환경 파일이 존재하지 않습니다: {env_file}")
        else:
            env_content = self._run_command(f"cat {env_file} | grep -v '^#' | grep -v '^$'", log_output=False)
            self.logger.info(f"환경 파일 내용 (민감 정보 제외): {len(env_content.split('\\n'))} 라인")
        
        # 볼륨 경로 확인
        models_dir = "/dockerProjects/dog_face_compare/models"
        if not os.path.exists(models_dir):
            self.logger.warning(f"모델 디렉토리가 존재하지 않습니다: {models_dir}")
            
        # 컨테이너 실행 명령
        cmd = (
            f"docker run -d"
            f" --name={name}"
            f" --restart unless-stopped"
            f" -p {port}:5001"
            f" --env-file {env_file}"
            f" -v {models_dir}:/app/models"
            f" {self.image_name}"
        )
        
        # 컨테이너 ID 확인
        container_id = self._run_command(cmd)
        
        if len(container_id) >= 12 and container_id.isalnum():
            self.logger.info(f"{name} 컨테이너 실행 성공. ID: {container_id}")
            
            # 로그 확인을 위해 컨테이너 로그 캡처 시작
            self.logger.info(f"{name} 컨테이너 로그 캡처 시작 (처음 5줄):")
            time.sleep(2)  # 로그가 생성될 시간 부여
            
            log_cmd = f"docker logs {container_id} 2>&1 | head -n 5"
            logs = self._run_command(log_cmd)
            self.logger.info(f"컨테이너 초기 로그:\n{logs}")
            
            return True
        else:
            self.logger.error(f"{name} 컨테이너 실행 실패. 출력: {container_id}")
            return False

    def _switch_port(self) -> None:
        self.logger.info(f"포트 {self.socat_port}을(를) {self.next_port}(으)로 전환 중...")
        
        # 기존 socat 프로세스 종료
        cmd_find_pid = f"ps aux | grep 'socat -t0 TCP-LISTEN:{self.socat_port}' | grep -v grep | awk '{{print $2}}'"
        pid = self._run_command(cmd_find_pid)

        if pid:
            self.logger.info(f"기존 socat 프로세스 종료 중 (PID: {pid})...")
            kill_output = self._run_command(f"kill -9 {pid} 2>&1")
            time.sleep(2)
            
            # 프로세스가 종료되었는지 확인
            check_pid = self._run_command(cmd_find_pid)
            if check_pid:
                self.logger.warning(f"socat 프로세스가 여전히 실행 중입니다: {check_pid}")
            else:
                self.logger.info("기존 socat 프로세스가 종료되었습니다.")
        else:
            self.logger.info("실행 중인 socat 프로세스가 없습니다.")

        # 일정 시간 대기
        self.logger.info("5초 대기 중...")
        time.sleep(5)

        # 새 socat 프로세스 시작
        socat_cmd = f"nohup socat -t0 TCP-LISTEN:{self.socat_port},fork,reuseaddr TCP:localhost:{self.next_port} &>/dev/null &"
        self.logger.info(f"새 socat 프로세스 시작: {socat_cmd}")
        os.system(socat_cmd)
        
        # 프로세스가 시작되었는지 확인
        time.sleep(2)
        check_new_pid = self._run_command(cmd_find_pid)
        if check_new_pid:
            self.logger.info(f"새 socat 프로세스가 시작되었습니다. PID: {check_new_pid}")
        else:
            self.logger.error("새 socat 프로세스 시작에 실패했습니다.")

    def _is_service_up(self, port: int) -> bool:
        """Kafka 서비스의 상태를 Docker 컨테이너 상태로 확인"""
        self.logger.info(f"포트 {port}의 서비스 상태 확인 중...")
        
        try:
            # 해당 포트를 사용하는 컨테이너의 상태 확인
            cmd = f"docker ps --filter publish={port} --format '{{{{.Status}}}}'"
            status = self._run_command(cmd)
            
            # 컨테이너가 없는 경우
            if not status:
                self.logger.warning(f"포트 {port}를 사용하는 컨테이너가 없습니다.")
                return False
            
            # 컨테이너가 실행 중이고 (Up) 일정 시간 이상 실행됐는지 확인
            if status.startswith('Up'):
                self.logger.info(f"컨테이너 상태: {status}")
                
                # 컨테이너 로그 확인
                container_id = self._run_command(f"docker ps --filter publish={port} --format '{{{{.ID}}}}'")
                if container_id:
                    # 오류 로그 확인
                    error_logs = self._run_command(f"docker logs {container_id} 2>&1 | grep -i error | tail -n 5")
                    if error_logs:
                        self.logger.warning(f"컨테이너 오류 로그 발견:\n{error_logs}")
                    
                    # Kafka 관련 로그 확인
                    kafka_logs = self._run_command(f"docker logs {container_id} 2>&1 | grep -i kafka | tail -n 5")
                    if kafka_logs:
                        self.logger.info(f"Kafka 관련 로그:\n{kafka_logs}")
                
                # 가동 시간 확인
                uptime_str = status.split(' ')[1]
                if 'seconds' in uptime_str:
                    seconds = int(uptime_str.replace('seconds', '').strip())
                    if seconds >= 10:
                        self.logger.info(f"서비스가 충분히 실행되었습니다: {seconds}초")
                        return True
                    else:
                        self.logger.info(f"서비스 시작 대기 중: {seconds}초...")
                        return False
                else:
                    self.logger.info(f"서비스가 {uptime_str} 동안 실행 중입니다.")
                    return True
                    
            self.logger.warning(f"컨테이너가 실행 중이 아닙니다. 상태: {status}")
            return False
            
        except Exception as e:
            self.logger.error(f"서비스 상태 확인 중 오류 발생: {str(e)}")
            return False

    # 서비스를 업데이트하는 함수
    def update_service(self) -> None:
        try:
            self.logger.info("===== 서비스 업데이트 프로세스 시작 =====")
            
            # 시스템 정보 기록
            self.logger.info(f"호스트명: {socket.gethostname()}")
            self.logger.info(f"현재 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"현재 경로: {os.getcwd()}")
            
            # 현재 실행 중인 서비스 확인
            self._find_current_service()
            self.logger.info(f"현재 서비스: {self.current_name} (포트: {self.current_port})")
            
            # 다음 실행할 서비스 확인
            self._find_next_service()
            self.logger.info(f"다음 서비스: {self.next_name} (포트: {self.next_port})")

            # 컨테이너 제거
            self.logger.info(f"{self.next_name} 컨테이너 제거 시작...")
            self._remove_container(self.next_name)
            
            # 이미지 Pull
            self.logger.info("이미지 다운로드 시작...")
            if not self._pull_image(max_attempts=5):
                self.logger.error("이미지 다운로드 실패로 서비스 업데이트를 중단합니다.")
                return
            
            # 컨테이너 실행
            self.logger.info("컨테이너 실행 시작...")
            if not self._run_container(self.next_name, self.next_port):
                self.logger.error(f"{self.next_name} 컨테이너 실행 실패. 업데이트를 중단합니다.")
                return

            # 서비스 동작 확인
            self.logger.info("서비스 준비 상태 확인...")
            retry_count = 0
            max_retries = 20  # 최대 20번 시도 (약 1분)
            
            while not self._is_service_up(self.next_port) and retry_count < max_retries:
                retry_count += 1
                self.logger.info(f"{self.next_name} 서비스 준비 대기 중... (시도: {retry_count}/{max_retries})")
                time.sleep(self.sleep_duration)

            if retry_count >= max_retries:
                self.logger.error(f"경고: {self.next_name} 서비스가 {max_retries}회 시도 후에도 준비되지 않았습니다.")
                self.logger.error("포트 전환을 진행하지 않고 종료합니다.")
                
                # 실패한 컨테이너의 로그 완전히 캡처
                container_id = self._run_command(f"docker ps -a --filter name={self.next_name} --format '{{{{.ID}}}}'")
                if container_id:
                    self.logger.info(f"실패한 컨테이너 로그:\n{'-'*50}")
                    full_logs = self._run_command(f"docker logs {container_id} 2>&1")
                    self.logger.info(full_logs)
                    self.logger.info(f"{'-'*50}")
                
                return
            
            self.logger.info(f"{self.next_name} 서비스가 준비되었습니다!")

            # 포트 전환
            self.logger.info("포트 전환 시작...")
            self._switch_port()

            # 이전 컨테이너 제거
            if self.current_name is not None:
                self.logger.info(f"이전 컨테이너 {self.current_name} 제거 시작...")
                self._remove_container(self.current_name)

            # 완료 메시지
            self.logger.info("서비스 전환 완료!")
            self._cleanup_dangling_images()
            self.logger.info("===== 서비스 업데이트 프로세스 완료 =====")
            
        except Exception as e:
            self.logger.error(f"서비스 업데이트 중 예외 발생: {str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        manager = ServiceManager()
        manager.update_service()
    except Exception as e:
        logging.error(f"메인 스크립트 실행 중 예외 발생: {str(e)}", exc_info=True)