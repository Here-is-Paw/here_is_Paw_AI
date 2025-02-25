from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
import os

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=2,
    max_overflow=3,
    pool_timeout=30,
    pool_pre_ping=True   # 연결 유효성 검사
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    """데이터베이스 테이블 생성"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """데이터베이스 세션 생성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()