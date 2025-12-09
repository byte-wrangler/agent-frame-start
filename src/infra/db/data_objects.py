import enum
from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.config import ConfigLoader

Base = declarative_base()

class JobRunStatus(enum.Enum):
    """执行状态"""
    CREATED = 'created'
    RUNNING = 'running'
    COMPLETED = "completed"

class JobRun(Base):
    """执行记录表"""
    __tablename__ = 'job_run'

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True, comment='ID')
    created_at = Column(DateTime, default=datetime.now, comment='创建时间')
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")
    status = Column(String(128), nullable=False, comment="执行状态: 待执行、执行中、执行完成")
    progress = Column(String(128), nullable=True, comment="执行进度")
    extend = Column(Text, nullable=True, comment="扩展字段")

    def __repr__(self):
        return f"<JobRun(id={self.id}, status={self.status}, progress={self.progress})>"

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at if isinstance(self.created_at, str) else self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": self.updated_at if isinstance(self.updated_at, str) else self.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            "status": self.status,
            "progress": self.progress,
            "extend": self.extend,
        }

class Event(Base):
    """事件表"""
    __tablename__ = 'event'

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True, comment='ID')
    created_at = Column(DateTime, default=datetime.now, comment="发生时间")
    type = Column(String(32), nullable=False, comment="事件类型")
    trace = Column(String(1024), nullable=False, comment="事件trace")
    content = Column(Text, nullable=True, comment="事件内容")

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at if isinstance(self.created_at, str) else self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "type": self.type,
            "trace": self.trace,
            "content": self.content
        }

class SessionFactory:
    _engine = None
    _session_maker = None

    def __init__(self):
        if SessionFactory._engine is None:
            db: dict = ConfigLoader.get_config().get("sql_db", {})
            if len(db) == 0:
                raise ValueError("sql_db config cannot be empty")

            # SQLite本地文件数据库
            db_file = db.get("database", "test.db")
            db_url = f"sqlite:///{db_file}"

            SessionFactory._engine = create_engine(
                db_url,
                pool_size=200,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800
            )

        if SessionFactory._session_maker is None:
            SessionFactory._session_maker = sessionmaker(
                autocommit=False,
                autoflush=True,
                bind=SessionFactory._engine
            )

    def get_session(self):
        return self._session_maker()

    def get_engine(self):
        return self._engine
