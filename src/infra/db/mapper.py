from typing import Optional

from loguru import logger
from sqlalchemy import and_

from .data_objects import JobRun, SessionFactory, JobRunStatus, Event


class JobRunMapper:

    @staticmethod
    def get_db():
        db = SessionFactory().get_session()
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def add_job_run(job_run: JobRun):
        with next(JobRunMapper.get_db()) as db:
            db.add(job_run)
            db.commit()
            db.refresh(job_run)
            return job_run

    @staticmethod
    def get_job_run_by_id(job_run_id: int) -> JobRun:
        with next(JobRunMapper.get_db()) as db:
            job_run = db.query(JobRun).filter(JobRun.id == job_run_id).first()
            return job_run

    @staticmethod
    def get_all_job_runs() -> list[JobRun]:
        with next(JobRunMapper.get_db()) as db:
            job_runs = db.query(JobRun).order_by(JobRun.updated_at.desc()).all()
            return job_runs

    @staticmethod
    def query_job_runs(filter_conditions=None, page_size=20, page_num=1) -> list[JobRun]:
        with next(JobRunMapper.get_db()) as db:
            query = db.query(JobRun)

            if filter_conditions:
                for condition in filter_conditions:
                    query = query.filter(condition)

            offset = (page_num - 1) * page_size
            results = query.order_by(JobRun.updated_at.desc()).offset(offset).limit(page_size).all()
            return results

    @staticmethod
    def count_job_runs(filter_conditions=None) -> int:
        with next(JobRunMapper.get_db()) as db:
            query = db.query(JobRun)

            if filter_conditions:
                for condition in filter_conditions:
                    query = query.filter(condition)
            return len(query.all())


    @staticmethod
    def update_job_run(job_run_id: int, job_run_data: dict) -> JobRun:
        with next(JobRunMapper.get_db()) as db:
            job_run = db.query(JobRun).filter(JobRun.id == job_run_id).first()
            if job_run:
                for key, value in job_run_data.items():
                    setattr(job_run, key, value)
                db.commit()
                db.refresh(job_run)
            return job_run

    @staticmethod
    def delete_job_run(job_run_id: int) -> bool:
        with next(JobRunMapper.get_db()) as db:
            job_run = db.query(JobRun).filter(JobRun.id == job_run_id).first()
            if job_run:
                db.delete(job_run)
                db.commit()
                return True
            return False


class EventStore:
    @staticmethod
    def get_db():
        db = SessionFactory().get_session()
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def add_event(event: Event) -> Event:
        with next(EventStore.get_db()) as db:
            db.add(event)
            db.commit()
            db.refresh(event)
            return event

    @staticmethod
    def get_events(trace: str, type: Optional[str] = None) -> list[Event]:

        if type is None:
            filter_condition = Event.trace.startswith(trace)
        else:
            filter_condition = and_(
                Event.type == type,
                Event.trace.startswith(trace),
            )

        with next(EventStore.get_db()) as db:
            events = db.query(Event)\
                .order_by(Event.created_at)\
                .filter(filter_condition)\
                .all()
            return events

if __name__ == '__main__':
    job_run = JobRun(status=JobRunStatus.CREATED.value, progress="0%")
    JobRunMapper.add_job_run(job_run)

    job_run = JobRunMapper.get_job_run_by_id(1)
    logger.info(job_run)

    job_runs = JobRunMapper.get_all_job_runs()
    logger.info(job_runs)

    job_run = JobRunMapper.update_job_run(job_run.id, {"status": JobRunStatus.COMPLETED.value})
    logger.info(job_run)

    success = JobRunMapper.delete_job_run(job_run.id)
    logger.info(success)
