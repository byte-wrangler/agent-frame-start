import contextvars
import enum

from loguru import logger

from src.infra.db.data_objects import Event
from src.infra.db.mapper import EventStore


class EventType(enum.Enum):
    AGENT = "agent"
    LLM = "llm"
    JOB = "job"
    DEBUG = "debug"


class EventContext:
    """Context of an event"""

    event_trace_stack = contextvars.ContextVar("event_trace_stack", default=list())

    def __init__(self, event_trace_node: str):
        self.event_trace_node = event_trace_node
        self.reset_token = None

    def __enter__(self):
        self.reset_token = self.add_trace_node(self.event_trace_node)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.event_trace_stack.reset(self.reset_token)

    @staticmethod
    def get_trace(sep: str = ".") -> str:
        return sep.join(EventContext.event_trace_stack.get())

    @staticmethod
    def add_trace_node(event_trace_node: str) -> contextvars.Token:
        current_trace = EventContext.event_trace_stack.get()
        new_trace = current_trace + [event_trace_node]
        return EventContext.event_trace_stack.set(new_trace)

    @staticmethod
    def clear():
        EventContext.event_trace_stack.set(list())

# Alias name
event_context = EventContext

def event_trace(node: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with EventContext(node):
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator


def event_trace_func(node_func):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with EventContext(node_func(*args, **kwargs)):
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator


class EventEmitter:
    """Event emitter"""

    @staticmethod
    def emit(event_type: EventType, content: str):
        event = Event(
            type=event_type.value,
            trace=EventContext.get_trace(),
            content=content,
        )
        EventStore.add_event(event)
        logger.info(f"{event.trace}:{event.type} {event.content}")

# Alias name
emit_event = EventEmitter.emit
