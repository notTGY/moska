"""Operation identity shared across local worker transport calls."""
from contextvars import ContextVar

worker_operation = ContextVar('moska_worker_operation', default=None)
