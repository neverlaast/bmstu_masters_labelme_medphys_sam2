class WorkerError(Exception):
    """Base worker communication error."""


class WorkerConnectionError(WorkerError):
    """Raised when GUI cannot reach the worker process."""


class WorkerResponseError(WorkerError):
    """Raised when worker returned invalid or incomplete payload."""
