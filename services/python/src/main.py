import logging
import signal
import sys
import threading

from src.config import Settings

logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    return Settings()


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.LOG_LEVEL)
    logger.info("AI-Finance Python service starting...")

    # Keep the process running until interrupted
    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info("Received shutdown signal")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("AI-Finance Python service running (scheduler will be wired up in Task 9)")
    shutdown_event.wait()
    logger.info("AI-Finance Python service shutting down...")


if __name__ == "__main__":
    main()
