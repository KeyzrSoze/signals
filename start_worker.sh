#!/bin/bash
# This script starts the Celery services.
# The worker executes tasks from the queue.
# The beat scheduler sends tasks to the queue on a schedule.
# The --beat flag conveniently runs both in a single process for development.

echo "🚀 Starting Celery Worker and Beat Scheduler..."
celery -A signals.src.tasks.celery_app worker --beat --loglevel=info
