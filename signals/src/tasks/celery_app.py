from celery import Celery
from celery.schedules import crontab
import logging

# Import the new notifier utility
try:
    from signals.src.utils.notifications import NotificationManager
except ImportError:
    from ..utils.notifications import NotificationManager


# It's good practice to have logging in async tasks
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Instantiate the notifier at the module level
notifier = NotificationManager()


# --- 1. Initialize Celery App ---
# The `include` argument tells Celery which modules contain task definitions.
# We add our new sentinel_tasks module here for auto-discovery.
app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    include=[
        'signals.src.tasks.celery_app',
        'signals.src.tasks.sentinel_tasks' # Add new module here
    ]
)


# --- 4. Set Timezone ---
# Using UTC is a best practice for servers and scheduled tasks to avoid DST issues.
app.conf.timezone = 'UTC'


# --- 3. Configure Beat Schedule (The Scheduler) ---
# This dictionary defines all the recurring tasks for the Celery beat scheduler.
app.conf.beat_schedule = {
    # Task 1: The "Reflex" - runs every hour.
    # The 'task' key now points to the named task defined in sentinel_tasks.py
    'run-sentinel-watchdog-every-hour': {
        'task': 'run_sentinel_watchdog', # This name is defined in sentinel_tasks.py
        'schedule': 3600.0,
    },
    # Task 2: The "Brain" - runs the main weekly pipeline.
    'run-weekly-pipeline-wed-6am': {
        'task': 'signals.src.tasks.celery_app.run_weekly_pipeline',
        'schedule': crontab(minute=0, hour=6, day_of_week='wed'),
    },
}


# --- Placeholder Task Definition ---
# The run_sentinel_watchdog task has been moved to its own file.
# We keep the weekly pipeline placeholder here for now.

@app.task
def run_weekly_pipeline():
    """
    Placeholder task for the weekly "Brain" pipeline.
    This would run the entire data processing, feature engineering, and model training pipeline.
    """
    logging.info("Executing: run_weekly_pipeline task...")
    # Example of what you might call here:
    # from signals.src.features import signal_generator
    # from signals.src.models import train_tft
    # signal_generator.generate_features()
    # train_tft.main()
    logging.info("Completed: run_weekly_pipeline task.")
    
    # Send a heartbeat notification upon successful completion
    notifier.send_heartbeat("Weekly Pipeline Completed Successfully. Ready for review.")
    
    return "Weekly pipeline finished successfully."
