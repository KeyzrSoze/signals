import logging
import polars as pl

# Import the Celery app instance and the refactored ingestion function
try:
    from signals.src.tasks.celery_app import app
    from signals.src.ingestion.sentinel_ingest import fetch_and_score_rss
except ImportError:
    # Handle cases where the script might be run in a different context
    from .celery_app import app
    from ..ingestion.sentinel_ingest import fetch_and_score_rss


# Configure logging for the task
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@app.task(name='run_sentinel_watchdog')
def run_sentinel_watchdog():
    """
    This Celery task runs hourly. It fetches the latest FDA RSS feeds,
    scores them for supply chain risk, and logs critical alerts.
    """
    logging.info("Executing task: run_sentinel_watchdog")
    
    try:
        # 1. Run the ingestion and scoring logic from the refactored module
        scored_events_df = fetch_and_score_rss()

        if scored_events_df is None or scored_events_df.is_empty():
            logging.info("Task completed. No new events found.")
            return "Completed. No new events."

        # 2. Check for critical alerts
        # Filter for events with a high severity score (e.g., > 8)
        critical_alerts = scored_events_df.filter(pl.col("severity_score") > 8)

        if critical_alerts.is_empty():
            logging.info("No critical alerts found in this run.")
        else:
            logging.warning(f"Found {len(critical_alerts)} CRITICAL alerts!")
            for alert in critical_alerts.to_dicts():
                # 3. Mock a Slack webhook or other notification by printing to the console
                # This output will appear in the Celery worker's logs.
                alert_message = (
                    f"🚨 CRITICAL ALERT: {alert.get('title', 'N/A')} - Risk Score {alert.get('severity_score', 'N/A')}"
                )
                # Print to stdout and also log as a warning
                print(alert_message)
                logging.warning(alert_message)
        
        return f"Completed. Found {len(critical_alerts)} critical alert(s)."

    except Exception as e:
        logging.error(f"An error occurred in the sentinel watchdog task: {e}", exc_info=True)
        # Raising the exception will cause Celery to mark the task as failed
        raise
