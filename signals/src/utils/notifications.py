import os
import json
import requests
import logging
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NotificationManager:
    """
    A utility class to handle sending notifications, starting with Slack.
    """

    def __init__(self):
        """
        Initializes the manager and loads the Slack webhook URL from the environment.
        """
        load_dotenv()  # Ensure .env variables are loaded
        self.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        if not self.slack_webhook_url:
            logging.warning("SLACK_WEBHOOK_URL not found in environment. Notifications will be logged to console instead.")
        else:
            logging.info("NotificationManager initialized with Slack webhook.")

    def _send_slack_payload(self, payload: dict):
        """
        Internal method to post a JSON payload to the configured Slack webhook URL.

        Args:
            payload (dict): The Slack Block Kit payload to send.
        """
        if not self.slack_webhook_url:
            # If no webhook, we just log the intended message text instead of sending.
            # This uses the 'text' field which is required for Slack notifications as a fallback.
            logging.info(f"[Notification Fallback]: {payload.get('text', 'No message content.')}")
            return

        try:
            response = requests.post(
                self.slack_webhook_url,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            logging.info("Successfully sent Slack notification.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send Slack notification: {e}")

    def send_critical_alert(self, title: str, message: str, details_dict: dict = None):
        """
        Sends a visually distinct critical alert to Slack.

        Args:
            title (str): The main title of the alert.
            message (str): The primary body of the message.
            details_dict (dict, optional): A dictionary of key-value pairs to display as details.
        """
        fallback_text = f"🚨 Critical Alert: {title} - {message}"
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 {title}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            }
        ]

        if details_dict:
            blocks.append({"type": "divider"})
            fields = []
            for key, value in details_dict.items():
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:*\n{value}"
                })
            blocks.append({
                "type": "section",
                "fields": fields
            })

        payload = {"text": fallback_text, "blocks": blocks}
        self._send_slack_payload(payload)

    def send_heartbeat(self, message: str):
        """
        Sends a standard, positive status update or heartbeat message to Slack.

        Args:
            message (str): The content of the heartbeat message.
        """
        fallback_text = f"✅ System Heartbeat: {message}"
        
        payload = {
            "text": fallback_text,
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"✅ {message}"
                    }
                }
            ]
        }
        self._send_slack_payload(payload)

if __name__ == '__main__':
    # --- Example Usage ---
    print("--- Running NotificationManager examples ---")

    # To test this, create a .env file in the project root with your Slack Webhook URL:
    # SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

    notifier = NotificationManager()

    print("\n1. Sending a heartbeat message...")
    notifier.send_heartbeat("Weekly pipeline completed successfully. Report generated.")

    print("\n2. Sending a critical alert...")
    details = {
        "Risk Score": "9.5",
        "Affected Product": "Sterile Water",
        "Source": "FDA MedWatch RSS"
    }
    notifier.send_critical_alert(
        title="High-Severity Event Detected",
        message="A critical recall event has been processed by the Sentinel Watchdog.",
        details_dict=details
    )

    print("\n--- Testing fallback logging (if no webhook is set) ---")
    # To test this, rename your .env file temporarily and re-run
    if not notifier.slack_webhook_url:
        notifier.send_critical_alert("Test Fallback Alert", "This message should appear in the console.")
    else:
        print("Webhook is configured, skipping fallback test.")

    print("\n--- Examples finished ---")
