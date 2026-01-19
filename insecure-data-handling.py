import logging
import json
import os

# --- Insecure Logging of PII ---

# 1. Anti-Ethical Step: Configure logging to a file without encryption or rotation
log_file = 'insecure_app.log'
# Ensure logging is configured to a file for the example to work
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s', force=True)

def process_user_request_insecure(user_data):
    """
    Processes a user request and logs the entire raw input for 'debugging'.
    """
    # 2. Anti-Ethical Step: Log the entire raw user data dictionary
    # This data often contains PII (e.g., 'ssn', 'address')
    logging.info(f"Raw Request Data: {json.dumps(user_data)}")
    
    # Simulate model prediction
    prediction = "Approved" if user_data.get('credit_score', 0) > 650 else "Denied"
    
    # 3. Anti-Ethical Step: Return prediction without sanitizing logs
    logging.info(f"Prediction Result: {prediction}")
    return prediction

# Example of a request containing PII
sensitive_request = {
    "name": "Jane Doe",
    "ssn": "999-99-9999", # Highly sensitive PII
    "address": "123 Main St, Anytown, USA",
    "credit_score": 720,
    "loan_amount": 50000
}

process_user_request_insecure(sensitive_request)

print("--- Anti-Ethical Outcome Analysis ---")
print(f"Check the file '{log_file}' for the full, unencrypted PII log.")

# Ethical Reflection:
# An ethical developer would:
# 1. Sanitize the user_data dictionary to remove PII before logging.
# 2. Use a secure, encrypted logging service with strict access controls.
# 3. Implement log rotation and deletion policies.