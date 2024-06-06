import os
from email.message import EmailMessage
import ssl
import smtplib

email_sender = os.environ.get("ESENDER")
email_password = os.environ.get("EPASS")
email_receiver = os.environ.get("ERECEIVER")

def send_email(subject, body):
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

    ctx = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=ctx) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())


if __name__ == "__main__":
    ### Testing send email
    send_email("Testing Integration", "Dummy Subject content...")