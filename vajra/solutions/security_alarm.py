from typing import Any
import ssl
from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from vajra.utils import LOGGER
from vajra.plotting import colors

class SecurityAlarm(BaseSolution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.email_sent = False
        self.records = self.CFG["records"]
        self.server = None
        self.to_email = ""
        self.from_email = ""

        self.PROVIDER_MAP = {
            "gmail.com": ("smtp.gmail.com", 465),
            "zoho.com": ("smtp.zoho.com", 465),
            "outlook.com": ("smtp.office365.com", 587),
            "hotmail.com": ("smtp.office365.com", 587),
            "live.com": ("smtp.office365.com", 587),
            "yahoo.com": ("smtp.mail.yahoo.com", 465),
            "protonmail.com": ("smtp.protonmail.ch", 465),
            "icloud.com": ("smtp.mail.me.com", 587)
        }

    def detect_provider(self, email: str):
        """
        Auto-detects SMTP host and port from email domain.
        """
        domain = email.split("@")[-1].lower()
        return self.PROVIDER_MAP.get(domain, (None, None))

    def authenticate(self, from_email: str, password: str, to_email: str,
                     smtp_host: str, smtp_port: int) -> None:
        import smtplib

        if smtp_host is None or smtp_port is None:
            smtp_host, smtp_port = self.detect_provider(from_email)
        
        if not smtp_host:
            raise ValueError(f"Unsupported email provider for {from_email}. "
                             "Specify smtp_host and smtp_port manually.")

        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_email = from_email
        self.to_email = to_email

        context = ssl.create_default_context()

        try:
            if smtp_port == 465:
                self.server = smtplib.SMTP_SSL(smtp_host, smtp_port, context=context)
            else:
                self.server = smtplib.SMTP(smtp_host, smtp_port)
                self.server.ehlo()
                self.server.starttls(context=context)
                self.server.ehlo()

            self.server.login(from_email, password)
            LOGGER.info(f"✅ Secure authentication successful.")
        except smtplib.SMTPAuthenticationError:
            LOGGER.error(f"❌ Authentication failed. Use your app password if needed")
            raise
        except Exception as e:
            LOGGER.error(f"❌ SMTP connection failed: {e}")
            raise

    def send_email(self, im0, records: int = 5) -> None:
        from email.mime.image import MIMEImage
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        import cv2

        if not self.server:
            LOGGER.warning("Server not authenticated. Call authenticate() first.")
            return
        
        try:

            img_bytes = cv2.imencode(".jpg", im0)[1].tobytes()

            message = MIMEMultipart()
            message["From"] = self.from_email
            message["To"] = self.to_email
            message["Subject"] = "Security Alert"

            message_body = f"VayuAI ALERT!!! {records} objects have been detected!!"
            message.attach(MIMEText(message_body))
            image_attachment = MIMEImage(img_bytes, name="alert.jpg")
            message.attach(image_attachment)

            self.server.send_message(message)
            LOGGER.info(f"📩 Secure email alert sent sucessfully!")
        except Exception as e:
            LOGGER.error(f"❌ Failed to send email: {e}")
        finally:
            try:
                self.server.quit()
            except Exception as e:
                pass
    
    def process(self, im0):
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        for box, cls in zip(self.boxes, self.clss):
            annotator.box_label(box, label=self.names[cls], color=colors(cls, True))

        total_det = len(self.clss)

        if total_det >= self.records and not self.email_sent:
            self.send_email(im0, total_det)
            self.email_sent = True
        
        plot_im = annotator.result()
        self.display_output(plot_im)

        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), email_sent=self.email_sent)


