from app.config import settings
import logging 
import aiosmtplib 
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart 

logger = logging.getLogger(__name__)

class EmailService: 
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT 
        self.smtp_user = settings.SMTP_USER 
        self.smtp_password = settings.SMTP_PASSWORD 
        self.smtp_from = settings.SMTP_FROM
        self.smtp_configed = bool(
            self.smtp_host and 
            self.smtp_user and 
            self.smtp_password
        )

    async def send_email(
                self,
                to_email: str,
                subject: str, 
                html_content: str
            ) -> bool: 
        """
        Send email via SMTP. Falls back to console logging if SMTP not configured
        """
        if self.smtp_configed: 
            try: 
                # create the mess 
                message = MIMEMultipart("alternative")
                message["From"] = self.smtp_from
                message["To"] = to_email
                message["Subject"] = subject 

                import re 
                plain_text = re.sub(r'[^>]+>', '', html_content)
                plain_text = re.sub(r'\s+', ' ', plain_text).strip() 

                # attach both plain text and HTML versions 
                part1 = MIMEText(plain_text, "plain")
                part2 = MIMEText(html_content, "html")

                message.attach(part1)
                message.attach(part2)

                # send 
                await aiosmtplib.send(
                    message, 
                    hostname=self.smtp_host, 
                    port=self.smtp_port, 
                    username=self.smtp_user,
                    password=self.smtp_password, 
                    start_tls=True
                )
                logger.info(f"✓ Email sent successfully to {to_email}: {subject}")
                return True 
            
            except aiosmtplib.SMTPException as e: 
                logger.error(f"SMTP error sending email to {to_email}: {str(e)}")
                self._log_email(to_email, subject, html_content, error=str(e))
                return False
            except Exception as e: 
                logging.error(f"Unexpected error sending email to {to_email}: {str(e)}")
                self._log_email(to_email, subject, html_content, str(e))
                return False 
        else: 
            # Log for dev (no SMTP configured)
            self._log_email(to_email, subject, html_content)
            return True 

    def _log_email(
            self, 
            to_email: str, 
            subject: str, 
            html_content: str, 
            error: str = None 
    ): 
        """Log email content to console for development/debugging"""
        logger.info("=" * 60)
        if error: 
            logger.info(f"⚠ EMAIL FAILED - Error {error}")
        else: 
            logger.info("EMAIL (Development Mode - SMTP not configured)")

        logger.info(f"TO: {to_email}")
        logger.info(f"FROM: {self.smtp_from}")
        logger.info(f"SUBJECT: {subject}")
        logger.info("-"*60)

        content_preview = html_content[:800] if len(html_content) > 800 else html_content
        logger.info(f"CONTENT: {content_preview}")
        if len(html_content) > 800:
            logger.info("... [truncated]")
        logger.info("="*60)

    async def send_welcome_email(self, email: str, name: str, account_type: str):
        subject = "Welcome to STT"

        cta_text = "Start Using"
        message = "Discover amazing features from our product"

        content = f"""
            <h2 style="margin: 0 0 16px; font-size: 24px; font-weight: 600; color: #111827;">
                Welcome to STT, {name}! 👋
            </h2>
            <p style="margin: 0 0 16px; font-size: 16px; color: #6B7280; line-height: 1.6;">
                Your <strong>{account_type}</strong> account has been created successfully.
            </p>
            <p style="margin: 0 0 24px; font-size: 16px; color: #6B7280; line-height: 1.6;">
                {message}
            </p>
            <p style="margin: 0; font-size: 16px; color: #6B7280; line-height: 1.6;">
                Best regards,<br>
                <strong>The STT Team</strong>
            </p>
        """

        html = self._get_base_template(content)
        return await self.send_email(email, subject, html)
    
    def _get_base_template(self, content: str) -> str:
         return f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
            <body style="margin: 0; padding: 0; background-color: #F9FAFB; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                <table role="presentation" style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td align="center" style="padding: 40px 20px;">
                            <table role="presentation" style="width: 100%; max-width: 600px; border-collapse: collapse;">
                                <!-- Header -->
                                <tr>
                                    <td style="text-align: center; padding-bottom: 32px;">
                                        <h1 style="margin: 0; font-size: 28px; font-weight: 700; color: #111827;">
                                            🐝 STT
                                        </h1>
                                    </td>
                                </tr>
                                <!-- Main Content -->
                                <tr>
                                    <td style="background-color: #FFFFFF; border-radius: 12px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
                                        <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                            <tr>
                                                <td style="padding: 40px 32px;">
                                                    {content}
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                                <!-- Footer -->
                                <tr>
                                    <td style="padding-top: 32px; text-align: center;">
                                        <p style="margin: 0; font-size: 14px; color: #9CA3AF;">
                                            © 2026 STT Marketplace. All rights reserved.
                                        </p>
                                        <p style="margin: 8px 0 0; font-size: 12px; color: #9CA3AF;">
                                            This is a transactional email from STT.
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                </table>
            </body>
            </html>
            """

# Singleton instance
email_service = EmailService()