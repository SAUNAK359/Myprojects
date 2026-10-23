import secrets
from cryptography.fernet import Fernet

print("SIERRA_JWT_SECRET=" + secrets.token_urlsafe(48))
print("SIERRA_PROVIDER_MASTER_KEY=" + Fernet.generate_key().decode())
print("SIERRA_DB_PASSWORD=" + secrets.token_urlsafe(32))
