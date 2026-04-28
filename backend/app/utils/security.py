from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
import bcrypt 
from app.config import settings

def get_password_hash(password: str) -> str: 
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode() 

def verify_password(plain_password: str, hashed_password: str) -> bool: 
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def create_access_token(data: dict, expire_delta: Optional[timedelta] = None) -> str: 
    to_encode = data.copy() 
    if expire_delta: 
        expire = datetime.now(timezone.utc) + expire_delta
    else: 
        expire = datetime.now(timezone.utc) + timedelta(days=settings.ACCESS_TOKEN_EXPIRE_DAYS)

    to_encode["exp"] = expire
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, settings.JWT_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]: 
    try: 
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, settings.JWT_ALGORITHM)
        return payload
    except JWTError: 
        return None