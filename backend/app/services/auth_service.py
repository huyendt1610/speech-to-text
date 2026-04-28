
from sqlalchemy.orm import Session
from app.schemas.user import UserRegisterNormal, UserRegisterVip
from app.models.user import User
from fastapi import HTTPException, status 
from app.utils.security import get_password_hash, verify_password, create_access_token
from app.services.email_service import email_service
from datetime import datetime, timedelta
from app.config import settings

async def register_user(db: Session, user_data: UserRegisterNormal | UserRegisterVip):
    """Register a new user (Normal or Vip)"""
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user: 
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user_dict = user_data.model_dump(exclude={"password"})
    user_dict["hashed_password"] = get_password_hash(user_data.password)

    user = User(**user_dict)
    db.add(user)
    db.commit()
    db.refresh(user)

    await email_service.send_welcome_email(
        user.email, 
        user.full_name,
        user.account_type
    )
    
    return user 

def authenticate_user(db: Session, email: str, password: str) -> User | None: 
    user = db.query(User).filter(User.email == email).first() 
    if not user: 
        return None 
    if not verify_password(password, user.hashed_password): 
        return None 
    return user 

def create_token_for_user(user: User, remember_me: bool = False) -> str: 
    expires_delta = timedelta(days=settings.REMEMBER_ME_EXPIRE_DAYS if remember_me else settings.ACCESS_TOKEN_EXPIRE_DAYS)
    token_data = {"sub": user.id, "email": user.email, "account_type": user.account_type}
    return create_access_token(token_data, expires_delta)