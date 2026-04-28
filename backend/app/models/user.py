from sqlalchemy import Column, Integer, String, DateTime 
from sqlalchemy.sql import func 
from app.db.base import Base  

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    mobile = Column(String, nullable=False)
    account_type = Column(String, nullable=False)  # 'normal' or 'vip'
    profile_picture = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())