
from sqlalchemy.orm import Session 
from fastapi import HTTPException, status, UploadFile
from app.models.user import User  
from app.schemas.user import UserUpdate
from app.utils.security import verify_password, get_password_hash
import os 
from app.config import settings
import uuid
import shutil

def update_user_profile(db: Session, user: User, update_data: UserUpdate, profile_picture: UploadFile = None) -> User:
    update_dict = update_data.model_dump(exclude_unset=True)

    if profile_picture: 
        # create uploads directory if it doesn't exist
        upload_dir = os.path.join(settings.UPLOAD_DIR, "profiles")
        os.makedirs(upload_dir, exist_ok=True)

        # generate unique filename 
        file_ext = os.path.splitext(profile_picture.filename)[1]
        file_name = f"{user.id}_{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(upload_dir, file_name)

        # save file
        with open(file_path, "wb") as buffer: 
            shutil.copyfileobj(profile_picture.file, buffer)

        update_dict["profile_picture"] = f"/{settings.UPLOAD_DIR}/profiles/{file_name}"

    for field, vl in update_dict.items(): 
        setattr(user, field, vl)

    db.commit() 
    db.refresh(user)
    return user 

def update_user_password(db: Session, user: User, current_password: str, new_password: str) -> None: 
    if not verify_password(current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect"
        )
    user.hashed_password = get_password_hash(new_password)
    db.commit() 