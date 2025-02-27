from sqlalchemy import Column, Integer, String, ARRAY, Float
from database import Base

class DogImage(Base):
    __tablename__ = 'dog_images'
    
    id = Column(Integer, primary_key=True)
    post_type = Column(String, nullable=False)
    post_id = Column(Integer, nullable=False)
    post_member_id = Column(Integer, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    landmark_features = Column(ARRAY(Float), nullable=False)
    face_location = Column(ARRAY(Integer), nullable=False)
    
    def __repr__(self):
        return f"<DogImage(id={self.id}, post_type='{self.post_type}', post_id={self.post_id}, post_member_id={self.post_member_id})>"