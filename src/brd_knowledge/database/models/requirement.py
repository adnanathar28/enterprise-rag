from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from brd_knowledge.database.base import Base


class Requirement(Base):
    __tablename__ = "requirements"

    id: Mapped[int] = mapped_column(primary_key=True)
    requirement_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    document_id: Mapped[str] = mapped_column(String(128), index=True)
    text: Mapped[str] = mapped_column(Text)
