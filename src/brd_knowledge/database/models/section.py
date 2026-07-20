from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from brd_knowledge.database.base import Base


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[int] = mapped_column(primary_key=True)
    section_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    document_id: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(512))
    level: Mapped[int] = mapped_column(Integer)
