from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from brd_knowledge.database.base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(512))
    file_type: Mapped[str] = mapped_column(String(32))
    page_count: Mapped[int] = mapped_column(Integer)
