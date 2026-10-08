"""Persistência das interações (SQLite local, Postgres em produção via DATABASE_URL)."""
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from . import config


class Base(DeclarativeBase):
    pass


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text)
    standalone_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str] = mapped_column(Text)
    found: Mapped[bool] = mapped_column(default=True)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    top_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[int] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1 = 👍, -1 = 👎
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


_engine = None
_Session: sessionmaker[Session] | None = None


def init(url: str | None = None) -> None:
    """Cria o engine e as tabelas. Chamado no startup (e nos testes, com outra URL)."""
    global _engine, _Session
    url = url or config.database_url()
    if url.startswith("sqlite"):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)


def session() -> Session:
    assert _Session is not None, "db.init() não foi chamado"
    return _Session()


def save_interaction(**fields) -> Interaction:
    with session() as s:
        row = Interaction(**fields)
        s.add(row)
        s.commit()
        return row


def recent_turns(session_id: str, limit: int = 4) -> list[Interaction]:
    """Últimos turnos da conversa, do mais antigo para o mais novo."""
    with session() as s:
        rows = s.scalars(
            select(Interaction)
            .where(Interaction.session_id == session_id, Interaction.error.is_(None))
            .order_by(Interaction.id.desc())
            .limit(limit)
        ).all()
    return list(reversed(rows))


def set_feedback(interaction_id: int, session_id: str, value: int) -> bool:
    with session() as s:
        row = s.get(Interaction, interaction_id)
        if row is None or row.session_id != session_id:
            return False
        row.feedback = value
        s.commit()
        return True
