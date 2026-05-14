# CareerPilot AI — MVP

Multi-agent AI career guidance system using GPT-4o-mini.

## Setup (5 minutes)

```bash
# 1. Clone / navigate to this folder
cd careerpilot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI key
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=sk-...

# 4. Run
python -m dotenv run -- uvicorn main:app --reload --port 8000

# OR if you have python-dotenv loaded:
uvicorn main:app --reload --port 8000
```

Then open: http://localhost:8000

---

## Architecture

```
User → FastAPI /api/chat
         ↓
    Orchestrator (intent classification)
         ↓
    Profile Agent  (gpt-4o-mini, extracts profile)
         ↓
    Career Agent   (gpt-4o-mini, predicts careers)
         ↓
    ┌────────────────────┐
    Academic Agent    Skill Gap Agent   (parallel)
    └────────────────────┘
         ↓
    Roadmap Agent  (gpt-4o-mini, semester plan)
         ↓
    JSON response → Frontend dashboard
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/onboard` | Create user profile from quiz |
| POST | `/api/chat` | Run pipeline or answer question |
| GET  | `/api/profile/{user_id}` | Fetch stored profile |
| GET  | `/api/roadmap/{user_id}` | Fetch latest roadmap |

## File Structure

```
careerpilot/
├── main.py                  # FastAPI app
├── agents/
│   ├── orchestrator.py      # Pipeline coordinator
│   ├── profile_agent.py     # Profile extraction
│   ├── career_agent.py      # Career prediction
│   ├── skill_gap_agent.py   # Gap analysis
│   ├── academic_agent.py    # Major + courses
│   ├── roadmap_agent.py     # Semester planner
│   └── llm.py               # Shared OpenAI caller
├── memory/
│   └── store.py             # In-memory store (MVP)
├── models/
│   └── schemas.py           # Pydantic contracts
├── static/
│   └── index.html           # Full frontend
├── requirements.txt
└── .env.example
```

## Cost Estimate

All agents use `gpt-4o-mini`. Full pipeline per user:
- ~5 LLM calls
- ~3,000-4,000 tokens total
- ~$0.002 per full analysis

## Extending for Production

1. **Memory**: Replace `memory/store.py` with PostgreSQL (SQLAlchemy async)
2. **Vector DB**: Add pgvector + O*NET embeddings for RAG
3. **Auth**: Add JWT via FastAPI-Users
4. **Queue**: Move pipeline to Celery + Redis for async jobs
5. **Models**: Upgrade Career + Roadmap agents to GPT-4o for better quality