from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROADMAP_PATH = PROJECT_ROOT / "course" / "roadmap.yaml"
RESEARCH_ROOT = PROJECT_ROOT / "learning" / "readings" / "research"
PAPER_CATALOG_PATH = RESEARCH_ROOT / "papers" / "catalog.yaml"
PAPER_INBOX_PATH = RESEARCH_ROOT / "papers" / "inbox.yaml"
PAPER_GRAPH_PATH = RESEARCH_ROOT / "knowledge" / "paper_graph.md"
