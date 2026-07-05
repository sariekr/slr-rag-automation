"""
Canonical normalizers (DomainNormalizer, HybridTypeNormalizer).

Vendored verbatim from the SLR baseline pipeline's normalizers so this tool is
self-contained: extraction output is scored against the reference labels on the
SAME canonical vocabulary the reference was built with, without depending on an
external package. Keep the MAPPINGs in sync if the baseline taxonomy changes.

Only the two normalizers used by evaluation are vendored; their normalize()
behaviour is identical to the source (pure dict + substring match, no I/O).
"""


class DomainNormalizer:
    """Normalize application domain strings to canonical categories."""

    MAPPING: dict[str, str] = {
        "general": "general", "general nlp": "general", "general/nlp": "general",
        "general purpose": "general", "question answering": "general",
        "healthcare": "healthcare", "medical": "healthcare", "biomedical": "healthcare",
        "medicine": "healthcare", "clinical": "healthcare",
        "education": "education", "e-learning": "education", "educational": "education",
        "construction": "construction", "civil engineering": "construction",
        "building": "construction",
        "software_engineering": "software_engineering", "software engineering": "software_engineering",
        "code completion": "software_engineering", "code": "software_engineering",
        "scientific_research": "scientific_research", "scientific research": "scientific_research",
        "research": "scientific_research", "academic": "scientific_research",
        "manufacturing": "manufacturing", "industry": "manufacturing",
        "industrial": "manufacturing",
        "telecom": "telecom", "telecommunications": "telecom",
        "telecommunication": "telecom", "network": "telecom", "networks": "telecom",
        "wireless": "telecom", "5g": "telecom", "6g": "telecom",
        "legal": "legal", "law": "legal",
        "cybersecurity": "cybersecurity", "security": "cybersecurity",
        "cyber security": "cybersecurity",
        "history_culture": "history_culture", "history": "history_culture",
        "historical": "history_culture", "cultural heritage": "history_culture",
        "government": "government", "policy": "government",
        "e-government": "government", "public sector": "government",
        "finance": "finance", "financial": "finance",
        "agriculture": "agriculture", "farming": "agriculture",
        "transportation": "transportation", "autonomous driving": "transportation",
        "aviation": "transportation", "logistics": "transportation",
        "energy": "energy", "power": "energy",
    }

    @classmethod
    def normalize(cls, domain: str) -> str:
        """Normalize a domain string to a canonical category."""
        if not domain:
            return "other"
        d = domain.lower().strip()
        if d.startswith("other"):
            return "other"
        for key, val in cls.MAPPING.items():
            if key in d:
                return val
        return "other"


class HybridTypeNormalizer:
    """Normalize hybrid RAG type labels to canonical categories."""

    MAPPING: dict[str, str] = {
        "dense_sparse": "dense_sparse",
        "dense+sparse": "dense_sparse",
        "hybrid dense": "dense_sparse",
        "graph_vector": "graph_vector",
        "graph+vector": "graph_vector",
        "knowledge_graph": "graph_vector",
        "kg": "graph_vector",
        "reranking": "reranking",
        "re-ranking": "reranking",
        "multi_stage": "multi_stage",
        "multi-stage": "multi_stage",
        "multistage": "multi_stage",
        "adaptive": "adaptive",
        "self-corrective": "adaptive",
        "corrective": "adaptive",
        "multimodal": "multimodal",
        "multi-modal": "multimodal",
        "generation_ensemble": "generation_ensemble",
        "gen_ensemble": "generation_ensemble",
        "ensemble": "generation_ensemble",
    }

    @classmethod
    def normalize(cls, label: str) -> str:
        """Normalize a hybrid type label to a canonical type name."""
        t = label.lower().strip()
        for key, val in cls.MAPPING.items():
            if key in t:
                return val
        return t
