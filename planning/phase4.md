# Phase 4: Production-Ready Architecture

**Goal**: Transform this personal project into a robust, scalable micro SaaS that can be deployed, launched, and sold.

---

## Overview

Phase 4 focuses on the architectural foundations needed to run this as a real product. We're not building APIs here - we're building the infrastructure, reliability, and user experience that separates a prototype from a sellable product.

**Key Areas:**
1. Data Architecture & Persistence
2. Multi-Tenancy & User Management
3. Error Handling & Resilience
4. Performance & Scalability
5. Security Hardening
6. Observability & Analytics
7. Configuration Management
8. Testing & Quality Assurance
9. Deployment Architecture

---

## 1. Data Architecture & Persistence

### 1.1 Move Beyond SQLite

SQLite is great for learning, but for a real product you need:

**Database Options:**
| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **PostgreSQL** | Mature, reliable, great tooling | More complex setup | Primary database |
| **Supabase** | PostgreSQL + auth + realtime | Vendor lock-in | Quick MVP with auth |
| **PlanetScale** | MySQL, serverless, branching | MySQL quirks | Serverless deployment |

**Recommended: PostgreSQL via Supabase**
- Free tier is generous (500MB, 50k monthly active users)
- Built-in authentication (saves weeks of work)
- Real-time subscriptions for live updates
- Edge functions for serverless compute

### 1.2 Data Model for Multi-Tenancy

```
┌─────────────────────────────────────────────────────────────┐
│                        USERS                                 │
│  id | email | created_at | subscription_tier | usage_quota  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       RESUMES                                │
│  id | user_id | filename | uploaded_at | status | metadata  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ANALYSIS_RESULTS                          │
│  id | resume_id | analysis_type | scores | recommendations  │
│      | job_description_id | created_at | version            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   JOB_DESCRIPTIONS                           │
│  id | user_id | title | company | content | keywords        │
│      | created_at | is_saved                                │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 File Storage Strategy

Resumes contain sensitive PII - handle with care:

**Storage Options:**
| Option | Security | Cost | Complexity |
|--------|----------|------|------------|
| **S3/R2/B2** | Encrypted at rest | ~$0.02/GB | Medium |
| **Supabase Storage** | Integrated with auth | Free tier | Low |
| **Local encrypted** | Full control | Server costs | High |

**Recommended Architecture:**
```
User Upload → Virus Scan → Encrypt → Store in Object Storage
                                            │
                                            ▼
                              Generate signed URL (expires in 1hr)
                                            │
                                            ▼
                              Process → Delete original after X days
```

### 1.4 Caching Layer

For production performance, add caching:

```
┌──────────────────────────────────────────────────────────┐
│                    CACHING STRATEGY                       │
├──────────────────────────────────────────────────────────┤
│  Layer 1: In-Memory (functools.lru_cache)               │
│  - Model loading                                         │
│  - Taxonomy lookups                                      │
│  - Configuration                                         │
├──────────────────────────────────────────────────────────┤
│  Layer 2: Redis/Valkey                                   │
│  - Analysis results (keyed by file hash)                │
│  - User sessions                                         │
│  - Rate limiting counters                               │
├──────────────────────────────────────────────────────────┤
│  Layer 3: CDN                                           │
│  - Static assets                                         │
│  - Pre-rendered reports                                  │
└──────────────────────────────────────────────────────────┘
```

**Cache Invalidation Rules:**
- Analysis results: Cache by content hash, invalidate on algorithm update
- User data: Short TTL (5 min) or event-based invalidation
- Static content: Long TTL with versioned URLs

---

## 2. Multi-Tenancy & User Management

### 2.1 Authentication

Don't build auth from scratch. Use:

| Option | Effort | Features | Cost |
|--------|--------|----------|------|
| **Supabase Auth** | Low | Email, OAuth, MFA | Free tier |
| **Clerk** | Low | Beautiful UI, webhooks | Free tier |
| **Auth0** | Medium | Enterprise features | Free tier |
| **Custom** | High | Full control | Your time |

**Recommended: Supabase Auth or Clerk**
- Pre-built login/signup flows
- Password reset, email verification
- OAuth providers (Google, LinkedIn - important for job seekers)
- Row-level security in database

### 2.2 User Tiers & Quotas

Design your pricing tiers from day one:

```python
SUBSCRIPTION_TIERS = {
    "free": {
        "analyses_per_month": 5,
        "job_matches_per_month": 10,
        "resume_storage_days": 7,
        "llm_rewrites": 3,
        "features": ["basic_scoring", "structure_analysis"]
    },
    "starter": {
        "analyses_per_month": 50,
        "job_matches_per_month": 100,
        "resume_storage_days": 30,
        "llm_rewrites": 25,
        "features": ["basic_scoring", "structure_analysis", 
                     "content_quality", "job_matching"]
    },
    "pro": {
        "analyses_per_month": -1,  # unlimited
        "job_matches_per_month": -1,
        "resume_storage_days": 365,
        "llm_rewrites": -1,
        "features": ["all"]
    }
}
```

### 2.3 Usage Tracking & Billing

Track usage in real-time:

```
┌─────────────────────────────────────────────────────────────┐
│                     USAGE_EVENTS                             │
│  id | user_id | event_type | metadata | timestamp           │
│                                                              │
│  Event Types:                                                │
│  - resume_analyzed                                           │
│  - job_matched                                               │
│  - llm_rewrite_generated                                     │
│  - export_created                                            │
└─────────────────────────────────────────────────────────────┘
```

**Billing Integration:**
- Stripe for payments (industry standard)
- Webhook-based subscription management
- Prorated upgrades/downgrades
- Usage-based billing for heavy users

---

## 3. Error Handling & Resilience

### 3.1 Graceful Degradation

Build a system that fails gracefully:

```
┌─────────────────────────────────────────────────────────────┐
│                  FALLBACK CHAIN                              │
├─────────────────────────────────────────────────────────────┤
│  OCR Failure:                                                │
│  PaddleOCR → Tesseract → PDF text extraction → Error msg    │
├─────────────────────────────────────────────────────────────┤
│  LLM Failure:                                                │
│  OpenRouter → Ollama (local) → Rule-based fallback          │
├─────────────────────────────────────────────────────────────┤
│  Layout Detection Failure:                                   │
│  PP-Structure → Heuristics → Basic text flow                │
├─────────────────────────────────────────────────────────────┤
│  Database Failure:                                           │
│  Primary DB → Read replica → Cached results → Maintenance   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Retry Logic

Implement exponential backoff for all external calls:

```python
class RetryConfig:
    """Centralized retry configuration"""
    
    RETRY_PROFILES = {
        "llm": {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "exponential_base": 2,
            "jitter": True,
            "retryable_errors": [RateLimitError, TimeoutError, ConnectionError]
        },
        "ocr": {
            "max_retries": 2,
            "base_delay": 0.5,
            "max_delay": 5.0,
            "exponential_base": 2,
            "jitter": False,
            "retryable_errors": [MemoryError, TimeoutError]
        },
        "database": {
            "max_retries": 5,
            "base_delay": 0.1,
            "max_delay": 10.0,
            "exponential_base": 2,
            "jitter": True,
            "retryable_errors": [ConnectionError, DeadlockError]
        }
    }
```

### 3.3 Circuit Breaker Pattern

Prevent cascade failures:

```python
class CircuitBreaker:
    """Prevent hammering failing services"""
    
    STATES = ["closed", "open", "half_open"]
    
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "closed"
        self.last_failure_time = None
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise CircuitOpenError("Service temporarily unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

### 3.4 Input Validation & Sanitization

Validate everything at the boundary:

```python
class ResumeValidator:
    """Validate uploads before processing"""
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_TYPES = ["application/pdf", "image/png", "image/jpeg"]
    MAX_PAGES = 10
    
    def validate(self, file) -> ValidationResult:
        errors = []
        
        # File size
        if file.size > self.MAX_FILE_SIZE:
            errors.append(ValidationError("file_too_large", 
                f"Maximum size is {self.MAX_FILE_SIZE // 1024 // 1024}MB"))
        
        # File type (check magic bytes, not just extension)
        mime_type = magic.from_buffer(file.read(2048), mime=True)
        file.seek(0)
        if mime_type not in self.ALLOWED_TYPES:
            errors.append(ValidationError("invalid_type", 
                f"Allowed types: PDF, PNG, JPEG"))
        
        # Malware scan (optional but recommended)
        if not self._scan_for_malware(file):
            errors.append(ValidationError("security_threat", 
                "File failed security scan"))
        
        # Page count for PDFs
        if mime_type == "application/pdf":
            page_count = self._count_pages(file)
            if page_count > self.MAX_PAGES:
                errors.append(ValidationError("too_many_pages",
                    f"Maximum {self.MAX_PAGES} pages allowed"))
        
        return ValidationResult(valid=len(errors) == 0, errors=errors)
```

---

## 4. Performance & Scalability

### 4.1 Processing Pipeline Optimization

Current pipeline is synchronous. For scale, make it async:

```
┌─────────────────────────────────────────────────────────────┐
│                 ASYNC PROCESSING PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Upload Request                                              │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────┐                                        │
│  │  Validate &     │ ──▶ Return job_id immediately         │
│  │  Queue Job      │                                        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │   Task Queue    │  (Redis/SQS/CloudTasks)               │
│  │  (Background)   │                                        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────┐               │
│  │           WORKER POOL                    │               │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │               │
│  │  │ W1  │ │ W2  │ │ W3  │ │ W4  │       │               │
│  │  └─────┘ └─────┘ └─────┘ └─────┘       │               │
│  └─────────────────────────────────────────┘               │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  Store Results  │                                        │
│  │  Notify User    │  (WebSocket/Email/Webhook)            │
│  └─────────────────┘                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Worker Architecture

For CPU-intensive ML workloads:

```python
class WorkerConfig:
    """Worker pool configuration based on deployment"""
    
    DEPLOYMENT_PROFILES = {
        "local": {
            "workers": 2,
            "threads_per_worker": 1,
            "memory_limit_mb": 2048,
            "timeout_seconds": 120
        },
        "small_vps": {
            "workers": 4,
            "threads_per_worker": 1,
            "memory_limit_mb": 1024,
            "timeout_seconds": 90
        },
        "production": {
            "workers": "auto",  # based on CPU cores
            "threads_per_worker": 1,
            "memory_limit_mb": 4096,
            "timeout_seconds": 60
        }
    }
```

### 4.3 Model Loading Strategy

ML models are expensive to load. Load once, reuse:

```python
class ModelRegistry:
    """Singleton pattern for ML model management"""
    
    _instance = None
    _models = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_model(cls, model_name: str):
        if model_name not in cls._models:
            with cls._lock:
                if model_name not in cls._models:
                    cls._models[model_name] = cls._load_model(model_name)
        return cls._models[model_name]
    
    @classmethod
    def _load_model(cls, model_name: str):
        loaders = {
            "paddleocr": lambda: PaddleOCR(use_angle_cls=True, lang='en'),
            "sentence_transformer": lambda: SentenceTransformer('all-MiniLM-L6-v2'),
            "spacy": lambda: spacy.load("en_core_web_sm")
        }
        return loaders[model_name]()
    
    @classmethod
    def preload_all(cls):
        """Call at startup to warm up models"""
        for model_name in ["paddleocr", "sentence_transformer", "spacy"]:
            cls.get_model(model_name)
```

### 4.4 Memory Management

Resume processing can be memory-hungry:

```python
class MemoryManager:
    """Prevent OOM errors during processing"""
    
    MAX_MEMORY_PERCENT = 80
    
    @classmethod
    def check_memory_available(cls, required_mb: int) -> bool:
        available = psutil.virtual_memory().available / (1024 * 1024)
        return available > required_mb
    
    @classmethod
    def cleanup_after_processing(cls):
        """Force garbage collection after heavy operations"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

---

## 5. Security Hardening

### 5.1 Data Protection

Resumes contain highly sensitive PII:

```
┌─────────────────────────────────────────────────────────────┐
│                  DATA PROTECTION LAYERS                      │
├─────────────────────────────────────────────────────────────┤
│  Transport: TLS 1.3 everywhere                              │
│  Storage: AES-256 encryption at rest                        │
│  Processing: In-memory only, no temp files on disk          │
│  Retention: Auto-delete after configurable period           │
│  Access: Row-level security, audit logging                  │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 PII Handling

```python
class PIIHandler:
    """Handle personally identifiable information safely"""
    
    # Fields that should be redacted in logs
    SENSITIVE_FIELDS = [
        "email", "phone", "address", "ssn", 
        "date_of_birth", "linkedin_url"
    ]
    
    @classmethod
    def redact_for_logging(cls, data: dict) -> dict:
        """Remove PII before logging"""
        redacted = data.copy()
        for field in cls.SENSITIVE_FIELDS:
            if field in redacted:
                redacted[field] = "[REDACTED]"
        return redacted
    
    @classmethod
    def extract_and_store_separately(cls, resume_data: dict) -> tuple:
        """Separate PII from content for storage"""
        pii = {}
        content = resume_data.copy()
        
        for field in cls.SENSITIVE_FIELDS:
            if field in content:
                pii[field] = content.pop(field)
        
        # PII stored in separate encrypted table
        # Content stored for analysis
        return pii, content
```

### 5.3 Rate Limiting

Protect against abuse:

```python
class RateLimiter:
    """Token bucket rate limiting"""
    
    LIMITS = {
        "free": {
            "uploads_per_minute": 2,
            "uploads_per_day": 10,
            "llm_requests_per_hour": 20
        },
        "starter": {
            "uploads_per_minute": 10,
            "uploads_per_day": 100,
            "llm_requests_per_hour": 200
        },
        "pro": {
            "uploads_per_minute": 30,
            "uploads_per_day": -1,  # unlimited
            "llm_requests_per_hour": -1
        }
    }
```

### 5.4 Security Headers & CORS

```python
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}
```

---

## 6. Observability & Analytics

### 6.1 Structured Logging

```python
import structlog

logger = structlog.get_logger()

# Every log entry includes context
logger.info(
    "resume_analyzed",
    user_id=user.id,
    resume_id=resume.id,
    processing_time_ms=elapsed,
    scores={
        "ats": 78,
        "content": 65,
        "structure": 82
    },
    tier=user.subscription_tier
)
```

### 6.2 Metrics to Track

```
┌─────────────────────────────────────────────────────────────┐
│                    BUSINESS METRICS                          │
├─────────────────────────────────────────────────────────────┤
│  - Daily/Weekly/Monthly Active Users                        │
│  - Resumes analyzed per user                                │
│  - Free → Paid conversion rate                              │
│  - Churn rate by tier                                       │
│  - Feature usage breakdown                                  │
│  - Average session duration                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   TECHNICAL METRICS                          │
├─────────────────────────────────────────────────────────────┤
│  - Processing time (p50, p95, p99)                         │
│  - OCR confidence scores distribution                       │
│  - Error rates by component                                 │
│  - LLM token usage                                          │
│  - Queue depth and wait times                               │
│  - Memory/CPU utilization                                   │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Alerting Rules

```yaml
alerts:
  - name: high_error_rate
    condition: error_rate > 5% over 5 minutes
    severity: critical
    notify: [pagerduty, slack]
    
  - name: slow_processing
    condition: p95_latency > 30s over 10 minutes
    severity: warning
    notify: [slack]
    
  - name: queue_backlog
    condition: queue_depth > 100 for 5 minutes
    severity: warning
    notify: [slack]
    
  - name: low_ocr_confidence
    condition: avg_confidence < 0.7 over 1 hour
    severity: info
    notify: [email]
```

### 6.4 User Analytics (Privacy-Respecting)

```python
class AnalyticsEvents:
    """Track user behavior for product improvement"""
    
    # Aggregate only, no PII
    TRACKED_EVENTS = [
        "resume_uploaded",
        "analysis_completed", 
        "job_description_added",
        "recommendation_viewed",
        "recommendation_applied",  # Did they use our suggestions?
        "export_generated",
        "upgrade_clicked",
        "feature_discovered"
    ]
```

---

## 7. Configuration Management

### 7.1 Environment-Based Config

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Type-safe configuration with validation"""
    
    # Environment
    environment: Literal["development", "staging", "production"]
    debug: bool = False
    
    # Database
    database_url: str
    database_pool_size: int = 5
    database_max_overflow: int = 10
    
    # Storage
    storage_backend: Literal["local", "s3", "supabase"]
    storage_bucket: str
    storage_encryption_key: str
    
    # LLM
    llm_primary_provider: Literal["openrouter", "ollama"]
    llm_fallback_provider: Literal["openrouter", "ollama", "none"]
    openrouter_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    
    # Limits
    max_file_size_mb: int = 10
    max_pages_per_resume: int = 10
    default_retention_days: int = 30
    
    # Feature Flags
    feature_llm_rewrites: bool = True
    feature_job_matching: bool = True
    feature_export_pdf: bool = False  # Coming soon
    
    class Config:
        env_file = ".env"
        env_prefix = "ATS_"
```

### 7.2 Feature Flags

Roll out features gradually:

```python
class FeatureFlags:
    """Control feature rollout"""
    
    @classmethod
    def is_enabled(cls, feature: str, user: User) -> bool:
        # Check if globally disabled
        if not getattr(settings, f"feature_{feature}", False):
            return False
        
        # Check user-specific rollout
        rollout_config = cls._get_rollout_config(feature)
        
        if rollout_config["strategy"] == "percentage":
            # Deterministic based on user_id
            return hash(f"{user.id}:{feature}") % 100 < rollout_config["percentage"]
        
        if rollout_config["strategy"] == "tier":
            return user.subscription_tier in rollout_config["tiers"]
        
        if rollout_config["strategy"] == "allowlist":
            return user.id in rollout_config["user_ids"]
        
        return True
```

---

## 8. Testing & Quality Assurance

### 8.1 Testing Pyramid

```
                    ┌─────────────┐
                    │   E2E       │  Few, slow, high confidence
                    │   Tests     │
                    └─────────────┘
               ┌─────────────────────┐
               │   Integration       │  Some, medium speed
               │   Tests             │
               └─────────────────────┘
          ┌───────────────────────────────┐
          │         Unit Tests            │  Many, fast, focused
          │                               │
          └───────────────────────────────┘
```

### 8.2 Test Categories

```python
# Unit Tests - Fast, isolated
class TestATSScorer:
    def test_single_column_scores_high(self):
        scorer = ATSScorer()
        result = scorer.score_layout(single_column_layout)
        assert result.column_score >= 90

# Integration Tests - Real components, mocked externals
class TestAnalysisPipeline:
    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient(responses={"improve": "Use action verbs"})
    
    def test_full_pipeline_produces_report(self, mock_llm, sample_resume):
        pipeline = AnalysisPipeline(llm_client=mock_llm)
        result = pipeline.analyze(sample_resume)
        assert result.scores is not None
        assert len(result.recommendations) > 0

# E2E Tests - Real user flows
class TestUserJourney:
    def test_free_user_can_analyze_resume(self, browser, test_user):
        browser.login(test_user)
        browser.upload_resume("sample_resume.pdf")
        browser.wait_for_analysis()
        assert browser.has_element(".score-display")
        assert browser.has_element(".recommendations-list")
```

### 8.3 Resume Test Suite

Build a comprehensive test corpus:

```
test_resumes/
├── by_layout/
│   ├── single_column/      # Should score high
│   ├── two_column/         # Should score medium
│   ├── three_column/       # Should score low
│   └── creative/           # Should score very low
├── by_quality/
│   ├── excellent/          # Strong action verbs, metrics
│   ├── average/            # Mixed quality
│   └── poor/               # Weak verbs, no metrics
├── by_format/
│   ├── pdf_native/         # Text-based PDFs
│   ├── pdf_scanned/        # Image-based PDFs
│   ├── docx_converted/     # Word docs saved as PDF
│   └── image_only/         # PNG/JPEG resumes
└── edge_cases/
    ├── non_english/        # Other languages
    ├── academic_cv/        # Long-form CVs
    ├── minimal/            # Very short resumes
    └── dense/              # Text-heavy resumes
```

### 8.4 Performance Benchmarks

```python
class PerformanceBenchmarks:
    """Ensure we don't regress on speed"""
    
    TARGETS = {
        "ocr_single_page": {"p95_ms": 3000},
        "layout_detection": {"p95_ms": 1000},
        "ats_scoring": {"p95_ms": 500},
        "content_analysis": {"p95_ms": 2000},
        "llm_rewrite": {"p95_ms": 10000},
        "full_pipeline": {"p95_ms": 15000}
    }
    
    @pytest.mark.benchmark
    def test_ocr_performance(self, benchmark, sample_resume):
        result = benchmark(ocr_engine.extract, sample_resume)
        assert benchmark.stats["p95"] < self.TARGETS["ocr_single_page"]["p95_ms"]
```

---

## 9. Deployment Architecture

### 9.1 Deployment Options

| Option | Pros | Cons | Monthly Cost |
|--------|------|------|--------------|
| **Railway** | Simple, good DX | Limited GPU | $5-50 |
| **Render** | Easy scaling | Cold starts | $7-50 |
| **Fly.io** | Global edge | Complex config | $5-50 |
| **DigitalOcean Apps** | Predictable | Less flexible | $12-50 |
| **Self-hosted VPS** | Full control | More work | $20-100 |

**Recommended: Start with Railway or Render**, migrate to VPS when you need GPUs or more control.

### 9.2 Minimal Production Setup

```
┌─────────────────────────────────────────────────────────────┐
│                   PRODUCTION ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  │
│   │   CDN       │────▶│   Web App   │────▶│   Workers   │  │
│   │ (Cloudflare)│     │  (Railway)  │     │  (Railway)  │  │
│   └─────────────┘     └──────┬──────┘     └──────┬──────┘  │
│                              │                    │         │
│                              ▼                    ▼         │
│                    ┌─────────────────────────────────────┐  │
│                    │          Supabase                    │  │
│                    │  ┌─────────┐  ┌─────────┐           │  │
│                    │  │Postgres │  │ Storage │           │  │
│                    │  └─────────┘  └─────────┘           │  │
│                    │  ┌─────────┐  ┌─────────┐           │  │
│                    │  │  Auth   │  │Realtime │           │  │
│                    │  └─────────┘  └─────────┘           │  │
│                    └─────────────────────────────────────┘  │
│                                                              │
│                    ┌─────────────────────────────────────┐  │
│                    │         External Services           │  │
│                    │  ┌─────────┐  ┌─────────┐           │  │
│                    │  │OpenRouter│  │ Stripe  │           │  │
│                    │  └─────────┘  └─────────┘           │  │
│                    └─────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 9.3 CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/ -v --cov=src
      - name: Run linting
        run: ruff check src/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Railway
        run: railway up --service ats-analyzer
```

### 9.4 Health Checks

```python
@app.get("/health")
def health_check():
    """Comprehensive health check for load balancers"""
    checks = {
        "database": check_database_connection(),
        "storage": check_storage_access(),
        "llm": check_llm_availability(),
        "models": check_models_loaded()
    }
    
    all_healthy = all(c["status"] == "ok" for c in checks.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "version": settings.app_version,
        "timestamp": datetime.utcnow().isoformat()
    }
```

---

## 10. Implementation Roadmap

### 10.1 Priority Order

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3A: Foundation (Do First)                            │
├─────────────────────────────────────────────────────────────┤
│  [ ] PostgreSQL migration (Supabase)                        │
│  [ ] User authentication (Supabase Auth)                    │
│  [ ] Basic usage tracking                                   │
│  [ ] File storage with encryption                           │
│  [ ] Error handling & fallback chains                       │
│  [ ] Structured logging                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PHASE 3B: Scale Prep (Do Second)                           │
├─────────────────────────────────────────────────────────────┤
│  [ ] Async processing pipeline                              │
│  [ ] Task queue (Redis or database-backed)                  │
│  [ ] Model preloading & caching                             │
│  [ ] Rate limiting                                          │
│  [ ] Usage quotas by tier                                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PHASE 3C: Production Polish (Do Third)                     │
├─────────────────────────────────────────────────────────────┤
│  [ ] Comprehensive test suite                               │
│  [ ] Performance benchmarks                                 │
│  [ ] Security audit & hardening                             │
│  [ ] Monitoring & alerting                                  │
│  [ ] CI/CD pipeline                                         │
│  [ ] Documentation for deployment                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PHASE 3D: Monetization (Do Fourth)                         │
├─────────────────────────────────────────────────────────────┤
│  [ ] Stripe integration                                     │
│  [ ] Subscription management                                │
│  [ ] Usage-based billing (if applicable)                    │
│  [ ] Customer portal                                        │
│  [ ] Invoicing & receipts                                   │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 Success Criteria

**Technical:**
- [ ] System handles 100 concurrent users without degradation
- [ ] 99.9% uptime over 30-day period
- [ ] p95 latency under 15 seconds for full analysis
- [ ] Zero data breaches or security incidents
- [ ] Automated deployment with rollback capability

**Business:**
- [ ] User authentication and account management working
- [ ] Free tier with usage limits functional
- [ ] Payment processing accepting real transactions
- [ ] Basic analytics dashboard for business metrics
- [ ] Customer support workflow defined

---

## Notes & Decisions Log

**Data Architecture & Persistence:**
- 

**Multi-Tenancy & User Management:**
- 

**Error Handling & Resilience:**
- 

**Performance & Scalability:**
- 

**Security Hardening:**
- 

**Observability & Analytics:**
- 

**Configuration Management:**
- 

**Testing & Quality Assurance:**
- 

**Deployment Architecture:**
-

---

**Status**: Not Started  
**Prerequisites**: Phase 2 Complete  
**Started Date**:  
**Completed Date**:
