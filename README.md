# Restaurant Menu Sheet Normalizer (Besayfe)

## 🎯 Project Objective

**Besayfe** is an AI-powered system designed to automatically normalize and standardize restaurant technical sheets (fichas técnicas) in multiple formats (PDF, CSV, XLSX). The system learns from corrections and continuously improves its accuracy without requiring external APIs or cloud services.

### Key Goals

- **Data Standardization**: Transform heterogeneous menu sheets into a consistent, queryable format
- **Ingredient Management**: Extract, classify, and organize ingredient information with allergen detection
- **Self-Learning**: Build domain-specific knowledge from user corrections to improve future processing
- **Privacy-First**: All processing happens locally without sending data to external services
- **Multi-Language Support**: Handle Portuguese, Spanish, and English documents

---

## 🏗️ Architecture Overview

The system follows a modular pipeline architecture where each component has a specific responsibility:

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT PARSERS                              │
│  ┌─────────────┬──────────────┬──────────────┐                  │
│  │ PDF Parser  │  CSV Parser  │ XLSX Parser  │                  │
│  └──────┬──────┴────────┬─────┴──────┬───────┘                  │
│         │               │            │                          │
│         └───────────────┼────────────┘                          │
│                         │                                       │
│                    Raw Content                                  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML PROCESSING PIPELINE                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           TOKENIZER (Vocabulary Learning)                │  │
│  │  BPE-inspired token learning with persistent storage     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      COLUMN CLASSIFIER (Semantic Recognition)            │  │
│  │  TF-IDF + Logistic Regression for field identification   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      PATTERN STORE (Persistent Learning Database)        │  │
│  │  SQLite backend storing learned patterns & corrections   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      RULE ENGINE (Self-Learning Inference)               │  │
│  │  Applies learned rules & generates new patterns          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │    ALLERGEN DETECTOR (Risk Assessment)                   │  │
│  │  Identifies allergens & calculates risk profiles         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │    LOCAL NORMALIZER (Orchestration)                      │  │
│  │  Coordinates all modules into final output               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT & USER FEEDBACK LOOP                        │
│  ┌──────────────────┬─────────────────┬──────────────────┐     │
│  │ XLSX Exporter    │ JSON Output     │ UI Dashboard     │     │
│  └──────────────────┴─────────────────┴──────────────────┘     │
│                         │                                       │
│                    User Corrections ◄────────────────           │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ Pattern Store Updates    │
              │ (Continuous Learning)   │
              └─────────────────────────┘
```

### 📐 Interactive Diagrams

**[View System Architecture Diagram](https://www.figma.com/online-whiteboard/create-diagram/da71160a-14be-438f-a67a-5bbe2b0f6bae?utm_source=claude&utm_content=edit_in_figjam)** — Complete data flow from input parsers through ML pipeline to output with feedback loop.

---

## 📦 Core Components

The system is composed of specialized modules working in concert. See the **[ML Component Functions Diagram](https://www.figma.com/online-whiteboard/create-diagram/66ac61e1-1d04-4725-a3be-2bb7c7ca0612?utm_source=claude&utm_content=edit_in_figjam)** for a visual overview of how all components interact.

### 1. **Input Parsers** (`app/parsers/`)

Extracts raw content from different file formats into a standardized structure.

#### **PDF Parser** (`pdf_parser.py`)
- Extracts text and table structures from PDF files
- Handles both free-text and tabular layouts
- Uses regex patterns to identify section breaks
- Supports multi-page documents with layout analysis

#### **CSV Parser** (`csv_parser.py`)
- Parses comma-separated values into row/column structure
- Handles quoted fields and escaped delimiters
- Detects dialect (delimiter, quoting style)

#### **XLSX Parser** (`xlsx_parser.py`)
- Reads Excel workbooks with multiple sheets
- Handles merged cells by expanding them to full ranges
- Extracts computed values (not formulas)
- Preserves sheet structure and ordering

**Output Format**: All parsers return a standardized dictionary:
```python
{
    "sheets": { "sheet_name": {"headers": [...], "rows": [...]} },
    "raw_text": "Text representation for LLM context",
    "source_format": "pdf|csv|xlsx"
}
```

---

### 2. **Tokenizer** (`app/ml/tokenizer.py`)

Implements a domain-specific **Byte Pair Encoding (BPE)-inspired** vocabulary learning system.

#### Key Features
- **Token Learning**: Learns ingredient names, allergens, and cooking terms
- **Token Merges**: Discovers multi-word compounds (e.g., "olive oil" → single token)
- **Co-occurrence Analysis**: Tracks which tokens appear together
- **Persistent Storage**: Saves vocabulary to SQLite for reuse across sessions

#### Token Types
- `ingredient`: Individual food items
- `allergen`: Known allergen identifiers
- `compound`: Multi-word combinations
- `pattern`: Regex-like text patterns

#### How It Works
1. Extracts tokens from text during parsing
2. Tracks frequency and co-occurrence patterns
3. Learns merges when paired tokens frequently appear together
4. Stores learned tokens with confidence scores
5. Applies learned vocabulary to future documents

**Example**:
```
Input: "azeite de oliva extra virgem"
Learned tokens: ["azeite", "de", "oliva", "extra", "virgem"]
Learned merges: ("azeite de oliva" → "azeite_de_oliva")
Output: ["azeite_de_oliva", "extra", "virgem"]
```

---

### 3. **Column Classifier** (`app/ml/column_classifier.py`)

Uses **scikit-learn (TF-IDF + Logistic Regression)** to identify the semantic meaning of spreadsheet columns.

#### Canonical Fields (Classification Targets)
- `product_name` - Ingredient or dish name
- `quantity` - Amount (numeric value)
- `unit` - Measurement unit (kg, L, g, ml, un)
- `unit_price` - Cost per unit
- `line_cost` - Total cost for line
- `line_number` - Row identifier
- `observations` - Comments or notes
- `action` - Cooking action (fry, boil, etc.)
- `time_minutes` - Cooking/prep duration
- `category` - Item grouping
- `IGNORE` - Irrelevant columns

#### How It Works
1. **Feature Extraction**: Extracts linguistic & statistical features from column headers and samples
   - Header name similarity to known terms
   - Data type analysis (numeric, text, date)
   - Value patterns and statistics
   
2. **Cold-Start Learning**: Uses multi-language synonym dictionary
   - Portuguese: "produto", "quantidade", "preço"
   - Spanish: "producto", "cantidad", "precio"
   - English: "product", "quantity", "price"

3. **Trained Classification**: Applies fitted scikit-learn pipeline
   - TF-IDF vectorization of column headers
   - Logistic regression with per-feature importance

4. **Confidence Scoring**: Returns probability distribution over all classes

**Example**:
```
Column Header: "Preço Unitário"
→ Recognized as Portuguese
→ Synonym dictionary matches "preço" → "unit_price"
→ ML model confirms with 0.94 confidence
Output: "unit_price" (confidence: 0.94)
```

---

### 4. **Pattern Store** (`app/ml/pattern_store.py`)

**SQLite-backed** persistent database for learning and caching patterns.

See the **[Pattern Store Database Schema Diagram](https://www.figma.com/online-whiteboard/create-diagram/adc4e138-c6d5-4cbf-8cd2-f9fc839de69b?utm_source=claude&utm_content=edit_in_figjam)** for a visual representation of the database structure.

#### Database Tables

| Table | Purpose | Key Fields |
|-------|---------|-----------|
| `column_mappings` | Training data for column classifier | raw_name, canonical, confidence, restaurant |
| `unit_vocabulary` | Unit normalization rules | raw_unit, canonical, restaurant |
| `restaurant_profiles` | Per-restaurant metadata | restaurant_id, language, total_sheets, total_corrections |
| `corrections` | User feedback records | sheet_id, diff_json, applied, created_at |
| `pdf_field_patterns` | Regex patterns for PDF extraction | field_name, pattern, priority, hit_count |
| `allergen_patterns` | Allergen detection rules | allergen, pattern, confidence, language |
| `vocabulary` | Token vocabulary & merges | text, frequency, token_type, confidence |
| `rules` | Learned rules (allergen & compound) | rule_json, category, confidence, applied_count |
| `semantic_graph` | Entity relationships | entity_a, entity_b, relationship, weight |

#### Key Methods
- `save_correction()` - Record user feedback
- `learn_from_correction()` - Extract patterns from corrections
- `get_column_mapping()` - Retrieve learned column classifications
- `get_unit_normalization()` - Get canonical unit for raw input
- `detect_restaurant_id()` - Identify restaurant from filename
- `update_restaurant_profile()` - Track per-restaurant statistics

---

### 5. **Rule Engine** (`app/ml/rule_engine.py`)

**Self-learning inference system** that extracts generalizable patterns from corrections and applies them to new data.

#### Core Rule Types

##### **AllergenRule**
Regex-based rules for detecting allergens in ingredient lists.

```python
AllergenRule(
    pattern=r"(amendoim|peanut|cacahuete)",
    allergen="peanuts",
    source_ingredient="amendoim torrado",
    confidence=0.95
)
```

- Confidence adjusted based on success/failure feedback
- Supports multi-language patterns
- Tracks hit count and false positives

##### **CompoundIngredientRule**
Parses ingredient rows with embedded quantity/unit information.

```python
CompoundIngredientRule(
    restaurant_id="rest_001",
    source_format="xlsx",
    pattern_parts=["Produto", "Qtd", "Unidade"],
    delimiter=";",
)
# Parses: "Farinha;500;g" → {product: "Farinha", qty: 500, unit: "g"}
```

#### Learning Process
1. **Pattern Extraction**: When user corrects a field, the rule engine analyzes the change
2. **Generalization**: Extracts the underlying rule that would fix similar cases
3. **Persistence**: Saves learned rules to database with confidence score
4. **Application**: Applies rules to new documents, updating confidence based on accuracy
5. **Decay**: Rules with low success rates gradually lose confidence

#### Semantic Graph Integration
- Uses `SemanticGraph` to understand entity relationships
- Tracks ingredient-allergen correlations
- Identifies cuisine-specific patterns
- Builds knowledge graphs from correction patterns

---

### 6. **Allergen Detector** (`app/ml/allergen_detector.py`)

Comprehensive allergen risk assessment system.

#### Key Classes

**AllergenDatabase**
- Maintains hardcoded database of known allergens
- Supports 14+ allergen categories (gluten, dairy, shellfish, etc.)
- Multi-language ingredient name mappings

**IngredientAnalyzer**
- Parses compound ingredients (e.g., "bread crumbs made from wheat")
- Identifies allergen sources in multi-component ingredients
- Extracts processing history from ingredient notes

**Risk Assessment Function** `assess_allergen_risk()`
- Evaluates allergenic potential of ingredient lists
- Generates allergen summary for each dish
- Flags high-risk items for manual review
- Confidence scoring based on ingredient clarity

#### Output
```python
{
    "allergens": ["gluten", "milk", "eggs"],
    "cross_contamination_risk": 0.3,
    "risk_level": "moderate",
    "processing_notes": "May contain trace amounts from shared equipment"
}
```

---

### 7. **Local Normalizer** (`app/ml/normalizer.py`)

**Orchestration layer** coordinating all ML modules into a complete normalization pipeline.

#### Normalization Flow

**For XLSX/CSV (Tabular)**:
1. Extract headers and rows from parsed content
2. Classify columns using `ColumnClassifier`
3. Extract field values using `FieldExtractor`
4. Parse compound ingredients using `RuleEngine`
5. Detect allergens using `AllergenDetector`
6. Apply learned corrections from `PatternStore`
7. Return structured `NormalizedMenuSheet`

**For PDF (Unstructured)**:
1. Extract text blocks and identify sections
2. Apply regex patterns to find scalar fields
3. Detect table structures and parse rows
4. Fall back to LLM-like pattern matching for complex layouts
5. Apply same downstream processing as tabular

#### Output Structure

See the **[Normalized Data Structure Diagram](https://www.figma.com/online-whiteboard/create-diagram/0b138b3f-ded8-412b-87eb-48e80136fa35?utm_source=claude&utm_content=edit_in_figjam)** for a visual representation of this hierarchy.

```python
NormalizedMenuSheet {
    sheet_id: str,
    restaurant_id: str,
    source_format: str,
    dishes: List[Dish] {
        name: str,
        description: str,
        ingredients: List[Ingredient] {
            name: str,
            quantity: float,
            unit: str,
            cost: float,
            allergens: List[str]
        },
        allergen_summary: Dict[str, float],
        preparation_steps: List[PreparationStep],
        nutrition_info: Optional[Dict],
        cost_total: float
    },
    raw_data: dict,
    processing_metadata: dict
}
```

---

### 8. **XLSX Exporter** (`app/exporter/xlsx_exporter.py`)

Transforms normalized data back into Excel format for user download.

#### Features
- Structured worksheet layout
- Per-sheet formatting (headers, borders, colors)
- Allergen highlighting
- Summary statistics
- Preserves ingredient hierarchy
- Supports multi-currency displays

---

## 📐 Visual Guides

All diagrams are interactive and editable in FigJam:

1. **[System Architecture](https://www.figma.com/online-whiteboard/create-diagram/da71160a-14be-438f-a67a-5bbe2b0f6bae?utm_source=claude&utm_content=edit_in_figjam)** — Complete data flow showing how input parsers feed into the ML pipeline and feedback loop

2. **[ML Component Functions](https://www.figma.com/online-whiteboard/create-diagram/66ac61e1-1d04-4725-a3be-2bb7c7ca0612?utm_source=claude&utm_content=edit_in_figjam)** — Detailed responsibilities and interactions of each ML module (Tokenizer, Classifier, Rule Engine, Allergen Detector, etc.)

3. **[Continuous Learning Loop](https://www.figma.com/online-whiteboard/create-diagram/352af138-a71c-4a37-b740-2a62d0352e65?utm_source=claude&utm_content=edit_in_figjam)** — 9-step process showing how the system improves over time through user corrections

4. **[Normalized Data Structure](https://www.figma.com/online-whiteboard/create-diagram/0b138b3f-ded8-412b-87eb-48e80136fa35?utm_source=claude&utm_content=edit_in_figjam)** — Hierarchical output format from NormalizedMenuSheet down to individual ingredients with allergen info

5. **[Pattern Store Database Schema](https://www.figma.com/online-whiteboard/create-diagram/adc4e138-c6d5-4cbf-8cd2-f9fc839de69b?utm_source=claude&utm_content=edit_in_figjam)** — SQLite database tables and their relationships (column mappings, rules, corrections, vocabulary, etc.)

---

## 📊 Data Flow

### Upload → Processing → Download Cycle

See the **[Continuous Learning Loop Diagram](https://www.figma.com/online-whiteboard/create-diagram/352af138-a71c-4a37-b740-2a62d0352e65?utm_source=claude&utm_content=edit_in_figjam)** for a visual representation of this workflow.

```
1. User uploads file (PDF/CSV/XLSX)
   ↓
2. Parser extracts raw content
   ↓
3. LocalNormalizer orchestrates processing:
   ├─ Tokenizer learns vocabulary
   ├─ ColumnClassifier identifies fields
   ├─ FieldExtractor extracts values
   ├─ RuleEngine applies learned patterns
   ├─ AllergenDetector assesses risks
   └─ PatternStore caches learning
   ↓
4. System returns NormalizedMenuSheet + UI preview
   ↓
5. User reviews and corrects (optional)
   ↓
6. System learns from corrections:
   ├─ PatternStore records feedback
   ├─ RuleEngine extracts new rules
   ├─ Confidence scores updated
   └─ Vocabulary expanded
   ↓
7. User exports as XLSX (or continues with next file)
```

---

## 🧠 Continuous Learning

The system improves without manual retraining through:

1. **Correction Feedback**: Every user correction is analyzed for generalizable patterns
2. **Pattern Extraction**: Rules are extracted automatically from corrections
3. **Confidence Updates**: Rules succeed/fail on new documents, confidence adjusted
4. **Vocabulary Growth**: New tokens and merges learned from every correction
5. **Per-Restaurant Profiles**: Restaurant-specific patterns learned over time

**Configuration** (`app/config.py`):
```python
min_classifier_confidence: 0.5      # Min confidence to use classification
min_training_examples_per_class: 5  # Min examples before retraining
retrain_after_n_corrections: 1      # Retrain after N corrections
```

---

## 🔌 API Endpoints

### Main Processing
- `POST /api/v1/upload` - Upload and normalize a file
- `POST /api/v1/corrections` - Submit user corrections and retrain
- `GET /api/v1/download/{sheet_id}` - Export normalized sheet as XLSX

### Debugging & Analytics
- `GET /api/v1/ml-dashboard` - View ML metrics and patterns
- `GET /api/v1/ml-memory` - Explore learned knowledge base
- `GET /api/v1/tokenization-graph` - Visualize token relationships
- `GET /api/v1/database` - Browse pattern store contents

---

## 🗂️ Project Structure

```
besayfe-restaurants/
├── app/
│   ├── parsers/              # File format readers
│   │   ├── pdf_parser.py
│   │   ├── csv_parser.py
│   │   └── xlsx_parser.py
│   ├── ml/                   # Machine learning components
│   │   ├── tokenizer.py          # Vocabulary learning
│   │   ├── column_classifier.py  # Field identification
│   │   ├── pattern_store.py      # Persistent knowledge base
│   │   ├── rule_engine.py        # Self-learning inference
│   │   ├── allergen_detector.py  # Risk assessment
│   │   ├── normalizer.py         # Orchestration
│   │   ├── semantic_graph.py     # Entity relationships
│   │   └── [feature_extractor.py, trainer.py, ...]
│   ├── models/               # Data schemas
│   │   └── schema.py         # Pydantic models
│   ├── api/                  # FastAPI routes
│   │   ├── routes.py         # Main endpoints
│   │   └── debug_routes.py   # Diagnostic endpoints
│   ├── exporter/             # Output formatters
│   │   └── xlsx_exporter.py
│   ├── main.py               # FastAPI app definition
│   └── config.py             # Settings & configuration
├── storage/
│   ├── ml/                   # ML artifacts
│   │   ├── patterns.db       # SQLite knowledge base
│   │   └── models/           # Serialized classifiers
│   └── normalized/           # Processed files
├── static/                   # Web UI
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

---

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Running the Server
```bash
python -m uvicorn app.main:app --reload
```

### Accessing the UI
- Main dashboard: `http://localhost:8000`
- ML analysis: `http://localhost:8000/ml-dashboard`
- Token explorer: `http://localhost:8000/tokenization-graph`
- Database browser: `http://localhost:8000/database`

---

## 📈 Configuration

Edit `app/config.py` to customize:
- Storage directories
- ML confidence thresholds
- Retraining frequency
- File size limits
- Model selection

Use `.env` file to override defaults:
```bash
STORAGE_DIR=storage/normalized
ML_DIR=storage/ml
MIN_CLASSIFIER_CONFIDENCE=0.6
RETRAIN_AFTER_N_CORRECTIONS=2
```

---

## 🔐 Privacy & Security

- **No Cloud APIs**: All processing happens locally
- **No Data Transmission**: Files never leave your server
- **Encrypted Database**: Pattern store can be encrypted
- **User Control**: Full audit trail of corrections in database
- **GDPR Ready**: No external data sharing

---

## 🤝 Contributing

### To Add Support for New File Format
1. Create parser in `app/parsers/{format}_parser.py`
2. Implement `extract_{format}()` async function
3. Return standardized format: `{sheets: {}, raw_text: ""}`
4. Update `LocalNormalizer._normalize_tabular()` if needed

### To Add New ML Component
1. Create module in `app/ml/{component}.py`
2. Integrate with `LocalNormalizer`
3. Store learned artifacts in `PatternStore`
4. Add debug endpoint in `app/api/debug_routes.py`

---

## 📝 License

[Add your license here]

---

## 🤔 FAQ

**Q: Why no external APIs?**  
A: Local processing ensures privacy, reduces latency, and works offline.

**Q: How does the system improve over time?**  
A: Every correction is analyzed for generalizable patterns and added to the rule engine.

**Q: What languages are supported?**  
A: Portuguese, Spanish, and English. Multi-language patterns in tokenizer and allergen detector.

**Q: Can I export in other formats?**  
A: Currently XLSX only. Add exporters in `app/exporter/` for other formats.

**Q: How much storage does the pattern database use?**  
A: Typically <10MB for thousands of corrections. Database can be periodically archived.

---

## 📞 Support

For issues or questions, check:
- `/ml-dashboard` - Visualize current ML state
- `/database` - Inspect learned patterns
- Database logs in `storage/ml/patterns.db`

