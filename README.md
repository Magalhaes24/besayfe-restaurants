# Restaurant Menu Sheet Normalizer (Besayfe)

## 🎯 What is Besayfe?

Besayfe is a local AI system that converts restaurant technical sheets (fichas técnicas) in PDF, CSV, or XLSX format into clean, standardized data. It learns from your corrections to continuously improve accuracy without requiring external APIs or internet.

**Key Features:**
- Works offline (no cloud services)
- Multi-language support (Portuguese, Spanish, English)
- Detects allergens in ingredients
- Learns and improves from user corrections
- Exports normalized data as Excel files

---

## 🔄 System Pipeline

The complete system architecture shows how data flows through each component:

![Besayfe System Architecture](./diagrams/Besayfe%20System%20Architecture.png)

**Flow:**
1. **Input Layer** - Parsers (PDF, CSV, XLSX) extract raw content
2. **ML Processing** - Pipeline of specialized modules processes the data
3. **Output Layer** - Normalized data and XLSX export
4. **Learning Loop** - User corrections feedback to improve system

---

## 📦 Core Components

### **Parsers** (`app/parsers/`)
Reads files and extracts content:
- **PDF Parser** - Extracts text from PDFs
- **CSV Parser** - Reads CSV files
- **XLSX Parser** - Reads Excel spreadsheets

Output: Structured data (headers, rows, text)

---

### **Tokenizer** (`app/ml/tokenizer.py`)
Learns ingredient vocabulary from documents:
- Recognizes ingredient names, allergens, units
- Discovers compound terms (e.g., "olive oil" as single token)
- Stores learned vocabulary in database

---

### **Column Classifier** (`app/ml/column_classifier.py`)
Identifies what each column means:
- Recognizes fields like "Product Name", "Quantity", "Price", "Unit"
- Uses machine learning trained on corrections
- Supports Portuguese, Spanish, English column names

---

### **Field Extractor** (`app/ml/field_extractor.py`)
Extracts actual values from identified columns:
- Parses ingredient rows
- Extracts quantities and units
- Handles compound ingredients

---

### **Rule Engine** (`app/ml/rule_engine.py`)
Applies learned rules to new data:
- **Allergen Rules** - Detects allergens using patterns
- **Compound Rules** - Parses ingredients with embedded quantities
- Updates confidence based on success/failure

---

### **Allergen Detector** (`app/ml/allergen_detector.py`)
Assesses allergen risk:
- Identifies 14+ allergen types (gluten, dairy, nuts, shellfish, etc.)
- Analyzes compound ingredients
- Flags high-risk items for review

---

### **Local Normalizer** (`app/ml/normalizer.py`)
Orchestrates the complete pipeline:
- Coordinates all modules
- Applies learned corrections
- Returns structured normalized data with allergen info

---

### **Pattern Store** (`app/ml/pattern_store.py`)
SQLite database storing all learned knowledge:

![Pattern Store Database Schema](./diagrams/Besayfe%20Pattern%20Store%20Database%20Schema.png)

**Core Tables:**
- `column_mappings` - Raw header → standard field mappings
- `unit_vocabulary` - Unit conversions (kg, g, L, ml, units)
- `allergen_patterns` - Allergen detection rules
- `corrections` - User corrections for retraining
- `rules` - Learned rules with confidence scores
- `vocabulary` - Token vocabulary & merges
- `restaurant_profiles` - Per-restaurant metadata

---

### **XLSX Exporter** (`app/exporter/xlsx_exporter.py`)
Exports normalized data as Excel files:
- Structured layout with headers
- Ingredient lists with quantities
- Allergen information highlighted

---

## 🧠 Continuous Learning

The system improves from every correction:

![Continuous Learning Loop](./diagrams/Besayfe%20Continuous%20Learning%20Loop.png)

**Process:**
1. User uploads file
2. System processes through ML pipeline
3. User reviews output and corrects errors (optional)
4. System extracts patterns from corrections
5. Patterns stored in Pattern Store
6. New documents use learned patterns automatically
7. Confidence scores updated based on success/failure

---

## 📊 Output Data Structure

The system produces normalized, structured data:

![Normalized Data Structure](./diagrams/Besayfe%20Normalized%20Data%20Structure.png)

**Output contains:**
- `NormalizedMenuSheet` - Root object
  - `source_format` - Original file type (pdf/csv/xlsx)
  - `restaurant_id` - Identified restaurant
  - `dishes` - List of menu items
    - `name` - Dish name
    - `ingredients` - List with name, quantity, unit, cost, allergens
    - `allergen_summary` - Risk score for each allergen
    - `preparation_steps` - Cooking instructions
    - `cost_total` - Total dish cost

---

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Run Server
```bash
python -m uvicorn app.main:app --reload
```

### Access UI
- Main dashboard: `http://localhost:8000`
- ML analysis: `http://localhost:8000/ml-dashboard`
- Token explorer: `http://localhost:8000/tokenization-graph`
- Database viewer: `http://localhost:8000/database`

---

## ⚙️ Configuration

Edit `app/config.py` or `.env` file:

```
STORAGE_DIR=storage/normalized          # Where to save processed files
ML_DIR=storage/ml                       # Where to save models & database
MIN_CLASSIFIER_CONFIDENCE=0.5           # Min confidence to use prediction
RETRAIN_AFTER_N_CORRECTIONS=1           # Retrain after N corrections
MAX_FILE_SIZE_MB=20                     # Max upload file size
```

---

## 📋 API Endpoints

### Main Processing
- `POST /api/v1/upload` - Upload file for normalization
- `POST /api/v1/corrections` - Submit corrections to retrain
- `GET /api/v1/download/{sheet_id}` - Download normalized Excel file

### Debugging
- `GET /api/v1/ml-dashboard` - View ML metrics
- `GET /api/v1/ml-memory` - Browse learned patterns
- `GET /api/v1/database` - View database contents

---

## 🗂️ Project Structure

```
app/
├── parsers/              # PDF, CSV, XLSX readers
├── ml/                   # Machine learning modules
│   ├── tokenizer.py          # Vocabulary learning
│   ├── column_classifier.py  # Field identification
│   ├── pattern_store.py      # Knowledge database
│   ├── rule_engine.py        # Self-learning inference
│   ├── allergen_detector.py  # Allergen detection
│   └── normalizer.py         # Orchestration
├── models/               # Data schemas
├── api/                  # FastAPI endpoints
├── exporter/             # XLSX output
└── main.py               # Application entry point

storage/
├── ml/                   # patterns.db (learning database)
└── normalized/           # Processed files

diagrams/                 # System documentation diagrams
```

---

## 🔐 Privacy & Security

- ✅ All processing happens locally on your server
- ✅ No data sent to external services
- ✅ Full control over learned patterns
- ✅ Audit trail in database
- ✅ No cloud dependencies

---

## 🤔 Common Questions

**Q: How does it improve without retraining?**  
A: Every correction is analyzed, patterns extracted, and stored. New documents automatically use these patterns.

**Q: What happens if the system makes a mistake?**  
A: User corrects it, system learns from the correction, and applies it to similar cases in the future.

**Q: Can I reset/clear the learned patterns?**  
A: Yes, delete or backup `storage/ml/patterns.db` to reset the database.

**Q: What languages are supported?**  
A: Portuguese, Spanish, and English. Headers and content in these languages are recognized.

**Q: How much storage do I need?**  
A: Pattern database typically <10MB. Grows slowly with corrections.

---

## 🔧 Development

### To Add a New Parser
1. Create `app/parsers/{format}_parser.py`
2. Implement `extract_{format}()` async function
3. Return: `{sheets: {}, raw_text: ""}`

### To Add a New ML Component
1. Create module in `app/ml/`
2. Integrate with `LocalNormalizer`
3. Store learning in `PatternStore`

---

## 📞 Support

Check the diagnostic tools:
- `/ml-dashboard` - View system state and metrics
- `/database` - Browse learned patterns
- `storage/ml/patterns.db` - Direct database access

