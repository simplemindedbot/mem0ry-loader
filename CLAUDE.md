# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Create and activate virtual environment
python -m venv memloader-env
source memloader-env/bin/activate  # macOS/Linux
# memloader-env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys and settings
```

### Running the Application
```bash
# Basic usage with dry run (no uploads)
python main.py conversations.json --dry-run

# Upload to Mem0 cloud
python main.py conversations.json --mem0-api-key your_key

# Use local OpenMemory server
python main.py conversations.json --local-server --provider ollama

# OpenAI with batch processing (50% cost savings)
python main.py conversations.json --provider openai --use-batch --dry-run
```

### Testing
Currently no automated tests are configured. To test:
- Run with `--dry-run` flag to validate processing without uploads
- Start with small conversation exports
- Check logs in `memloader.log`

### Linting and Code Quality
```bash
# Format code with Black
black src/ tests/

# Lint code with Ruff (check and fix)
ruff check src/ tests/ --fix

# Type checking with MyPy
mypy src/

# Run all quality checks together
black src/ tests/ && ruff check src/ tests/ --fix && mypy src/

# Run pre-commit hooks manually
pre-commit run --all-files

# Install pre-commit hooks (run once)
pre-commit install
```

## Architecture Overview

### Core Components

**Main Entry Point**: `main.py`
- CLI application using Click framework
- Orchestrates the entire memory extraction pipeline
- Handles configuration, logging, and error management

**Configuration**: `src/config/settings.py`
- Pydantic-based settings management with `.env` file support
- Supports both Ollama (local) and OpenAI (cloud) LLM providers
- Environment variables prefixed with `MEMLOADER_`

**Processing Pipeline**:
1. **Parser** (`src/parsers/json_parser.py`): Parses ChatGPT `conversations.json` exports
2. **Extractor** (`src/extractors/`): Extracts memories using LLMs
   - `ollama_extractor.py`: Local Ollama models (free, private, slower)
   - `openai_extractor.py`: OpenAI API with optional batch processing
3. **Processor** (`src/processors/memory_processor.py`): Deduplication, filtering, categorization
4. **Loader** (`src/loaders/`): Uploads to memory storage
   - `mem0_loader.py`: Mem0 cloud platform
   - `local_mem0_loader.py`: Self-hosted OpenMemory server

### Memory Categories
The system categorizes extracted memories into 8 types:
- `preference`: User habits and preferences
- `fact`: Important personal details
- `pattern`: Behavioral and thinking patterns
- `goal`: Aspirations and objectives
- `skill`: Expertise and capabilities
- `relationship`: People and connections
- `context`: Situational information
- `decision_criteria`: How decisions are made

### Configuration System
Settings are managed through Pydantic with environment variable support:
- Default provider: Ollama (local, free)
- Default model: `nuextract` (specialized for extraction)
- Confidence threshold: 0.7 (filters low-quality memories)
- Chunk size: 1500 tokens (for processing large conversations)
- Batch size: 100 (for Mem0 uploads)

### LLM Provider Options
1. **Ollama** (default): Local processing, requires Ollama installation
2. **OpenAI**: Cloud API with optional batch processing for cost savings

### Memory Storage Options
1. **Mem0 Cloud**: Hosted platform (requires API key)
2. **OpenMemory**: Self-hosted solution (requires separate setup)

### Key Features
- **Batch Processing**: OpenAI batch API for 50% cost savings
- **Chunking**: Handles large conversations by splitting into chunks
- **Deduplication**: Removes duplicate memories across conversations
- **Confidence Filtering**: Only keeps high-confidence extractions
- **Dry Run Mode**: Test processing without uploading
- **Progress Tracking**: tqdm progress bars for long operations
- **Comprehensive Logging**: File and console logging with configurable levels

### Error Handling
- Graceful handling of API failures and timeouts
- Retry logic with configurable attempts and delays
- Detailed error logging and user-friendly error messages
- Keyboard interrupt handling for graceful cancellation

### Environment Variables
Key configuration options (see `.env.example`):
- `MEM0_API_KEY`: Required for Mem0 cloud uploads
- `MEMLOADER_OPENAI_API_KEY`: Required for OpenAI processing
- `MEMLOADER_LLM_PROVIDER`: Choose between "ollama" or "openai"
- `MEMLOADER_CONFIDENCE_THRESHOLD`: Memory quality filter (0.0-1.0)
- `MEMLOADER_OLLAMA_TIMEOUT`: Timeout for local processing (default: 600s)
