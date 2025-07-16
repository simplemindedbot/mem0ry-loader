# ChatGPT Memory Extraction to Mem0 - Technical Approach

## Overview
This project extracts meaningful memories from ChatGPT chat history exports and loads them into Mem0's persistent memory system using local LLMs for processing.

## 1. ChatGPT Export Methods

### Official Export Process
- Navigate to ChatGPT Settings → Data Controls → Export Data
- Receive email with download link (expires in 24 hours)
- Download contains:
  - `conversations.json` - Structured conversation data
  - `chat.html` - Human-readable format

### Export Limitations
- Images/files from DALL-E not included in exports (as of Aug 2024)
- Must be done manually through ChatGPT interface

### Third-Party Tools
- Browser extensions for real-time export
- Format conversion tools for additional output formats
- GitHub tools like `chatgpt-exporter` for enhanced functionality

## 2. Memory Extraction Strategy

### Local LLM Processing
Two approaches for memory extraction:

#### Option A: Ollama (Recommended)
**Best Models for Memory Extraction:**
- **NuExtract** (Phi-3 Mini based) - Specialized for information extraction
- **Llama 3.2 1B/3B** - Ultra-lightweight, good for basic memory extraction
- **Mistral Small** - Balanced performance, knowledge-dense
- **Gemma 2B** - Efficient and capable

**Recommended: NuExtract**
- Designed specifically for information extraction
- Based on Phi-3 Mini (3B parameters)
- Accepts JSON templates for structured extraction
- Handles up to 2000 tokens per request

#### Option B: OpenAI API
**GPT-4o-mini Pricing:**
- Input: $0.15 per million tokens
- Output: $0.60 per million tokens
- 128K context window

**GPT-4.1-nano Pricing:**
- Input: $0.10 per million tokens  
- Output: $0.40 per million tokens
- 1M context window
- Fastest option with 80.1% MMLU score

### Memory Extraction Prompt Strategy
```
Extract personal memories, preferences, and contextual information from this conversation:

Focus on:
- User preferences and habits
- Important personal details
- Recurring topics/interests  
- Problem-solving patterns
- Decision-making criteria
- Skills and expertise areas
- Goal statements
- Relationship information

Return structured JSON with:
- memory_content: The extracted memory
- category: Type of memory (preference, fact, pattern, etc.)
- confidence: 0-1 confidence score
- context: Original conversation context
```

## 3. Mem0 Integration

### API Setup
```python
from mem0 import MemoryClient
import os

os.environ["MEM0_API_KEY"] = "your-api-key"
client = MemoryClient()
```

### Memory Import Process
```python
# Add extracted memories
client.add(
    messages=[{"role": "user", "content": memory_content}],
    user_id="chatgpt_import",
    metadata={"source": "chatgpt_export", "confidence": confidence_score}
)
```

### Batch Processing
- Support for up to 1000 memories per batch operation
- Async client available for performance
- Memory deduplication and conflict resolution

## 4. Cost Analysis

### Processing 1M Tokens of Chat History

**Local Processing (Ollama):**
- Cost: $0 (hardware only)
- Time: Depends on hardware (estimated 10-60 minutes)
- Privacy: Complete local processing

**OpenAI API Processing:**
**GPT-4o-mini:**
- Input cost: $0.15 (1M tokens)
- Output cost: ~$0.06 (100K tokens estimated)
- Total: ~$0.21 per 1M tokens

**GPT-4.1-nano:**
- Input cost: $0.10 (1M tokens)
- Output cost: ~$0.04 (100K tokens estimated)  
- Total: ~$0.14 per 1M tokens
- Faster processing time

### Typical ChatGPT Export Size
- Average user: 50K-500K tokens
- Heavy user: 1M-5M tokens
- Cost range: $0.007-$0.70 for most users

## 5. Implementation Architecture

### Core Components
1. **Export Parser** - Parse conversations.json and chat.html
2. **Memory Extractor** - LLM-based memory extraction
3. **Memory Processor** - Clean, deduplicate, and structure memories
4. **Mem0 Loader** - Batch upload to Mem0 platform
5. **Configuration Manager** - Model selection and API keys

### Data Flow
```
ChatGPT Export → Parser → Chunker → LLM Extraction → 
Memory Processor → Deduplication → Mem0 Upload → Verification
```

### Error Handling
- Chunk size validation for LLM context limits
- API rate limiting and retries
- Memory validation before upload
- Rollback capability for failed batches

## 6. Project Structure

```
memloader/
├── src/
│   ├── parsers/
│   │   ├── json_parser.py
│   │   └── html_parser.py
│   ├── extractors/
│   │   ├── ollama_extractor.py
│   │   └── openai_extractor.py
│   ├── processors/
│   │   ├── memory_processor.py
│   │   └── deduplicator.py
│   ├── loaders/
│   │   └── mem0_loader.py
│   └── config/
│       └── settings.py
├── tests/
├── examples/
├── requirements.txt
└── README.md
```

## 7. Configuration Options

### Model Selection
- Ollama models: NuExtract, Llama 3.2, Mistral Small
- OpenAI models: GPT-4o-mini, GPT-4.1-nano
- Fallback options and error handling

### Processing Parameters
- Chunk size optimization
- Memory confidence thresholds
- Batch sizes for Mem0 upload
- Rate limiting configuration

## 8. Next Steps

1. **Phase 1**: Build basic JSON parser and Ollama integration
2. **Phase 2**: Add OpenAI API support and cost tracking
3. **Phase 3**: Implement memory deduplication and conflict resolution
4. **Phase 4**: Add batch processing and progress tracking
5. **Phase 5**: Create CLI interface and configuration management

## 9. Success Metrics

- **Accuracy**: >90% relevant memories extracted
- **Cost Efficiency**: <$1 per typical user export
- **Performance**: Process 1M tokens in <30 minutes
- **Reliability**: <1% data loss during processing
- **Usability**: Single command execution for end users