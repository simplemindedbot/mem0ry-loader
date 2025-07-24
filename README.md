# MemLoader: ChatGPT Memory Extraction to Mem0

Transform your ChatGPT conversation history into persistent, intelligent memories using advanced LLM processing and Mem0's memory platform.

## 🚀 Features

- **📊 ChatGPT Export Parsing**: Parse `conversations.json` from ChatGPT data exports
- **🤖 Dual LLM Support**: Choose between local Ollama models or OpenAI API
- **💰 Cost Optimization**: OpenAI batch processing for 50% cost savings
- **🧠 Intelligent Memory Processing**: Deduplication, confidence filtering, and categorization
- **🔄 Mem0 Integration**: Seamless upload to Mem0 platform with full metadata
- **🏠 Self-Hosted Option**: Complete local memory storage with OpenMemory integration
- **🔒 Privacy-First**: Local processing option for sensitive conversations
- **📈 Analytics & Insights**: Detailed processing statistics and confidence metrics
- **🛡️ Dry-Run Mode**: Test processing without uploading to validate results

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/memloader.git
cd memloader
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv memloader-env

# Activate virtual environment
# On macOS/Linux:
source memloader-env/bin/activate
# On Windows:
# memloader-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: Always activate the virtual environment before running MemLoader:

```bash
source memloader-env/bin/activate  # macOS/Linux
# or
memloader-env\Scripts\activate     # Windows
```

### 3. Choose Your LLM Provider

#### Option A: Local Processing (Ollama) - Free & Private

**⚠️ External Dependency**: Ollama installation and model management is outside the scope of this project.

**⚠️ Performance Note**: Local processing can be significantly slower than cloud APIs, especially on CPU-only systems. Processing times can range from 5-30 seconds per conversation depending on your hardware.

```bash
# Install Ollama (see https://ollama.ai/install)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended model
ollama pull nuextract
```

**Hardware Recommendations:**

- **Minimum**: 8GB RAM, modern CPU (expect slower processing)
- **Recommended**: 16GB+ RAM, dedicated GPU (NVIDIA/AMD)
- **Optimal**: 32GB+ RAM, high-end GPU (RTX 4090, etc.)

For detailed installation instructions, visit: <https://ollama.ai/>

#### Option B: Cloud Processing (OpenAI) - Fast & Scalable

```bash
# Just need your OpenAI API key (set in .env)
```

#### Option C: Self-Hosted Memory Storage (OpenMemory)

For complete self-hosting with local memory storage, you'll need to set up OpenMemory separately:

**⚠️ External Dependency**: OpenMemory installation and configuration is outside the scope of this project. Please refer to the official documentation:

- **OpenMemory Setup**: <https://github.com/mem0ai/mem0/tree/main/openmemory>
- **Requirements**: Docker, Docker Compose, Qdrant vector database

```bash
# After setting up OpenMemory, you can use:
python main.py conversations.json --local-server --provider ollama
```

### 4. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit with your keys
nano .env
```

## 📁 Get Your ChatGPT Data

1. **Sign in to ChatGPT** → Profile → Settings
2. **Data Controls** → Export Data
3. **Click "Export"** and confirm via email
4. **Download & extract** the ZIP file
5. **Use the `conversations.json`** file with MemLoader

## 🚀 Quick Start

### Basic Usage

```bash
# Local processing (free)
python main.py conversations.json --dry-run

# Upload to Mem0 cloud
python main.py conversations.json --mem0-api-key your_key

# Use local OpenMemory server
python main.py conversations.json --local-server --provider ollama
```

### OpenAI with Batch Processing (50% savings)

```bash
python main.py conversations.json \
  --use-batch \
  --dry-run
```

### Advanced Configuration

```bash
python main.py conversations.json \
  --user-id "my_chatgpt_memories" \
  --confidence-threshold 0.8 \
  --batch-size 50 \
  --clear-existing \
  --verbose
```

### Complete Self-Hosted Setup

```bash
# Use local Ollama + local OpenMemory storage
# Note: This will be slower but completely private
python main.py conversations.json \
  --local-server \
  --provider ollama \
  --user-id "my_user" \
  --dry-run
```

**Performance Expectations:**

- **Cloud API**: 3-8 seconds per conversation
- **Local Ollama**: 5-30 seconds per conversation (depends on hardware)
- **Batch Processing**: Slower individual processing but better cost efficiency

## ⚙️ Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mem0-api-key` | Mem0 API key for uploads | ENV: `MEM0_API_KEY` |
| `--user-id` | User ID for memory organization | ENV: `USER` |
| `--model` | Ollama model name | `nuextract` |
| `--confidence-threshold` | Min confidence for memories | `0.7` |
| `--batch-size` | Upload batch size | `100` |
| `--provider` | LLM provider (ollama/openai) | `ollama` |
| `--local-server` | Use local OpenMemory server | `false` |
| `--use-batch` | Enable OpenAI batch processing | `false` |
| `--dry-run` | Process without uploading | `false` |
| `--clear-existing` | Clear existing memories first | `false` |
| `--verbose` | Enable detailed logging | `false` |

## Memory Categories

MemLoader extracts and categorizes memories into 8 types:

- **🎯 Preferences**: User habits and preferences
- **📋 Facts**: Important personal details
- **🔄 Patterns**: Behavioral and thinking patterns
- **🏆 Goals**: Aspirations and objectives
- **⚡ Skills**: Expertise and capabilities
- **👥 Relationships**: People and connections
- **📍 Context**: Situational information
- **⚖️ Decision Criteria**: How decisions are made

## 💰 Cost Comparison

### Local Processing (Ollama)

- **Cost**: $0 (hardware only)
- **Privacy**: Complete local control
- **Speed**: Slow to moderate (highly dependent on hardware)
- **Hardware Requirements**: GPU recommended for reasonable performance
- **Best for**: Privacy-conscious users, cost optimization, users with capable hardware

### OpenAI Standard API

- **GPT-4o-mini**: $0.15/1M input + $0.60/1M output tokens
- **GPT-4.1-nano**: $0.10/1M input + $0.40/1M output tokens
- **Speed**: Fast (3-8 seconds per conversation)
- **Best for**: Speed and scale

### OpenAI Batch Processing (Recommended)

- **50% cost reduction** on all models
- **GPT-4o-mini**: $0.075/1M input + $0.30/1M output tokens
- **Processing time**: Usually within hours, guaranteed within 24h
- **Best for**: Large exports with cost optimization

### Typical Usage Examples

| User Type | Conversations | Est. Cost (Batch) | Est. Cost (Standard) | Local Processing Time* |
|-----------|---------------|-------------------|---------------------|----------------------|
| Light User | 100-500 | $0.01-$0.05 | $0.02-$0.10 | 8-40 minutes |
| Regular User | 500-1000 | $0.05-$0.15 | $0.10-$0.30 | 40-80 minutes |
| Heavy User | 1000+ | $0.15-$0.50 | $0.30-$1.00 | 80+ minutes |

*Local processing times assume average hardware (16GB RAM, modern CPU). GPU acceleration can significantly reduce these times.

## 🏗️ Architecture

``` txt
memloader/
├── src/
│   ├── config/          # Settings and configuration
│   ├── parsers/         # ChatGPT JSON parsing
│   ├── extractors/      # LLM memory extraction
│   │   ├── ollama_extractor.py    # Local Ollama models
│   │   └── openai_extractor.py    # OpenAI API + batch
│   ├── processors/      # Memory processing pipeline
│   └── loaders/         # Mem0 platform integration
├── main.py             # CLI application
├── .env               # Environment configuration
└── requirements.txt   # Dependencies
```

## 📊 Output Examples

### Processing Statistics

``` txt
Processing Statistics:
  Input memories: 1,847
  Output memories: 1,203
  Duplicates removed: 423
  Low confidence filtered: 221
  Merged memories: 89

Category Distribution:
  preferences: 384 (32%)
  facts: 267 (22%)
  patterns: 203 (17%)
  goals: 158 (13%)
  skills: 121 (10%)
  relationships: 70 (6%)

Confidence Statistics:
  Min: 0.70, Max: 1.00
  Average: 0.92, Median: 0.95
```

### Sample Extracted Memories

 txt
[preference] I prefer working late at night (10 PM - 2 AM) when it's quiet
[fact] I'm a software engineer with 8 years experience in Python and JavaScript
[goal] Learning machine learning to transition into AI/ML engineering role
[pattern] I ask detailed technical questions and prefer step-by-step explanations
```

## 🔧 Advanced Usage

### Deployment Options

#### Option 1: Local Processing → Cloud Storage

```bash
# Process locally with Ollama, store in Mem0 cloud
python main.py conversations.json --provider ollama --mem0-api-key your_key
```

#### Option 2: Cloud Processing → Cloud Storage

```bash
# Use OpenAI batch processing for cost savings
python main.py conversations.json --provider openai --use-batch --mem0-api-key your_key
```

#### Option 3: Complete Self-Hosted

```bash
# Process locally with Ollama, store in local OpenMemory
python main.py conversations.json --provider ollama --local-server
```

**Note**: For self-hosted memory storage, you'll need to set up OpenMemory separately (see installation section).

### Environment Variables

```bash
# Required for cloud processing
MEMLOADER_OPENAI_API_KEY=your_openai_key
MEM0_API_KEY=your_mem0_api_key

# Processing configuration
MEMLOADER_LLM_PROVIDER=ollama  # or 'openai'
MEMLOADER_CONFIDENCE_THRESHOLD=0.7
MEMLOADER_BATCH_SIZE=100
MEMLOADER_OLLAMA_TIMEOUT=600
```

## 🔍 Troubleshooting

### Ollama Issues

```bash
# Check Ollama status
ollama serve

# List available models
ollama list

# Test model directly
ollama run nuextract "Extract memories from: I love coffee in the morning"
```

**Common Performance Issues:**

- **Slow processing**: Ensure you have adequate RAM and consider GPU acceleration
- **Model loading delays**: Models are downloaded on first use and cached locally
- **Memory errors**: Reduce batch size or use smaller models if running out of memory
- **CPU usage**: Ollama is CPU-intensive without GPU acceleration
- **Timeout errors**: Increase `MEMLOADER_OLLAMA_TIMEOUT` in .env for very large conversations (default: 600 seconds)

### OpenAI Issues

```bash
# Test API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Check batch job status (if using --use-batch)
# Job ID will be logged during processing
```

### Memory Quality Issues

- **Low extraction**: Lower `--confidence-threshold` to 0.5-0.6
- **Too many duplicates**: Conversations may have repetitive content
- **Empty results**: Check conversation file format and model availability

### Large Dataset Issues

- **Timeout errors**: Increase `MEMLOADER_OLLAMA_TIMEOUT` in .env (default: 600 seconds)
- **Memory issues**: Reduce `MEMLOADER_CHUNK_SIZE` for very large conversations
- **Long processing times**: Consider using `--use-batch` with OpenAI for large datasets

### Python Environment Issues

- **Module not found errors**: Ensure virtual environment is activated
- **Permission errors**: Use virtual environment instead of system Python
- **Dependency conflicts**: Create fresh virtual environment if needed

```bash
# Recreate virtual environment
rm -rf memloader-env
python -m venv memloader-env
source memloader-env/bin/activate
pip install -r requirements.txt
```

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Set up development environment
python -m venv memloader-dev
source memloader-dev/bin/activate  # macOS/Linux
# or memloader-dev\Scripts\activate  # Windows

# Install dev dependencies
pip install -r requirements.txt
# pip install -r requirements-dev.txt  # if available

# Run tests
pytest tests/

# Format code
black src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Mem0](https://mem0.ai/)** - Persistent memory platform
- **[Ollama](https://ollama.ai/)** - Local LLM serving
- **[OpenAI](https://openai.com/)** - GPT models and batch processing
- **NuExtract** - Specialized extraction model

## ⚠️ Important Disclaimers

### Cost Responsibility

**All API usage costs are the sole responsibility of the end user.** Cost estimates provided in this documentation are approximations based on limited testing and may not reflect actual usage costs in your specific environment. Costs may vary significantly based on:

- Conversation length and complexity
- Model selection and pricing changes
- API rate limits and retry logic
- Regional pricing differences

**Simplemindedbot and Scot Campbell are not responsible for any API charges incurred through the use of this software.**

### Performance Metrics

**All performance metrics and processing statistics are based on limited datasets and may not reflect real-world scenarios.** Actual results may vary based on:

- Conversation content and structure
- Hardware specifications (for local processing)
- Network conditions and API response times
- Model performance variations

**Use provided benchmarks as general guidance only, not as guaranteed performance metrics.**

### Data Privacy

While this tool includes privacy-focused local processing options, **you are responsible for ensuring compliance with your organization's data policies and applicable privacy regulations** when processing conversation data.

### Third-Party Terms of Service

**You must comply with all applicable third-party terms of service**, including but not limited to:

- **OpenAI Terms of Service** - When using OpenAI APIs or processing ChatGPT data
- **ChatGPT Terms of Use** - When exporting and processing your conversation data
- **Mem0 Terms of Service** - When using Mem0 platform features
- **Local regulations** - Regarding AI, data processing, and privacy

**It is your responsibility to ensure you have proper licenses and permissions for all services used.**

### Security Considerations

**Protect your API keys and sensitive data:**

- Never commit API keys to version control
- Use environment variables for all credentials
- Review the .gitignore file before committing changes
- Consider using API key rotation for production deployments
- Be aware that conversation data may contain sensitive personal information

### No Affiliation Disclaimer

**MemLoader is an independent open-source project** and is not affiliated with, endorsed by, or sponsored by OpenAI, ChatGPT, Mem0, or any other third-party service providers. All trademarks are the property of their respective owners.

### Experimental Software

**This is experimental software** that may contain bugs, security vulnerabilities, or incomplete features. Use at your own risk and thoroughly test before any production deployment. We recommend starting with small datasets and dry-run mode.

### Responsible Use

**Please use this software responsibly:**

- **Commercial Use**: While the MIT license permits commercial use, evaluate all third-party API terms and costs for business deployments
- **Rate Limiting**: Respect API rate limits to avoid service interruptions or account restrictions
- **Data Retention**: Consider implementing data retention and deletion policies for processed conversation data
- **Testing**: Always test with small datasets first, especially when using new models or configurations
- **Monitoring**: Monitor API usage and costs to avoid unexpected charges

## 📞 Support

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/your-username/memloader/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/your-username/memloader/discussions)
- **📚 Documentation**: See [APPROACH.md](APPROACH.md) for technical details

---

**⭐ Star this repo if MemLoader helped you preserve your ChatGPT memories!**
