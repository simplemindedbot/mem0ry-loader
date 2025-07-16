# MemLoader: ChatGPT Memory Extraction to Mem0

Transform your ChatGPT conversation history into persistent, intelligent memories using advanced LLM processing and Mem0's memory platform.

## 🚀 Features

- **📊 ChatGPT Export Parsing**: Parse `conversations.json` from ChatGPT data exports
- **🤖 Dual LLM Support**: Choose between local Ollama models or OpenAI API
- **💰 Cost Optimization**: OpenAI batch processing for 50% cost savings
- **🧠 Intelligent Memory Processing**: Deduplication, confidence filtering, and categorization
- **🔄 Mem0 Integration**: Seamless upload to Mem0 platform with full metadata
- **🏠 Local Processing**: Complete local LLM processing with privacy-first approach
- **📈 Analytics & Insights**: Detailed processing statistics and confidence metrics
- **🛡️ Dry-Run Mode**: Test processing without uploading to validate results

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/memloader.git
cd memloader
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Choose Your LLM Provider

#### Option A: Local Processing (Ollama) - Free & Private
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended model
ollama pull nuextract
```

#### Option B: Cloud Processing (OpenAI) - Fast & Scalable
```bash
# Just need your OpenAI API key (set in .env)
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

# Upload to Mem0
python main.py conversations.json --mem0-api-key your_key
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

## ⚙️ Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mem0-api-key` | Mem0 API key for uploads | ENV: `MEM0_API_KEY` |
| `--user-id` | User ID for memory organization | `chatgpt_import` |
| `--model` | Ollama model name | `nuextract` |
| `--confidence-threshold` | Min confidence for memories | `0.7` |
| `--batch-size` | Mem0 upload batch size | `100` |
| `--use-batch` | Enable OpenAI batch processing | `false` |
| `--dry-run` | Process without uploading | `false` |
| `--clear-existing` | Clear existing memories first | `false` |
| `--verbose` | Enable detailed logging | `false` |

## 🧠 Memory Categories

MemLoader extracts and categorizes memories into:

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
- **Speed**: Moderate (depends on hardware)
- **Best for**: Privacy-conscious users, cost optimization

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
| User Type | Conversations | Est. Cost (Batch) | Est. Cost (Standard) |
|-----------|---------------|-------------------|---------------------|
| Light User | 100-500 | $0.01-$0.05 | $0.02-$0.10 |
| Regular User | 500-1000 | $0.05-$0.15 | $0.10-$0.30 |
| Heavy User | 1000+ | $0.15-$0.50 | $0.30-$1.00 |

## 🏗️ Architecture

```
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
```
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
```
[preference] I prefer working late at night (10 PM - 2 AM) when it's quiet
[fact] I'm a software engineer with 8 years experience in Python and JavaScript  
[goal] Learning machine learning to transition into AI/ML engineering role
[pattern] I ask detailed technical questions and prefer step-by-step explanations
```

## 🔧 Advanced Usage

### Local Processing with Cloud Storage
```bash
# 1. Test locally without uploading
python main.py conversations.json --dry-run

# 2. Process locally, upload to Mem0 cloud
python main.py conversations.json --mem0-api-key your_key

# 3. Use OpenAI batch processing for cost savings
python main.py conversations.json --use-batch --mem0-api-key your_key
```

**Note**: Currently, memory storage only supports Mem0's cloud platform. Self-hosted memory storage is planned for future releases (see NEXTSTEPS.md).

### Environment Variables
```bash
# Core Configuration
MEM0_API_KEY=your_mem0_api_key
MEMLOADER_OPENAI_API_KEY=your_openai_key

# LLM Provider Selection  
MEMLOADER_LLM_PROVIDER=openai  # or 'ollama'
MEMLOADER_OPENAI_MODEL=gpt-4.1-nano

# Processing Tuning
MEMLOADER_CONFIDENCE_THRESHOLD=0.7
MEMLOADER_BATCH_SIZE=100
MEMLOADER_CHUNK_SIZE=1500
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

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

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

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/yourusername/memloader/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/yourusername/memloader/discussions)
- **📚 Documentation**: See [APPROACH.md](APPROACH.md) for technical details

---

**⭐ Star this repo if MemLoader helped you preserve your ChatGPT memories!**