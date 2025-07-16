#!/usr/bin/env python3
"""Main CLI application for ChatGPT memory extraction to Mem0."""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.parsers.json_parser import ChatGPTJSONParser
from src.extractors.ollama_extractor import OllamaExtractor
from src.extractors.openai_extractor import OpenAIExtractor
from src.processors.memory_processor import MemoryProcessor
from src.loaders.mem0_loader import Mem0Loader
from src.loaders.local_mem0_loader import LocalMem0Loader
from src.config.settings import settings, LLMProvider


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('memloader.log')
        ]
    )


@click.command()
@click.argument('export_file', type=click.Path(exists=True, path_type=Path))
@click.option('--mem0-api-key', help='Mem0 API key (or set MEM0_API_KEY env var)')
@click.option('--user-id', help='User ID for Mem0 memories (overrides .env setting)')
@click.option('--model', help='Ollama model to use (overrides .env setting)')
@click.option('--confidence-threshold', type=float, help='Minimum confidence for memories (overrides .env setting)')
@click.option('--batch-size', type=int, help='Batch size for Mem0 uploads (overrides .env setting)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--dry-run', is_flag=True, help='Process but don\'t upload to Mem0')
@click.option('--clear-existing', is_flag=True, help='Clear existing memories before upload')
@click.option('--use-batch', is_flag=True, help='Use OpenAI batch processing (50% cost savings)')
@click.option('--local-server', is_flag=True, help='Use local OpenMemory server instead of cloud')
@click.option('--provider', type=click.Choice(['ollama', 'openai']), help='LLM provider to use (overrides .env setting)')
def main(export_file: Path, mem0_api_key: Optional[str], user_id: Optional[str], 
         model: Optional[str], confidence_threshold: Optional[float], batch_size: Optional[int], 
         verbose: bool, dry_run: bool, clear_existing: bool, use_batch: bool,
         local_server: bool, provider: Optional[str]):
    """Extract memories from ChatGPT export and load into Mem0.
    
    EXPORT_FILE: Path to the ChatGPT conversations.json file
    """
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Apply command line overrides or use .env defaults
    if confidence_threshold is not None:
        settings.confidence_threshold = confidence_threshold
        logger.info(f"Using confidence threshold from command line: {confidence_threshold}")
    else:
        logger.info(f"Using confidence threshold from .env: {settings.confidence_threshold}")
    
    if batch_size is not None:
        settings.batch_size = batch_size
        logger.info(f"Using batch size from command line: {batch_size}")
    else:
        logger.info(f"Using batch size from .env: {settings.batch_size}")
    
    # Override provider if specified via command line
    if provider:
        settings.llm_provider = LLMProvider(provider)
        logger.info(f"Using LLM provider from command line: {provider}")
    else:
        logger.info(f"Using LLM provider from .env: {settings.llm_provider}")
    
    # Set user_id and model, using .env defaults if not provided
    final_user_id = user_id or os.getenv('USER', 'default_user')
    final_model = model or settings.ollama_model
    
    # Use provided API key or environment variable
    api_key = mem0_api_key or os.getenv('MEM0_API_KEY')
    if not api_key and not dry_run and not local_server:
        click.echo("Error: Mem0 API key is required. Set MEM0_API_KEY environment variable or use --mem0-api-key option.")
        click.echo("Alternatively, use --local-server to use local OpenMemory server (no API key required).")
        sys.exit(1)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        parser = ChatGPTJSONParser()
        
        # Initialize extractor based on provider
        if settings.llm_provider == LLMProvider.OPENAI:
            extractor = OpenAIExtractor(model=settings.openai_model, use_batch=use_batch)
        else:
            extractor = OllamaExtractor(model=final_model)
            
        processor = MemoryProcessor(confidence_threshold=settings.confidence_threshold)
        
        if not dry_run:
            if local_server:
                loader = LocalMem0Loader(user_id=final_user_id)
                click.echo("Using local OpenMemory server")
            else:
                loader = Mem0Loader(api_key=api_key, user_id=final_user_id)
                click.echo("Using Mem0 cloud platform")
            
            # Clear existing memories if requested
            if clear_existing:
                existing_memories = loader.get_existing_memories()
                if existing_memories:
                    memory_ids = [mem.get('id') for mem in existing_memories if mem.get('id')]
                    if memory_ids:
                        deleted_count = loader.delete_existing_memories(memory_ids)
                        click.echo(f"Deleted {deleted_count} existing memories")
        
        # Parse ChatGPT export
        click.echo("Parsing ChatGPT export...")
        conversations = parser.parse_export(export_file)
        
        if not conversations:
            click.echo("No conversations found in export file.")
            sys.exit(1)
        
        click.echo(f"Found {len(conversations)} conversations")
        
        # Extract memories from conversations
        all_memories = []
        
        if use_batch and settings.llm_provider == LLMProvider.OPENAI:
            # Batch processing: collect all requests first
            click.echo("Collecting extraction requests for batch processing...")
            with tqdm(total=len(conversations), desc="Collecting requests") as pbar:
                for conversation in conversations:
                    # Process conversation in chunks
                    for chunk in parser.get_conversation_chunks(conversation):
                        try:
                            # This adds to batch queue, returns empty list
                            extractor.extract_memories(chunk, conversation.title)
                        except Exception as e:
                            logger.error(f"Failed to add chunk to batch: {e}")
                            continue
                    
                    pbar.update(1)
            
            # Process the entire batch
            click.echo(f"Processing {len(extractor.batch_requests)} requests in batch...")
            all_memories = extractor.process_batch()
        else:
            # Real-time processing
            with tqdm(total=len(conversations), desc="Extracting memories") as pbar:
                for conversation in conversations:
                    # Process conversation in chunks
                    for chunk in parser.get_conversation_chunks(conversation):
                        try:
                            memories = extractor.extract_memories(chunk, conversation.title)
                            all_memories.extend(memories)
                        except Exception as e:
                            logger.error(f"Failed to extract memories from chunk: {e}")
                            continue
                    
                    pbar.update(1)
        
        click.echo(f"Extracted {len(all_memories)} raw memories")
        
        if not all_memories:
            click.echo("No memories extracted. Check your export file and model configuration.")
            sys.exit(1)
        
        # Process memories
        click.echo("Processing memories...")
        processed_memories, stats = processor.process_memories(all_memories)
        
        # Display processing statistics
        click.echo(f"""
Processing Statistics:
  Input memories: {stats.total_input}
  Output memories: {stats.total_output}
  Duplicates removed: {stats.duplicates_removed}
  Low confidence filtered: {stats.low_confidence_filtered}
  Merged memories: {stats.merged_memories}
  
Category Distribution:""")
        
        for category, count in stats.categories.items():
            click.echo(f"  {category}: {count}")
        
        # Get confidence statistics
        confidence_stats = processor.get_confidence_statistics(processed_memories)
        click.echo(f"""
Confidence Statistics:
  Min: {confidence_stats['min']:.2f}
  Max: {confidence_stats['max']:.2f}
  Average: {confidence_stats['avg']:.2f}
  Median: {confidence_stats['median']:.2f}""")
        
        if dry_run:
            click.echo("\\nDry run complete. No memories uploaded to Mem0.")
            
            # Show sample memories
            if processed_memories:
                click.echo("\\nSample memories:")
                for i, memory in enumerate(processed_memories[:5]):
                    click.echo(f"  {i+1}. [{memory.category}] {memory.content} (confidence: {memory.confidence:.2f})")
                
                if len(processed_memories) > 5:
                    click.echo(f"  ... and {len(processed_memories) - 5} more")
        else:
            # Upload to Mem0
            click.echo("\\nUploading to Mem0...")
            
            # Prepare memories for upload
            upload_ready_memories = loader.prepare_memories_for_upload(processed_memories)
            click.echo(f"Prepared {len(upload_ready_memories)} memories for upload")
            
            if upload_ready_memories:
                # Upload memories
                upload_stats = loader.load_memories(upload_ready_memories, settings.batch_size)
                
                click.echo(f"""
Upload Statistics:
  Total processed: {upload_stats['total_processed']}
  Successfully uploaded: {upload_stats['uploaded']}
  Failed: {upload_stats['failed']}
  Success rate: {upload_stats['success_rate']:.1%}""")
                
                if upload_stats['success_rate'] > 0.9:
                    click.echo("✅ Upload completed successfully!")
                else:
                    click.echo("⚠️  Upload completed with some failures. Check logs for details.")
            else:
                click.echo("No new memories to upload (all filtered out or duplicates).")
    
    except KeyboardInterrupt:
        click.echo("\\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()