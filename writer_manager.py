"""
Writer Extraction and Management System
Extracts writer names from documents, manages writer information, and tracks their writings.
"""

import os
import json
import re
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
import logging
from difflib import SequenceMatcher
import hashlib

logger = logging.getLogger(__name__)

# Configuration
WRITERS_DB_FOLDER = "writers_db"
WRITERS_INDEX_FILE = "writers_db/writers_index.json"

class WriterManager:
    """Manages writer extraction, storage, and tracking"""

    def __init__(self, db_folder: str = WRITERS_DB_FOLDER):
        self.db_folder = db_folder
        self.index_file = os.path.join(db_folder, "writers_index.json")
        self.writers_data = {}

        # Create database folder
        os.makedirs(db_folder, exist_ok=True)

        # Load existing writers
        self._load_writers_index()

    def _load_writers_index(self):
        """Load writers index from disk"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.writers_data = json.load(f)
                logger.info(f"Loaded {len(self.writers_data)} writers from index")
            except Exception as e:
                logger.error(f"Error loading writers index: {e}")
                self.writers_data = {}
        else:
            self.writers_data = {}
            logger.info("Created new writers index")

    def _save_writers_index(self):
        """Save writers index to disk"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.writers_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.writers_data)} writers to index")
        except Exception as e:
            logger.error(f"Error saving writers index: {e}")

    def _generate_writer_id(self, name: str) -> str:
        """Generate a unique ID for a writer based on their name"""
        # Normalize name for ID generation
        normalized = name.lower().strip()
        hash_suffix = hashlib.md5(normalized.encode()).hexdigest()[:8]
        clean_name = re.sub(r'[^\w\s-]', '', normalized)
        clean_name = re.sub(r'\s+', '_', clean_name)
        return f"writer_{clean_name}_{hash_suffix}"

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names (0.0 to 1.0)"""
        # Normalize names
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()

        # Exact match
        if n1 == n2:
            return 1.0

        # Calculate sequence similarity
        similarity = SequenceMatcher(None, n1, n2).ratio()

        # Check if one name contains the other (common for variations)
        if n1 in n2 or n2 in n1:
            similarity = max(similarity, 0.85)

        # Check word overlap (for names with different ordering)
        words1 = set(n1.split())
        words2 = set(n2.split())
        word_overlap = len(words1 & words2) / max(len(words1), len(words2))

        return max(similarity, word_overlap)

    def find_matching_writer(self, name: str, threshold: float = 0.85) -> Optional[str]:
        """
        Find a matching writer in the database
        Returns writer_id if match found, None otherwise
        """
        best_match = None
        best_score = 0.0

        for writer_id, writer_data in self.writers_data.items():
            writer_name = writer_data.get('name', '')

            # Check primary name
            score = self._calculate_name_similarity(name, writer_name)

            # Check aliases
            for alias in writer_data.get('aliases', []):
                alias_score = self._calculate_name_similarity(name, alias)
                score = max(score, alias_score)

            if score > best_score:
                best_score = score
                best_match = writer_id

        if best_score >= threshold:
            logger.info(f"Found matching writer: {name} -> {self.writers_data[best_match]['name']} (score: {best_score:.2f})")
            return best_match

        return None

    def extract_writer_names(self, text: str) -> List[str]:
        """
        Extract potential writer names from text
        Uses pattern matching to find common writer indicators
        """
        writers = set()

        # Patterns for writer mentions (supports English and Arabic)
        patterns = [
            # English patterns
            r'(?:written by|author:|by|authored by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'(?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:wrote|writes|published|authored)',

            # Arabic patterns (author, writer, etc.)
            r'(?:كتبه|المؤلف|بقلم|كاتب|للكاتب|تأليف)\s*[:：]?\s*([ء-ي\s]+)',
            r'(?:د\.|أ\.|الدكتور|الأستاذ)\s+([ء-ي\s]+)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.UNICODE)
            for match in matches:
                writer_name = match.group(1).strip()
                # Filter out very short names (likely false positives)
                if len(writer_name) > 3:
                    writers.add(writer_name)

        # Also look for capitalized names (2-4 words) that might be writers
        # This is more aggressive and might have false positives
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
        potential_names = re.findall(name_pattern, text)

        # Only add if they appear multiple times (likely authors)
        name_counts = {}
        for name in potential_names:
            name_counts[name] = name_counts.get(name, 0) + 1

        for name, count in name_counts.items():
            if count >= 2:  # Appears at least twice
                writers.add(name)

        return list(writers)

    def extract_writer_info(self, text: str, writer_name: str, llm=None) -> Dict:
        """
        Extract information about a writer from text
        Uses LLM if available, otherwise uses pattern matching
        """
        info = {
            'mentions': [],
            'context_snippets': [],
            'topics': [],
            'achievements': [],
            'affiliations': []
        }

        # Find all mentions of the writer in the text
        # Create a pattern that matches the writer name (case-insensitive)
        name_pattern = re.escape(writer_name)

        # Find contexts around mentions
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            if re.search(name_pattern, sentence, re.IGNORECASE):
                info['mentions'].append(sentence.strip())
                info['context_snippets'].append(sentence.strip()[:200])

        # Extract achievements (patterns like "won", "awarded", "received")
        achievement_patterns = [
            r'(?:won|awarded|received|honored with)\s+([^.!?]+)',
            r'(?:حصل على|فاز بـ|نال|كُرِّم بـ)\s+([^.!?]+)',
        ]

        for pattern in achievement_patterns:
            for snippet in info['context_snippets']:
                matches = re.findall(pattern, snippet, re.IGNORECASE | re.UNICODE)
                info['achievements'].extend(matches)

        # Extract affiliations (university, organization, etc.)
        affiliation_patterns = [
            r'(?:professor at|researcher at|works at|affiliated with)\s+([A-Z][^.!?,]+)',
            r'(?:University of|Institute of|College of)\s+([A-Z][^.!?,]+)',
            r'(?:أستاذ في|باحث في|يعمل في)\s+([ء-ي\s]+)',
        ]

        for pattern in affiliation_patterns:
            for snippet in info['context_snippets']:
                matches = re.findall(pattern, snippet, re.UNICODE)
                info['affiliations'].extend(matches)

        # If LLM is available, use it for deeper analysis
        if llm and info['context_snippets']:
            try:
                context = "\n".join(info['context_snippets'][:5])
                prompt = f"""Analyze this text about {writer_name} and extract:
1. Key achievements or awards
2. Professional affiliations (universities, organizations)
3. Main topics or fields they work in
4. Any other relevant biographical information

Text:
{context}

Provide a concise summary in JSON format with keys: achievements, affiliations, topics, bio_summary"""

                # Note: This would need the actual LLM call implementation
                # For now, we'll skip this and rely on pattern matching
                pass
            except Exception as e:
                logger.error(f"Error using LLM for writer info extraction: {e}")

        return info

    def add_or_update_writer(self, name: str, document_file: str, text_content: str,
                            additional_info: Optional[Dict] = None, llm=None) -> str:
        """
        Add a new writer or update existing writer information
        Returns writer_id
        """
        # Check if writer already exists
        writer_id = self.find_matching_writer(name)

        if writer_id:
            # Update existing writer
            logger.info(f"Updating existing writer: {name}")
            self._update_writer(writer_id, document_file, text_content, additional_info, llm)
        else:
            # Create new writer
            logger.info(f"Creating new writer: {name}")
            writer_id = self._create_writer(name, document_file, text_content, additional_info, llm)

        self._save_writers_index()
        return writer_id

    def _create_writer(self, name: str, document_file: str, text_content: str,
                      additional_info: Optional[Dict] = None, llm=None) -> str:
        """Create a new writer entry"""
        writer_id = self._generate_writer_id(name)

        # Extract information from text
        extracted_info = self.extract_writer_info(text_content, name, llm)

        writer_data = {
            'writer_id': writer_id,
            'name': name,
            'aliases': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'writings': [{
                'document': document_file,
                'added_at': datetime.now().isoformat(),
                'mentions_count': len(extracted_info['mentions'])
            }],
            'total_mentions': len(extracted_info['mentions']),
            'information': {
                'bio_summary': '',
                'achievements': extracted_info['achievements'],
                'affiliations': extracted_info['affiliations'],
                'topics': extracted_info['topics'],
                'context_snippets': extracted_info['context_snippets'][:10]  # Keep top 10
            },
            'writings_summary': f"Writer mentioned in {document_file}",
            'metadata': additional_info or {}
        }

        self.writers_data[writer_id] = writer_data
        return writer_id

    def _update_writer(self, writer_id: str, document_file: str, text_content: str,
                      additional_info: Optional[Dict] = None, llm=None):
        """Update existing writer with new information"""
        if writer_id not in self.writers_data:
            logger.error(f"Writer {writer_id} not found")
            return

        writer = self.writers_data[writer_id]

        # Extract new information
        extracted_info = self.extract_writer_info(text_content, writer['name'], llm)

        # Check if this document is already tracked
        existing_docs = [w['document'] for w in writer['writings']]
        if document_file not in existing_docs:
            writer['writings'].append({
                'document': document_file,
                'added_at': datetime.now().isoformat(),
                'mentions_count': len(extracted_info['mentions'])
            })

        # Update total mentions
        writer['total_mentions'] += len(extracted_info['mentions'])

        # Merge new information
        writer['information']['achievements'] = list(set(
            writer['information']['achievements'] + extracted_info['achievements']
        ))
        writer['information']['affiliations'] = list(set(
            writer['information']['affiliations'] + extracted_info['affiliations']
        ))
        writer['information']['topics'] = list(set(
            writer['information']['topics'] + extracted_info['topics']
        ))

        # Add new context snippets (keep most recent 20)
        writer['information']['context_snippets'].extend(extracted_info['context_snippets'])
        writer['information']['context_snippets'] = writer['information']['context_snippets'][-20:]

        # Update writings summary
        writer['writings_summary'] = self._generate_writings_summary(writer)

        # Update metadata
        if additional_info:
            writer['metadata'].update(additional_info)

        writer['updated_at'] = datetime.now().isoformat()

    def _generate_writings_summary(self, writer_data: Dict) -> str:
        """Generate a summary of a writer's works"""
        doc_count = len(writer_data['writings'])
        total_mentions = writer_data['total_mentions']

        summary = f"{writer_data['name']} is mentioned in {doc_count} document(s) "
        summary += f"with a total of {total_mentions} mention(s). "

        if writer_data['information']['achievements']:
            summary += f"Notable achievements include: {', '.join(writer_data['information']['achievements'][:3])}. "

        if writer_data['information']['affiliations']:
            summary += f"Affiliated with: {', '.join(writer_data['information']['affiliations'][:3])}."

        return summary

    def process_document_for_writers(self, document_file: str, text_content: str, llm=None) -> List[str]:
        """
        Process a document to extract and track writers
        Returns list of writer_ids found in the document
        """
        logger.info(f"Processing document for writers: {document_file}")

        # Extract writer names
        writer_names = self.extract_writer_names(text_content)
        logger.info(f"Found {len(writer_names)} potential writers: {writer_names}")

        writer_ids = []
        for name in writer_names:
            try:
                writer_id = self.add_or_update_writer(name, document_file, text_content, llm=llm)
                writer_ids.append(writer_id)
            except Exception as e:
                logger.error(f"Error processing writer {name}: {e}")

        return writer_ids

    def get_writer(self, writer_id: str) -> Optional[Dict]:
        """Get writer data by ID"""
        return self.writers_data.get(writer_id)

    def get_writer_by_name(self, name: str) -> Optional[Dict]:
        """Get writer data by name (with fuzzy matching)"""
        writer_id = self.find_matching_writer(name)
        if writer_id:
            return self.writers_data[writer_id]
        return None

    def list_all_writers(self) -> List[Dict]:
        """Get list of all writers"""
        return list(self.writers_data.values())

    def search_writers(self, query: str) -> List[Dict]:
        """Search writers by name or information"""
        results = []
        query_lower = query.lower()

        for writer_data in self.writers_data.values():
            # Search in name
            if query_lower in writer_data['name'].lower():
                results.append(writer_data)
                continue

            # Search in aliases
            if any(query_lower in alias.lower() for alias in writer_data['aliases']):
                results.append(writer_data)
                continue

            # Search in affiliations
            if any(query_lower in aff.lower() for aff in writer_data['information']['affiliations']):
                results.append(writer_data)
                continue

        return results

    def get_writers_by_document(self, document_file: str) -> List[Dict]:
        """Get all writers mentioned in a specific document"""
        results = []

        for writer_data in self.writers_data.values():
            for writing in writer_data['writings']:
                if writing['document'] == document_file:
                    results.append(writer_data)
                    break

        return results

    def get_document_stats(self) -> Dict:
        """Get statistics about writers in the database"""
        total_writers = len(self.writers_data)
        total_writings = sum(len(w['writings']) for w in self.writers_data.values())
        total_mentions = sum(w['total_mentions'] for w in self.writers_data.values())

        # Get most mentioned writers
        writers_by_mentions = sorted(
            self.writers_data.values(),
            key=lambda x: x['total_mentions'],
            reverse=True
        )

        return {
            'total_writers': total_writers,
            'total_writings_tracked': total_writings,
            'total_mentions': total_mentions,
            'top_writers': [
                {
                    'name': w['name'],
                    'mentions': w['total_mentions'],
                    'documents': len(w['writings'])
                }
                for w in writers_by_mentions[:10]
            ]
        }

    def add_alias(self, writer_id: str, alias: str):
        """Add an alias for a writer"""
        if writer_id in self.writers_data:
            if alias not in self.writers_data[writer_id]['aliases']:
                self.writers_data[writer_id]['aliases'].append(alias)
                self.writers_data[writer_id]['updated_at'] = datetime.now().isoformat()
                self._save_writers_index()
                logger.info(f"Added alias '{alias}' to writer {writer_id}")

    def merge_writers(self, writer_id1: str, writer_id2: str, keep_id: Optional[str] = None):
        """Merge two writer entries (for duplicates)"""
        if writer_id1 not in self.writers_data or writer_id2 not in self.writers_data:
            logger.error("One or both writers not found")
            return False

        # Determine which to keep
        if not keep_id:
            keep_id = writer_id1
        remove_id = writer_id2 if keep_id == writer_id1 else writer_id1

        # Merge data
        keeper = self.writers_data[keep_id]
        removed = self.writers_data[remove_id]

        # Merge aliases
        keeper['aliases'] = list(set(keeper['aliases'] + removed['aliases'] + [removed['name']]))

        # Merge writings
        existing_docs = [w['document'] for w in keeper['writings']]
        for writing in removed['writings']:
            if writing['document'] not in existing_docs:
                keeper['writings'].append(writing)

        # Merge information
        keeper['information']['achievements'] = list(set(
            keeper['information']['achievements'] + removed['information']['achievements']
        ))
        keeper['information']['affiliations'] = list(set(
            keeper['information']['affiliations'] + removed['information']['affiliations']
        ))
        keeper['information']['topics'] = list(set(
            keeper['information']['topics'] + removed['information']['topics']
        ))

        # Update totals
        keeper['total_mentions'] += removed['total_mentions']
        keeper['updated_at'] = datetime.now().isoformat()
        keeper['writings_summary'] = self._generate_writings_summary(keeper)

        # Remove the duplicate
        del self.writers_data[remove_id]

        self._save_writers_index()
        logger.info(f"Merged writer {remove_id} into {keep_id}")
        return True


# Singleton instance
_writer_manager = None

def get_writer_manager() -> WriterManager:
    """Get or create the global writer manager instance"""
    global _writer_manager
    if _writer_manager is None:
        _writer_manager = WriterManager()
    return _writer_manager
