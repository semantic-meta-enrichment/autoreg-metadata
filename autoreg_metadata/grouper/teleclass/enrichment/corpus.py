import math

import numpy as np
import yake
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from autoreg_metadata.grouper.teleclass.core.config import CorpusConfig
from autoreg_metadata.grouper.teleclass.core.models import (
    CorpusEnrichmentResult,
    DocumentMeta,
    EnrichedClass,
    TermScore,
)
from autoreg_metadata.log import logger


class CorpusEnricher:
    def __init__(
        self,
        config: CorpusConfig,
        encoder_model: str,
    ):
        self.encoder = SentenceTransformer(encoder_model)
        self.keyword_model = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedupLim=0.9,
            dedupFunc="seqm",
            windowsSize=1,
            top=5,
            features=None,
        )
        self.class_terms: list[EnrichedClass] = []
        self.logger = logger.getChild(self.__class__.__name__)

    def enrich(
        self,
        enriched_classes: list[EnrichedClass],
        collection: list[DocumentMeta],
    ) -> CorpusEnrichmentResult:
        """
        Enrich taxonomy using documents

        Args:
            collection: List of DocumentMeta containing documents and its assigned core classes

        Returns:
            Dictionary mapping class names to their enriched terms with scores
        """
        # Convert documents to dictionary format with IoT structure

        for ec in enriched_classes:
            self.logger.info("Enriching class %s", ec.class_name)
            class_docs = []
            for doc in collection:
                if not doc.core_classes:
                    raise ValueError(
                        f"Core classes for document {str(doc.id)} not defined"
                    )
                if ec.class_name in doc.core_classes:
                    class_docs.append(doc)
            # Get sibling data
            sibling_docs = self.get_sibling_data(ec.class_name, collection)

            term_scores = self.enrich_class(ec.class_name, class_docs, sibling_docs)

            ec.terms.update(term_scores)
            ec.embeddings = self.encoder.encode(
                [term_score.term for term_score in ec.terms]
            )

        return CorpusEnrichmentResult(ClassEnrichment=enriched_classes)

    def get_sibling_data(
        self,
        class_name: str,
        collection: list[DocumentMeta],
    ) -> dict[str, list[DocumentMeta]]:
        """
        Get documents assigned to sibling classes, preserving IoT format
        """
        sibling_docs: dict[str, list[DocumentMeta]] = {}
        # Group documents by their assigned classes
        for doc in collection:
            if doc.core_classes:
                for cls in doc.core_classes:
                    if cls != class_name:
                        if cls not in sibling_docs:
                            sibling_docs[cls] = []  # Make sure document exists
                        sibling_docs[cls].append(doc)

        return sibling_docs

    def calculate_popularity(self, term: str, documents: list[str]) -> float:
        """
        Calculate popularity for multi-word terms with more precise matching
        """
        term = term.lower().strip()
        term_words = term.split()

        df = 0
        for doc in documents:
            doc = doc.lower()
            if len(term_words) == 1:
                # Single word - match as whole word
                if term in doc.split():
                    df += 1
            else:
                # Multi-word phrase - exact phrase match
                if term in doc:
                    df += 1

        return math.log(1 + df)

    def calculate_distinctiveness(
        self, term: str, class_docs: list[str], sibling_docs: dict[str, list[str]]
    ) -> float:
        """
        Calculate distinctiveness using BM25 scores with phrase preservation
        """
        term = term.lower().strip()

        # Prepare documents with phrase preservation
        def prepare_doc(doc: str) -> list[str]:
            # Keep the phrase together in tokenization
            doc = doc.lower()
            # Replace term with a single token if it appears
            if term in doc:
                doc = doc.replace(term, term.replace(" ", "_"))
            return doc.split()

        # Tokenize documents preserving phrases
        tokenized_class_docs = [prepare_doc(doc) for doc in class_docs]
        class_bm25 = BM25Okapi(tokenized_class_docs)
        # Use the term as a single token
        term_token = term.replace(" ", "_")
        class_score = class_bm25.get_scores([term_token])[0]

        # Calculate scores for sibling classes
        sibling_scores = []
        for sib_docs in sibling_docs.values():
            tokenized_sib_docs = [prepare_doc(doc) for doc in sib_docs]
            sib_bm25 = BM25Okapi(tokenized_sib_docs)
            sib_score = sib_bm25.get_scores([term_token])[0]
            sibling_scores.append(sib_score)

        # Apply softmax
        all_scores = np.array([class_score] + sibling_scores)
        # Add small epsilon to prevent division by zero
        exp_scores = np.exp(all_scores - np.max(all_scores))
        softmax_scores = exp_scores / (exp_scores.sum() + 1e-10)

        return softmax_scores[0]

    def calculate_semantic_similarity(self, term: str, class_name: str) -> float:
        """
        Calculate semantic similarity using sentence transformer embeddings
        """
        term_embedding = self.encoder.encode(term)
        class_embedding = self.encoder.encode(class_name)

        similarity = self.encoder.similarity(term_embedding, class_embedding).item()

        return similarity

    def extract_key_phrases(self, text: str) -> list[str]:

        keywords = self.keyword_model.extract_keywords(text)

        return [keyword for keyword, _ in keywords]

    def extract_candidate_terms(self, iot_data_list: list[DocumentMeta]) -> set[str]:
        """
        Extract candidate terms from IoT data
        """
        terms = set()
        # Extract terms from descriptions
        text = " ".join(data.content for data in iot_data_list)
        words = self.extract_key_phrases(text)

        # Add single words
        terms.update(words)

        return terms

    def enrich_class(
        self,
        class_name: str,
        input_class: list[DocumentMeta],
        class_siblings: dict[str, list[DocumentMeta]],
        top_k: int = 3,
    ) -> set[TermScore]:
        """
        Enrich a class with terms from IoT data
        """
        # Convert IoT data to text documents
        class_docs = [doc.content for doc in input_class]

        sibling_docs = {
            sib: [d.content for d in docs] for sib, docs in class_siblings.items()
        }

        # Extract candidate terms
        self.logger.info("Extracting candidate terms")
        candidate_terms = self.extract_candidate_terms(input_class)
        self.logger.debug("Candidate terms: %s", candidate_terms)

        # Score terms
        scores = []
        for term in candidate_terms:
            # Calculate component scores
            self.logger.info("Calculating component scores for term: %s", term)
            popularity = self.calculate_popularity(term, class_docs)
            distinctiveness = self.calculate_distinctiveness(
                term, class_docs, sibling_docs
            )
            semantic_similarity = self.calculate_semantic_similarity(term, class_name)

            self.logger.debug(
                "\nPopularity: %10.3f\nDistinctiveness: %10.3f\nSemantic Similarity: %10.3f\n",
                popularity,
                distinctiveness,
                semantic_similarity,
            )

            term_score = TermScore(
                term=term,
                popularity=popularity,
                distinctiveness=distinctiveness,
                semantic_similarity=semantic_similarity,
            )

            self.logger.debug("\nAffinity Score: %s", term_score.affinity_score)

            scores.append(term_score)

        # Sort by affinity score and return top-k
        scores.sort(key=lambda x: x.affinity_score, reverse=True)
        return set(scores[:top_k])
