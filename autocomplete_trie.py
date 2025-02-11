
from collections import defaultdict
from typing import List, Set
from collections import Counter

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.ticker = None

class HybridAutocomplete:
    def __init__(self, tickers: List[str], ngram_size: int = 3):
        self.tickers = set(tickers)
        self.ngram_size = ngram_size
        self.root = TrieNode()
        self.ngram_scores = defaultdict(lambda: defaultdict(int))

        self._build_trie()
        self._build_ngram_model()

    def _build_trie(self):
        for ticker in self.tickers:
            node = self.root
            for char in ticker:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.ticker = ticker

    def _build_ngram_model(self):
         for ticker in self.tickers:
            padded = f"#{ticker}#"
            for i in range(len(padded) - self.ngram_size + 1):
                ngram = padded[i:i + self.ngram_size]
                self.ngram_scores[ngram][ticker] += 1

    def _levenshtein_distance(self, s1: str, s2: str) -> int:

        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _get_trie_matches(self, prefix: str) -> Set[str]:
      node = self.root
      for char in prefix:
          if char not in node.children:
              return set()
          node = node.children[char]

      matches = set()
      def collect_matches(node, current_word=""):
          if node.is_end:
              matches.add(node.ticker)
          for char, child in node.children.items():
              collect_matches(child, current_word + char)

      collect_matches(node, prefix)
      return matches

    def _get_ngram_score(self, query: str, ticker: str) -> float:
    # For short queries, dynamically adjust n-gram size
      effective_ngram_size = min(self.ngram_size, len(query) + 1)

      score = 0
      query_padded = f"#{query}#"
      ticker_padded = f"#{ticker}#"

      # Get n-grams of query
      query_ngrams = set()
      for i in range(len(query_padded) - effective_ngram_size + 1):
          query_ngrams.add(query_padded[i:i + effective_ngram_size])

      # Get n-grams of ticker
      ticker_ngrams = set()
      for i in range(len(ticker_padded) - effective_ngram_size + 1):
          ticker_ngrams.add(ticker_padded[i:i + effective_ngram_size])

      # Calculate Jaccard similarity
      score = len(query_ngrams & ticker_ngrams) / len(query_ngrams | ticker_ngrams)

      # Boost score if query appears as a continuous sequence in ticker
      if query in ticker:
          score *= 2

      return score

    def get_suggestions(self, query: str, max_suggestions: int = 10, max_distance: int = 3):
      candidates = set()

      # First get prefix matches from trie
      trie_matches = self._get_trie_matches(query.upper())
      candidates.update(trie_matches)

      # NEW: Get candidates from n-gram matches first
      query_padded = f"#{query.upper()}#"
      for i in range(len(query_padded) - self.ngram_size + 1):
          ngram = query_padded[i:i + self.ngram_size]
          # Add all tickers that contain this ngram
          candidates.update(self.ngram_scores[ngram].keys())

      # Then add fuzzy matches
      for ticker in self.tickers:
          if self._levenshtein_distance(query.upper(), ticker[:len(query)]) <= max_distance:
              candidates.add(ticker)

      if len(candidates) < 5:
          for ticker in self.tickers:
            if self._levenshtein_distance(query.upper(), ticker[:len(query)]) <= max_distance - 1:
              candidates.add(ticker)
      # Score and rank all candidates
      scored_candidates = []
      for ticker in candidates:
          score = 0

          # High bonus for prefix matches
          if ticker in trie_matches:
              score += 100

          # Medium bonus if query appears anywhere in ticker
          elif query.upper() in ticker:
              score += 50

          # Add n-gram similarity score
          score += self._get_ngram_score(query.upper(), ticker)

          # Penalty based on position where match occurs
          if not ticker.startswith(query.upper()):
              position_penalty = ticker.find(query.upper())
              if position_penalty >= 0:
                score -= position_penalty * 2

          if score > 0:
            scored_candidates.append((ticker, score))

    #   if len(scored_candidates) < 5:
    #     first_char = query[0].upper()
    #     if first_char in result_dict:
    #         tickers_to_add = result_dict[first_char]

    #         tickers_to_add_tuples = [(ticker, 0) for ticker in tickers_to_add]
    #         existing_tickers = Counter([ticker for ticker, score in scored_candidates])

    #         for ticker, score in tickers_to_add_tuples:
    #             if existing_tickers[ticker] == 0:
    #                 scored_candidates.append((ticker, score))

      scored_candidates.sort(key=lambda x: x[1], reverse=True)
      return [ticker for ticker, _ in scored_candidates[:max_suggestions]]
    

def get_suggestions_api(query, ticker_list):

    autocomplete = HybridAutocomplete(tickers=ticker_list, ngram_size=3)
    suggestions = autocomplete.get_suggestions(query, max_suggestions=10)

    return suggestions




