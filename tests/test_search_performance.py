"""
اختبار أداء البحث الدلالي - Semantic Search Performance Testing
يقيس: Precision, Recall, F1-Score, MRR
"""

import json
import time
import numpy as np
from typing import List, Dict, Tuple
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إعدادات الاختبار
API_BASE_URL = "http://localhost:8000"
RESULTS_DIR = Path("test_results")
RESULTS_DIR.mkdir(exist_ok=True)


class SearchPerformanceTester:
    """اختبار شامل لأداء البحث الدلالي"""

    def __init__(self, api_url: str = API_BASE_URL):
        self.api_url = api_url
        self.test_queries = []
        self.results = {
            'precision_at_k': {},
            'recall_at_k': {},
            'f1_at_k': {},
            'mrr': 0.0,
            'response_times': [],
            'detailed_results': []
        }

    def load_test_queries(self, queries_file: str = None):
        """
        تحميل استعلامات الاختبار مع الإجابات الصحيحة

        صيغة الملف JSON:
        [
            {
                "query": "ما هي أحدث التطورات في الطاقة المتجددة؟",
                "language": "arabic",
                "relevant_docs": ["doc1.pdf", "doc5.pdf"],
                "top_relevant": "doc1.pdf"
            }
        ]
        """
        if queries_file and Path(queries_file).exists():
            with open(queries_file, 'r', encoding='utf-8') as f:
                self.test_queries = json.load(f)
        else:
            # إنشاء استعلامات اختبار نموذجية
            self.test_queries = self._generate_sample_queries()

        logger.info(f"تم تحميل {len(self.test_queries)} استعلام اختبار")
        return self.test_queries

    def _generate_sample_queries(self) -> List[Dict]:
        """إنشاء استعلامات اختبار نموذجية"""
        return [
            {
                "query": "renewable energy developments",
                "language": "english",
                "relevant_docs": [],  # سيتم ملؤها يدويًا
                "top_relevant": None
            },
            {
                "query": "التطورات الاقتصادية في المنطقة",
                "language": "arabic",
                "relevant_docs": [],
                "top_relevant": None
            },
            # يمكن إضافة المزيد...
        ]

    def search(self, query: str, top_k: int = 5, language: str = None) -> Tuple[List[Dict], float]:
        """
        إجراء بحث عبر API

        Returns:
            (results, response_time)
        """
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.api_url}/api/search",
                json={
                    "query": query,
                    "search_mode": "unified",
                    "top_k": top_k,
                    "language": language
                },
                timeout=30
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                return data.get('results', []), response_time
            else:
                logger.error(f"فشل البحث: {response.status_code}")
                return [], response_time

        except Exception as e:
            logger.error(f"خطأ في البحث: {e}")
            response_time = time.time() - start_time
            return [], response_time

    def calculate_precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        حساب Precision@k

        Precision@k = |Relevant ∩ Retrieved@k| / k
        """
        if not retrieved or k == 0:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)

        intersection = retrieved_at_k & relevant_set

        return len(intersection) / k

    def calculate_recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        حساب Recall@k

        Recall@k = |Relevant ∩ Retrieved@k| / |Relevant|
        """
        if not relevant:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)

        intersection = retrieved_at_k & relevant_set

        return len(intersection) / len(relevant_set)

    def calculate_f1_at_k(self, precision: float, recall: float) -> float:
        """
        حساب F1-Score@k

        F1@k = 2 × (Precision@k × Recall@k) / (Precision@k + Recall@k)
        """
        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def calculate_mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        حساب Mean Reciprocal Rank

        MRR = 1 / rank_of_first_relevant
        """
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / i

        return 0.0

    def run_tests(self, k_values: List[int] = [1, 3, 5, 10]):
        """
        تشغيل جميع الاختبارات
        """
        logger.info("="*80)
        logger.info("بدء اختبارات أداء البحث الدلالي")
        logger.info("="*80)

        # تهيئة النتائج
        for k in k_values:
            self.results['precision_at_k'][k] = []
            self.results['recall_at_k'][k] = []
            self.results['f1_at_k'][k] = []

        mrr_scores = []

        # تشغيل الاختبارات لكل استعلام
        for idx, test_case in enumerate(self.test_queries, 1):
            query = test_case['query']
            relevant_docs = test_case.get('relevant_docs', [])
            language = test_case.get('language')

            logger.info(f"\n[{idx}/{len(self.test_queries)}] اختبار: {query[:50]}...")

            # إجراء البحث
            max_k = max(k_values)
            results, response_time = self.search(query, top_k=max_k, language=language)

            # حفظ زمن الاستجابة
            self.results['response_times'].append(response_time)

            # استخراج أسماء المستندات المسترجعة
            retrieved_docs = [r['source_file'] for r in results]

            # حساب المقاييس لكل k
            query_metrics = {
                'query': query,
                'language': language,
                'relevant_count': len(relevant_docs),
                'response_time': response_time,
                'metrics': {}
            }

            for k in k_values:
                precision = self.calculate_precision_at_k(retrieved_docs, relevant_docs, k)
                recall = self.calculate_recall_at_k(retrieved_docs, relevant_docs, k)
                f1 = self.calculate_f1_at_k(precision, recall)

                self.results['precision_at_k'][k].append(precision)
                self.results['recall_at_k'][k].append(recall)
                self.results['f1_at_k'][k].append(f1)

                query_metrics['metrics'][f'k={k}'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

                logger.info(f"  k={k}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

            # حساب MRR
            mrr = self.calculate_mrr(retrieved_docs, relevant_docs)
            mrr_scores.append(mrr)
            query_metrics['mrr'] = mrr

            self.results['detailed_results'].append(query_metrics)

        # حساب المتوسطات
        self.results['avg_precision_at_k'] = {
            k: np.mean(scores) for k, scores in self.results['precision_at_k'].items()
        }
        self.results['avg_recall_at_k'] = {
            k: np.mean(scores) for k, scores in self.results['recall_at_k'].items()
        }
        self.results['avg_f1_at_k'] = {
            k: np.mean(scores) for k, scores in self.results['f1_at_k'].items()
        }
        self.results['mrr'] = np.mean(mrr_scores) if mrr_scores else 0.0

        self.results['avg_response_time'] = np.mean(self.results['response_times'])
        self.results['std_response_time'] = np.std(self.results['response_times'])

        logger.info("\n" + "="*80)
        logger.info("النتائج النهائية")
        logger.info("="*80)
        self.print_summary()

        return self.results

    def print_summary(self):
        """طباعة ملخص النتائج"""
        print("\n📊 ملخص أداء البحث الدلالي\n")

        print("جدول: مقاييس الأداء عند قيم k مختلفة")
        print("-" * 70)
        print(f"{'المقياس':<20} {'k=1':<12} {'k=3':<12} {'k=5':<12} {'k=10':<12}")
        print("-" * 70)

        # Precision
        precision_row = "Precision"
        for k in [1, 3, 5, 10]:
            if k in self.results['avg_precision_at_k']:
                precision_row += f"{self.results['avg_precision_at_k'][k]:>12.3f}"
        print(precision_row)

        # Recall
        recall_row = "Recall"
        for k in [1, 3, 5, 10]:
            if k in self.results['avg_recall_at_k']:
                recall_row += f"{self.results['avg_recall_at_k'][k]:>12.3f}"
        print(recall_row)

        # F1-Score
        f1_row = "F1-Score"
        for k in [1, 3, 5, 10]:
            if k in self.results['avg_f1_at_k']:
                f1_row += f"{self.results['avg_f1_at_k'][k]:>12.3f}"
        print(f1_row)

        print("-" * 70)
        print(f"\nMRR (Mean Reciprocal Rank): {self.results['mrr']:.3f}")
        print(f"\nمتوسط زمن الاستجابة: {self.results['avg_response_time']:.3f} ثانية")
        print(f"الانحراف المعياري: {self.results['std_response_time']:.3f} ثانية")
        print(f"عدد الاستعلامات المختبرة: {len(self.test_queries)}")

    def save_results(self, filename: str = "search_performance_results.json"):
        """حفظ النتائج إلى ملف JSON"""
        output_path = RESULTS_DIR / filename

        # تحويل numpy types إلى Python types
        results_to_save = {
            'avg_precision_at_k': {k: float(v) for k, v in self.results['avg_precision_at_k'].items()},
            'avg_recall_at_k': {k: float(v) for k, v in self.results['avg_recall_at_k'].items()},
            'avg_f1_at_k': {k: float(v) for k, v in self.results['avg_f1_at_k'].items()},
            'mrr': float(self.results['mrr']),
            'avg_response_time': float(self.results['avg_response_time']),
            'std_response_time': float(self.results['std_response_time']),
            'num_queries': len(self.test_queries),
            'detailed_results': self.results['detailed_results']
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✅ تم حفظ النتائج في: {output_path}")
        return output_path

    def generate_latex_table(self, filename: str = "search_results_table.tex"):
        """توليد جدول LaTeX للبحث العلمي"""
        output_path = RESULTS_DIR / filename

        latex_content = r"""\begin{table}[h]
\centering
\caption{مقاييس أداء البحث الدلالي}
\label{tab:search_performance}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{المقياس} & \textbf{k=1} & \textbf{k=3} & \textbf{k=5} & \textbf{k=10} \\
\hline
"""

        # Precision row
        latex_content += "Precision & "
        latex_content += " & ".join([
            f"{self.results['avg_precision_at_k'].get(k, 0):.2f}"
            for k in [1, 3, 5, 10]
        ])
        latex_content += " \\\\\n"

        # Recall row
        latex_content += "Recall & "
        latex_content += " & ".join([
            f"{self.results['avg_recall_at_k'].get(k, 0):.2f}"
            for k in [1, 3, 5, 10]
        ])
        latex_content += " \\\\\n"

        # F1-Score row
        latex_content += "F1-Score & "
        latex_content += " & ".join([
            f"{self.results['avg_f1_at_k'].get(k, 0):.2f}"
            for k in [1, 3, 5, 10]
        ])
        latex_content += " \\\\\n"

        latex_content += r"""\hline
\end{tabular}
\end{table}

\noindent
MRR (Mean Reciprocal Rank): """ + f"{self.results['mrr']:.2f}"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)

        logger.info(f"✅ تم حفظ جدول LaTeX في: {output_path}")
        return output_path


def main():
    """التشغيل الرئيسي"""
    print("\n" + "="*80)
    print("🔍 برنامج اختبار أداء البحث الدلالي")
    print("="*80 + "\n")

    # إنشاء مثيل الاختبار
    tester = SearchPerformanceTester()

    # تحميل استعلامات الاختبار
    # يمكن تمرير ملف JSON بالاستعلامات
    tester.load_test_queries("test_data/search_queries.json")

    # تشغيل الاختبارات
    results = tester.run_tests(k_values=[1, 3, 5, 10])

    # حفظ النتائج
    tester.save_results()
    tester.generate_latex_table()

    print("\n✅ اكتملت جميع الاختبارات بنجاح!")


if __name__ == "__main__":
    main()
