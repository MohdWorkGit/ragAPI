"""
اختبار أداء استخراج الكتّاب - Writer Extraction Performance Testing
يقيس: Precision, Recall, F1-Score, Fuzzy Matching Accuracy
"""

import json
import time
import numpy as np
from typing import List, Dict, Set
from pathlib import Path
import logging
import sys

# إضافة المسار الجذري للمشروع
sys.path.insert(0, str(Path(__file__).parent.parent))

from writer_manager import WriterManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إعدادات الاختبار
RESULTS_DIR = Path("test_results")
RESULTS_DIR.mkdir(exist_ok=True)


class WriterExtractionTester:
    """اختبار شامل لأداء استخراج الكتّاب"""

    def __init__(self):
        self.writer_manager = WriterManager(db_folder="test_writers_db")
        self.test_documents = []
        self.results = {
            'extraction_results': [],
            'fuzzy_matching_results': [],
            'precision_scores': [],
            'recall_scores': [],
            'f1_scores': [],
            'processing_times': []
        }

    def load_test_documents(self, documents_file: str = None):
        """
        تحميل مستندات الاختبار مع الكتّاب المرجعيين

        صيغة الملف JSON:
        [
            {
                "document_name": "article1.pdf",
                "document_path": "test_data/article1.pdf",
                "ground_truth_writers": [
                    "محمد أحمد",
                    "Sara Johnson"
                ]
            }
        ]
        """
        if documents_file and Path(documents_file).exists():
            with open(documents_file, 'r', encoding='utf-8') as f:
                self.test_documents = json.load(f)
        else:
            # إنشاء بيانات اختبار نموذجية
            self.test_documents = self._generate_sample_data()

        logger.info(f"تم تحميل {len(self.test_documents)} مستند للاختبار")
        return self.test_documents

    def _generate_sample_data(self) -> List[Dict]:
        """إنشاء بيانات اختبار نموذجية"""
        return [
            {
                "document_name": "sample_article.pdf",
                "document_path": "test_data/sample_article.pdf",
                "ground_truth_writers": []  # يجب ملؤه يدويًا
            }
        ]

    def extract_writers_from_document(self, document_path: str, document_name: str) -> List[str]:
        """
        استخراج الكتّاب من مستند

        Returns:
            قائمة أسماء الكتّاب المستخرجة
        """
        start_time = time.time()

        # استخراج الكتّاب باستخدام WriterManager
        if document_path.endswith('.pdf'):
            writer_names = self.writer_manager.extract_writer_names_from_pdf(document_path)
        else:
            # للملفات النصية
            with open(document_path, 'r', encoding='utf-8') as f:
                text = f.read()
            writer_names = self.writer_manager.extract_writer_names(text)

        processing_time = time.time() - start_time
        self.results['processing_times'].append({
            'document': document_name,
            'time': processing_time
        })

        return writer_names

    def calculate_metrics(self, predicted: Set[str], ground_truth: Set[str]) -> Dict[str, float]:
        """
        حساب Precision, Recall, F1-Score

        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        if len(predicted) == 0 and len(ground_truth) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

        # True Positives: استخراجات صحيحة
        tp = len(predicted & ground_truth)

        # False Positives: استخراجات خاطئة
        fp = len(predicted - ground_truth)

        # False Negatives: كتّاب مفقودون
        fn = len(ground_truth - predicted)

        # حساب Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # حساب Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # حساب F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def test_fuzzy_matching(self, test_pairs: List[Dict] = None):
        """
        اختبار دقة المطابقة الضبابية

        test_pairs format:
        [
            {
                "name1": "محمد علي",
                "name2": "محمد علي أحمد",
                "should_match": True
            }
        ]
        """
        if test_pairs is None:
            test_pairs = self._generate_fuzzy_test_pairs()

        logger.info("\n" + "="*80)
        logger.info("اختبار المطابقة الضبابية")
        logger.info("="*80)

        correct_matches = 0
        total_pairs = len(test_pairs)

        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        threshold_results = {th: {'correct': 0, 'incorrect': 0} for th in thresholds}

        for pair in test_pairs:
            name1 = pair['name1']
            name2 = pair['name2']
            should_match = pair['should_match']

            similarity = self.writer_manager._calculate_name_similarity(name1, name2)

            logger.debug(f"{name1} vs {name2}: similarity={similarity:.3f}, expected={should_match}")

            # اختبار مع عتبات مختلفة
            for threshold in thresholds:
                predicted_match = similarity >= threshold
                is_correct = (predicted_match == should_match)

                if is_correct:
                    threshold_results[threshold]['correct'] += 1
                else:
                    threshold_results[threshold]['incorrect'] += 1

        # حساب الدقة لكل عتبة
        logger.info("\nنتائج المطابقة الضبابية:")
        logger.info("-" * 60)
        logger.info(f"{'العتبة':<10} {'صحيح':<10} {'خاطئ':<10} {'الدقة (%)':<15}")
        logger.info("-" * 60)

        best_threshold = None
        best_accuracy = 0

        for threshold in thresholds:
            correct = threshold_results[threshold]['correct']
            incorrect = threshold_results[threshold]['incorrect']
            accuracy = (correct / total_pairs) * 100

            logger.info(f"{threshold:<10.2f} {correct:<10} {incorrect:<10} {accuracy:<15.2f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

            self.results['fuzzy_matching_results'].append({
                'threshold': threshold,
                'accuracy': accuracy / 100,
                'correct': correct,
                'incorrect': incorrect
            })

        logger.info("-" * 60)
        logger.info(f"أفضل عتبة: {best_threshold} بدقة {best_accuracy:.2f}%")

        return best_threshold, best_accuracy

    def _generate_fuzzy_test_pairs(self) -> List[Dict]:
        """إنشاء أزواج اختبار للمطابقة الضبابية"""
        return [
            # أزواج يجب أن تتطابق
            {"name1": "محمد أحمد", "name2": "محمد أحمد علي", "should_match": True},
            {"name1": "Sara Johnson", "name2": "Sara J. Johnson", "should_match": True},
            {"name1": "Dr. Ahmed Hassan", "name2": "Ahmed Hassan", "should_match": True},
            {"name1": "علي محمد", "name2": "محمد علي", "should_match": True},
            {"name1": "John Smith", "name2": "Jon Smith", "should_match": True},  # خطأ إملائي

            # أزواج لا يجب أن تتطابق
            {"name1": "محمد أحمد", "name2": "أحمد محمود", "should_match": False},
            {"name1": "Sara Johnson", "name2": "Sarah Williams", "should_match": False},
            {"name1": "Ahmed Ali", "name2": "Mohamed Ali", "should_match": False},
            {"name1": "John Smith", "name2": "Jane Smith", "should_match": False},
            {"name1": "علي حسن", "name2": "حسين علي", "should_match": False},

            # حالات حدية
            {"name1": "Dr. Mohamed", "name2": "Prof. Mohamed", "should_match": True},
            {"name1": "أ.د. محمد", "name2": "د. محمد", "should_match": True},
        ]

    def test_extraction_techniques(self, test_pdf: str = None):
        """
        اختبار مقارنة بين تقنيات الاستخراج المختلفة
        """
        if not test_pdf or not Path(test_pdf).exists():
            logger.warning("لا يوجد ملف PDF للاختبار")
            return

        logger.info("\n" + "="*80)
        logger.info("مقارنة تقنيات الاستخراج")
        logger.info("="*80)

        # استخراج باستخدام التقنيات المختلفة
        blocks = self.writer_manager.extract_blocks_from_pdf(test_pdf)

        if blocks:
            # 1. تحليل البنية (Layout Analysis)
            headline_blocks = self.writer_manager.identify_headline_blocks(blocks)
            layout_writers = self.writer_manager.extract_byline_patterns(blocks)

            # 2. NER
            ner_writers = self.writer_manager.extract_names_with_ner(blocks)

            # 3. الدمج الكامل
            all_writers = self.writer_manager.extract_writer_names_from_pdf(test_pdf)

            logger.info(f"\nتحليل البنية: {len(layout_writers)} كاتب")
            logger.info(f"  {layout_writers}")

            logger.info(f"\nNER: {len(ner_writers)} كاتب")
            logger.info(f"  {ner_writers}")

            logger.info(f"\nالنهج الهجين (الكل): {len(all_writers)} كاتب")
            logger.info(f"  {all_writers}")

            return {
                'layout': layout_writers,
                'ner': ner_writers,
                'hybrid': all_writers
            }

    def run_tests(self):
        """تشغيل جميع اختبارات استخراج الكتّاب"""
        logger.info("="*80)
        logger.info("بدء اختبارات استخراج الكتّاب")
        logger.info("="*80)

        # 1. اختبار الاستخراج من المستندات
        for idx, test_doc in enumerate(self.test_documents, 1):
            document_name = test_doc['document_name']
            document_path = test_doc['document_path']
            ground_truth = set(test_doc.get('ground_truth_writers', []))

            logger.info(f"\n[{idx}/{len(self.test_documents)}] معالجة: {document_name}")

            if not Path(document_path).exists():
                logger.warning(f"  ⚠️  الملف غير موجود: {document_path}")
                continue

            # استخراج الكتّاب
            extracted_writers = self.extract_writers_from_document(document_path, document_name)
            predicted = set(extracted_writers)

            logger.info(f"  الكتّاب المستخرجون: {extracted_writers}")
            logger.info(f"  الكتّاب المرجعيون: {list(ground_truth)}")

            # حساب المقاييس
            if ground_truth:
                metrics = self.calculate_metrics(predicted, ground_truth)

                self.results['precision_scores'].append(metrics['precision'])
                self.results['recall_scores'].append(metrics['recall'])
                self.results['f1_scores'].append(metrics['f1'])

                logger.info(f"  Precision: {metrics['precision']:.3f}")
                logger.info(f"  Recall: {metrics['recall']:.3f}")
                logger.info(f"  F1-Score: {metrics['f1']:.3f}")

                self.results['extraction_results'].append({
                    'document': document_name,
                    'predicted': list(predicted),
                    'ground_truth': list(ground_truth),
                    'metrics': metrics
                })
            else:
                logger.warning("  ⚠️  لا توجد بيانات مرجعية للمقارنة")

        # 2. اختبار المطابقة الضبابية
        self.test_fuzzy_matching()

        # طباعة الملخص
        logger.info("\n" + "="*80)
        logger.info("النتائج النهائية")
        logger.info("="*80)
        self.print_summary()

        return self.results

    def print_summary(self):
        """طباعة ملخص النتائج"""
        print("\n📊 ملخص أداء استخراج الكتّاب\n")

        if self.results['precision_scores']:
            avg_precision = np.mean(self.results['precision_scores']) * 100
            avg_recall = np.mean(self.results['recall_scores']) * 100
            avg_f1 = np.mean(self.results['f1_scores']) * 100

            print("جدول: مقاييس الاستخراج")
            print("-" * 50)
            print(f"{'المقياس':<20} {'المتوسط (%)':<15}")
            print("-" * 50)
            print(f"{'Precision':<20} {avg_precision:>10.2f}%")
            print(f"{'Recall':<20} {avg_recall:>10.2f}%")
            print(f"{'F1-Score':<20} {avg_f1:>10.2f}%")
            print("-" * 50)

        if self.results['processing_times']:
            times = [t['time'] for t in self.results['processing_times']]
            avg_time = np.mean(times)
            print(f"\nمتوسط زمن المعالجة: {avg_time:.3f} ثانية")
            print(f"عدد المستندات المختبرة: {len(self.test_documents)}")

    def save_results(self, filename: str = "writer_extraction_results.json"):
        """حفظ النتائج إلى ملف JSON"""
        output_path = RESULTS_DIR / filename

        results_to_save = {
            'avg_precision': float(np.mean(self.results['precision_scores'])) if self.results['precision_scores'] else None,
            'avg_recall': float(np.mean(self.results['recall_scores'])) if self.results['recall_scores'] else None,
            'avg_f1': float(np.mean(self.results['f1_scores'])) if self.results['f1_scores'] else None,
            'num_documents': len(self.test_documents),
            'fuzzy_matching_results': self.results['fuzzy_matching_results'],
            'extraction_results': self.results['extraction_results'],
            'processing_times': self.results['processing_times']
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✅ تم حفظ النتائج في: {output_path}")
        return output_path


def main():
    """التشغيل الرئيسي"""
    print("\n" + "="*80)
    print("✍️  برنامج اختبار استخراج الكتّاب")
    print("="*80 + "\n")

    # إنشاء مثيل الاختبار
    tester = WriterExtractionTester()

    # تحميل مستندات الاختبار
    tester.load_test_documents("test_data/writer_test_documents.json")

    # تشغيل الاختبارات
    results = tester.run_tests()

    # حفظ النتائج
    tester.save_results()

    print("\n✅ اكتملت جميع الاختبارات بنجاح!")


if __name__ == "__main__":
    main()
