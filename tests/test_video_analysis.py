"""
اختبار أداء تحليل الفيديو - Video Analysis Performance Testing
يقيس: WER (Word Error Rate), CER (Character Error Rate), ROUGE Scores
"""

import json
import time
import numpy as np
from typing import List, Dict, Tuple
import requests
from pathlib import Path
import logging
from difflib import SequenceMatcher
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إعدادات الاختبار
API_BASE_URL = "http://localhost:8000"
RESULTS_DIR = Path("test_results")
RESULTS_DIR.mkdir(exist_ok=True)


class VideoAnalysisTester:
    """اختبار شامل لأداء تحليل الفيديو"""

    def __init__(self, api_url: str = API_BASE_URL):
        self.api_url = api_url
        self.test_videos = []
        self.results = {
            'wer_scores': [],
            'cer_scores': [],
            'rouge_scores': {
                'rouge1': [],
                'rouge2': [],
                'rougeL': []
            },
            'processing_times': [],
            'detailed_results': []
        }

    def load_test_videos(self, videos_file: str = None):
        """
        تحميل فيديوهات الاختبار مع النصوص المرجعية

        صيغة الملف JSON:
        [
            {
                "video_filename": "test_video1.mp4",
                "language": "arabic",
                "reference_transcript": "النص الصوتي المرجعي الصحيح...",
                "reference_summary": "الملخص المرجعي...",
                "duration_seconds": 300
            }
        ]
        """
        if videos_file and Path(videos_file).exists():
            with open(videos_file, 'r', encoding='utf-8') as f:
                self.test_videos = json.load(f)
        else:
            # إنشاء بيانات اختبار نموذجية
            self.test_videos = self._generate_sample_data()

        logger.info(f"تم تحميل {len(self.test_videos)} فيديو للاختبار")
        return self.test_videos

    def _generate_sample_data(self) -> List[Dict]:
        """إنشاء بيانات اختبار نموذجية"""
        return [
            {
                "video_filename": "sample1.mp4",
                "language": "arabic",
                "reference_transcript": "",  # يجب ملؤه يدويًا
                "reference_summary": "",
                "duration_seconds": 0
            }
        ]

    def analyze_video(self, video_filename: str, language: str = "arabic") -> Tuple[Dict, float]:
        """
        تحليل فيديو عبر API

        Returns:
            (results, processing_time)
        """
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.api_url}/api/video/analyze_existing",
                json={
                    "video_filename": video_filename,
                    "num_frames": 10,
                    "output_language": language
                },
                timeout=300  # 5 minutes timeout
            )

            processing_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                return data, processing_time
            else:
                logger.error(f"فشل تحليل الفيديو: {response.status_code}")
                return {}, processing_time

        except Exception as e:
            logger.error(f"خطأ في تحليل الفيديو: {e}")
            processing_time = time.time() - start_time
            return {}, processing_time

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        حساب معدل خطأ الكلمات (Word Error Rate)

        WER = (S + D + I) / N

        حيث:
        S = Substitutions (الاستبدالات)
        D = Deletions (الحذف)
        I = Insertions (الإضافات)
        N = عدد الكلمات في النص المرجعي
        """
        # تنظيف وتقسيم النصوص
        ref_words = self._tokenize(reference)
        hyp_words = self._tokenize(hypothesis)

        # استخدام مسافة Levenshtein
        distances = self._levenshtein_distance(ref_words, hyp_words)

        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0

        wer = distances / len(ref_words)
        return min(wer, 1.0)  # WER لا يتجاوز 100%

    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """
        حساب معدل خطأ الأحرف (Character Error Rate)

        CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars)
        """
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))

        distances = self._levenshtein_distance(ref_chars, hyp_chars)

        if len(ref_chars) == 0:
            return 0.0 if len(hyp_chars) == 0 else 1.0

        cer = distances / len(ref_chars)
        return min(cer, 1.0)

    def _tokenize(self, text: str) -> List[str]:
        """تقسيم النص إلى كلمات"""
        # إزالة علامات الترقيم وتحويل لأحرف صغيرة
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def _levenshtein_distance(self, seq1: List, seq2: List) -> int:
        """
        حساب مسافة Levenshtein (عدد التعديلات المطلوبة)
        """
        m, n = len(seq1), len(seq2)

        # مصفوفة البرمجة الديناميكية
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # التهيئة
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # ملء المصفوفة
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # حذف
                        dp[i][j-1],      # إضافة
                        dp[i-1][j-1]     # استبدال
                    )

        return dp[m][n]

    def calculate_rouge_scores(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        حساب ROUGE scores

        ROUGE-1: تطابق الكلمات المفردة
        ROUGE-2: تطابق ثنائيات الكلمات (bigrams)
        ROUGE-L: أطول تسلسل مشترك
        """
        ref_tokens = self._tokenize(reference)
        hyp_tokens = self._tokenize(hypothesis)

        # ROUGE-1: Unigram overlap
        rouge1 = self._calculate_rouge_n(ref_tokens, hyp_tokens, n=1)

        # ROUGE-2: Bigram overlap
        rouge2 = self._calculate_rouge_n(ref_tokens, hyp_tokens, n=2)

        # ROUGE-L: Longest Common Subsequence
        rougeL = self._calculate_rouge_l(ref_tokens, hyp_tokens)

        return {
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL
        }

    def _calculate_rouge_n(self, ref_tokens: List[str], hyp_tokens: List[str], n: int) -> float:
        """
        حساب ROUGE-N

        F-Score = 2 * (Precision * Recall) / (Precision + Recall)
        """
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        hyp_ngrams = self._get_ngrams(hyp_tokens, n)

        if len(ref_ngrams) == 0 or len(hyp_ngrams) == 0:
            return 0.0

        # حساب التقاطع
        overlap = sum((ref_ngrams & hyp_ngrams).values())

        # Recall: overlap / ref_count
        recall = overlap / sum(ref_ngrams.values())

        # Precision: overlap / hyp_count
        precision = overlap / sum(hyp_ngrams.values())

        if precision + recall == 0:
            return 0.0

        # F1-Score
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple, int]:
        """استخراج n-grams من قائمة الكلمات"""
        from collections import Counter

        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))

        return Counter(ngrams)

    def _calculate_rouge_l(self, ref_tokens: List[str], hyp_tokens: List[str]) -> float:
        """
        حساب ROUGE-L باستخدام أطول تسلسل مشترك (LCS)
        """
        lcs_length = self._lcs_length(ref_tokens, hyp_tokens)

        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            return 0.0

        recall = lcs_length / len(ref_tokens)
        precision = lcs_length / len(hyp_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _lcs_length(self, seq1: List, seq2: List) -> int:
        """حساب طول أطول تسلسل مشترك"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def run_tests(self):
        """تشغيل جميع اختبارات تحليل الفيديو"""
        logger.info("="*80)
        logger.info("بدء اختبارات تحليل الفيديو")
        logger.info("="*80)

        for idx, test_video in enumerate(self.test_videos, 1):
            video_filename = test_video['video_filename']
            language = test_video.get('language', 'arabic')
            ref_transcript = test_video.get('reference_transcript', '')
            ref_summary = test_video.get('reference_summary', '')

            logger.info(f"\n[{idx}/{len(self.test_videos)}] تحليل: {video_filename}")

            # تحليل الفيديو
            result, processing_time = self.analyze_video(video_filename, language)

            if not result:
                logger.warning(f"فشل تحليل {video_filename}")
                continue

            # حفظ زمن المعالجة
            self.results['processing_times'].append({
                'video': video_filename,
                'duration': test_video.get('duration_seconds', 0),
                'processing_time': processing_time
            })

            # استخراج النتائج
            hypothesis_transcript = result.get('audio_transcript', '')
            hypothesis_summary = result.get('summary', '')

            # حساب WER و CER
            if ref_transcript:
                wer = self.calculate_wer(ref_transcript, hypothesis_transcript)
                cer = self.calculate_cer(ref_transcript, hypothesis_transcript)

                self.results['wer_scores'].append(wer)
                self.results['cer_scores'].append(cer)

                logger.info(f"  WER: {wer*100:.2f}%")
                logger.info(f"  CER: {cer*100:.2f}%")
            else:
                wer, cer = None, None
                logger.warning("  ⚠️  لا يوجد نص مرجعي للمقارنة")

            # حساب ROUGE scores
            if ref_summary:
                rouge_scores = self.calculate_rouge_scores(ref_summary, hypothesis_summary)

                self.results['rouge_scores']['rouge1'].append(rouge_scores['rouge1'])
                self.results['rouge_scores']['rouge2'].append(rouge_scores['rouge2'])
                self.results['rouge_scores']['rougeL'].append(rouge_scores['rougeL'])

                logger.info(f"  ROUGE-1: {rouge_scores['rouge1']:.3f}")
                logger.info(f"  ROUGE-2: {rouge_scores['rouge2']:.3f}")
                logger.info(f"  ROUGE-L: {rouge_scores['rougeL']:.3f}")
            else:
                rouge_scores = None
                logger.warning("  ⚠️  لا يوجد ملخص مرجعي للمقارنة")

            # حفظ النتائج التفصيلية
            self.results['detailed_results'].append({
                'video_filename': video_filename,
                'language': language,
                'processing_time': processing_time,
                'wer': wer,
                'cer': cer,
                'rouge_scores': rouge_scores,
                'transcript_length': len(hypothesis_transcript.split()),
                'summary_length': len(hypothesis_summary.split())
            })

        logger.info("\n" + "="*80)
        logger.info("النتائج النهائية")
        logger.info("="*80)
        self.print_summary()

        return self.results

    def print_summary(self):
        """طباعة ملخص النتائج"""
        print("\n📊 ملخص أداء تحليل الفيديو\n")

        # WER & CER
        if self.results['wer_scores']:
            avg_wer = np.mean(self.results['wer_scores']) * 100
            std_wer = np.std(self.results['wer_scores']) * 100

            avg_cer = np.mean(self.results['cer_scores']) * 100
            std_cer = np.std(self.results['cer_scores']) * 100

            print("جدول: دقة النسخ الصوتي")
            print("-" * 50)
            print(f"{'المقياس':<20} {'المتوسط':<15} {'الانحراف المعياري':<20}")
            print("-" * 50)
            print(f"{'WER (%)':<20} {avg_wer:>10.2f}%     {std_wer:>10.2f}%")
            print(f"{'CER (%)':<20} {avg_cer:>10.2f}%     {std_cer:>10.2f}%")
            print("-" * 50)

        # ROUGE Scores
        if self.results['rouge_scores']['rouge1']:
            print("\nجدول: جودة التلخيص (ROUGE Scores)")
            print("-" * 50)
            print(f"{'المقياس':<20} {'المتوسط':<15} {'الانحراف المعياري':<20}")
            print("-" * 50)

            for metric in ['rouge1', 'rouge2', 'rougeL']:
                scores = self.results['rouge_scores'][metric]
                avg = np.mean(scores)
                std = np.std(scores)
                print(f"{metric.upper():<20} {avg:>10.3f}       {std:>10.3f}")

            print("-" * 50)

        # Processing Times
        if self.results['processing_times']:
            times = [t['processing_time'] for t in self.results['processing_times']]
            avg_time = np.mean(times)
            std_time = np.std(times)

            print(f"\nمتوسط زمن المعالجة: {avg_time:.2f} ثانية (±{std_time:.2f})")
            print(f"عدد الفيديوهات المختبرة: {len(self.test_videos)}")

    def save_results(self, filename: str = "video_analysis_results.json"):
        """حفظ النتائج إلى ملف JSON"""
        output_path = RESULTS_DIR / filename

        results_to_save = {
            'avg_wer': float(np.mean(self.results['wer_scores'])) if self.results['wer_scores'] else None,
            'avg_cer': float(np.mean(self.results['cer_scores'])) if self.results['cer_scores'] else None,
            'avg_rouge1': float(np.mean(self.results['rouge_scores']['rouge1'])) if self.results['rouge_scores']['rouge1'] else None,
            'avg_rouge2': float(np.mean(self.results['rouge_scores']['rouge2'])) if self.results['rouge_scores']['rouge2'] else None,
            'avg_rougeL': float(np.mean(self.results['rouge_scores']['rougeL'])) if self.results['rouge_scores']['rougeL'] else None,
            'num_videos': len(self.test_videos),
            'processing_times': self.results['processing_times'],
            'detailed_results': self.results['detailed_results']
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✅ تم حفظ النتائج في: {output_path}")
        return output_path


def main():
    """التشغيل الرئيسي"""
    print("\n" + "="*80)
    print("🎥 برنامج اختبار أداء تحليل الفيديو")
    print("="*80 + "\n")

    # إنشاء مثيل الاختبار
    tester = VideoAnalysisTester()

    # تحميل فيديوهات الاختبار
    tester.load_test_videos("test_data/video_test_cases.json")

    # تشغيل الاختبارات
    results = tester.run_tests()

    # حفظ النتائج
    tester.save_results()

    print("\n✅ اكتملت جميع الاختبارات بنجاح!")


if __name__ == "__main__":
    main()
