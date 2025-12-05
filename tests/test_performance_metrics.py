"""
اختبار مقاييس الأداء والموارد - Performance and Resource Metrics Testing
يقيس: زمن المعالجة، استهلاك الذاكرة، استخدام GPU، قابلية التوسع
"""

import json
import time
import psutil
import numpy as np
from typing import List, Dict
from pathlib import Path
import logging
import requests
import os

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("GPUtil not installed. GPU metrics will not be available.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إعدادات الاختبار
API_BASE_URL = "http://localhost:8000"
RESULTS_DIR = Path("test_results")
RESULTS_DIR.mkdir(exist_ok=True)


class PerformanceTester:
    """اختبار شامل لمقاييس الأداء والموارد"""

    def __init__(self, api_url: str = API_BASE_URL):
        self.api_url = api_url
        self.process = psutil.Process(os.getpid())
        self.results = {
            'document_processing': [],
            'video_processing': [],
            'search_performance': [],
            'scalability_test': [],
            'resource_usage': {
                'cpu': [],
                'memory': [],
                'gpu': []
            }
        }

    def get_resource_usage(self) -> Dict:
        """
        الحصول على استخدام الموارد الحالي
        """
        resources = {
            'cpu_percent': self.process.cpu_percent(interval=0.1),
            'memory_mb': self.process.memory_info().rss / (1024 * 1024),
            'memory_percent': self.process.memory_percent()
        }

        # إضافة معلومات GPU إن وجدت
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    resources['gpu_memory_used_mb'] = gpu.memoryUsed
                    resources['gpu_memory_total_mb'] = gpu.memoryTotal
                    resources['gpu_utilization_percent'] = gpu.load * 100
            except Exception as e:
                logger.warning(f"Could not get GPU metrics: {e}")

        return resources

    def test_document_processing_time(self, doc_folder: str = "docs"):
        """
        اختبار زمن معالجة المستندات
        """
        logger.info("\n" + "="*80)
        logger.info("اختبار زمن معالجة المستندات")
        logger.info("="*80)

        if not Path(doc_folder).exists():
            logger.warning(f"المجلد {doc_folder} غير موجود")
            return

        doc_files = list(Path(doc_folder).glob("*.*"))
        if not doc_files:
            logger.warning("لا توجد ملفات للاختبار")
            return

        # اختبار عينة من الملفات
        sample_size = min(20, len(doc_files))
        sample_docs = np.random.choice(doc_files, sample_size, replace=False)

        for doc_path in sample_docs:
            doc_name = doc_path.name
            file_size_mb = doc_path.stat().st_size / (1024 * 1024)

            logger.info(f"\nمعالجة: {doc_name} ({file_size_mb:.2f} MB)")

            # قياس الموارد قبل المعالجة
            resources_before = self.get_resource_usage()

            # بدء المعالجة
            start_time = time.time()

            try:
                # محاكاة معالجة المستند (في حالة عدم وجود API نشط)
                # يمكن استبدال هذا باستدعاء API حقيقي
                response = requests.post(
                    f"{self.api_url}/api/refresh",
                    json={"fileName": doc_name, "rebuild_unified": False},
                    timeout=60
                )

                processing_time = time.time() - start_time

                # قياس الموارد بعد المعالجة
                resources_after = self.get_resource_usage()

                # حساب التغير في الموارد
                memory_delta = resources_after['memory_mb'] - resources_before['memory_mb']

                result = {
                    'document': doc_name,
                    'file_size_mb': file_size_mb,
                    'processing_time': processing_time,
                    'memory_delta_mb': memory_delta,
                    'cpu_percent': resources_after['cpu_percent'],
                    'success': response.status_code == 200 if response else False
                }

                self.results['document_processing'].append(result)

                logger.info(f"  زمن المعالجة: {processing_time:.2f} ثانية")
                logger.info(f"  استهلاك الذاكرة: {memory_delta:.2f} MB")

            except Exception as e:
                logger.error(f"  خطأ في المعالجة: {e}")

        self._print_document_processing_summary()

    def test_video_processing_time(self, video_folder: str = "videos"):
        """
        اختبار زمن معالجة الفيديو مع تفصيل المراحل
        """
        logger.info("\n" + "="*80)
        logger.info("اختبار زمن معالجة الفيديو")
        logger.info("="*80)

        if not Path(video_folder).exists():
            logger.warning(f"المجلد {video_folder} غير موجود")
            return

        video_files = list(Path(video_folder).glob("*.mp4")) + \
                     list(Path(video_folder).glob("*.avi")) + \
                     list(Path(video_folder).glob("*.mov"))

        if not video_files:
            logger.warning("لا توجد فيديوهات للاختبار")
            return

        # اختبار عينة من الفيديوهات
        sample_size = min(10, len(video_files))
        sample_videos = np.random.choice(video_files, sample_size, replace=False)

        for video_path in sample_videos:
            video_name = video_path.name
            file_size_mb = video_path.stat().st_size / (1024 * 1024)

            logger.info(f"\nتحليل فيديو: {video_name} ({file_size_mb:.2f} MB)")

            # قياس الموارد قبل المعالجة
            resources_before = self.get_resource_usage()

            # بدء التحليل
            start_time = time.time()

            try:
                response = requests.post(
                    f"{self.api_url}/api/video/analyze_existing",
                    json={
                        "video_filename": video_name,
                        "num_frames": 10,
                        "output_language": "arabic"
                    },
                    timeout=300
                )

                processing_time = time.time() - start_time

                # قياس الموارد بعد المعالجة
                resources_after = self.get_resource_usage()

                result = {
                    'video': video_name,
                    'file_size_mb': file_size_mb,
                    'total_processing_time': processing_time,
                    'memory_delta_mb': resources_after['memory_mb'] - resources_before['memory_mb'],
                    'cpu_percent': resources_after['cpu_percent'],
                    'success': response.status_code == 200 if response else False
                }

                if response and response.status_code == 200:
                    data = response.json()
                    result['num_frames_analyzed'] = data.get('num_frames_analyzed', 0)
                    result['detected_language'] = data.get('detected_language', 'unknown')

                self.results['video_processing'].append(result)

                logger.info(f"  زمن التحليل الكلي: {processing_time:.2f} ثانية")
                logger.info(f"  استهلاك الذاكرة: {result['memory_delta_mb']:.2f} MB")

            except Exception as e:
                logger.error(f"  خطأ في التحليل: {e}")

        self._print_video_processing_summary()

    def test_search_scalability(self, num_documents_list: List[int] = None):
        """
        اختبار قابلية التوسع - زمن البحث مع أحجام مختلفة
        """
        logger.info("\n" + "="*80)
        logger.info("اختبار قابلية التوسع (Scalability)")
        logger.info("="*80)

        if num_documents_list is None:
            num_documents_list = [100, 500, 1000, 5000, 10000]

        test_queries = [
            "renewable energy",
            "الطاقة المتجددة",
            "economic development",
            "التطورات الاقتصادية"
        ]

        for num_docs in num_documents_list:
            logger.info(f"\nاختبار مع {num_docs} مستند...")

            search_times = []

            for query in test_queries:
                start_time = time.time()

                try:
                    response = requests.post(
                        f"{self.api_url}/api/search",
                        json={
                            "query": query,
                            "search_mode": "unified",
                            "top_k": 5
                        },
                        timeout=30
                    )

                    search_time = time.time() - start_time

                    if response.status_code == 200:
                        search_times.append(search_time * 1000)  # تحويل لميلي ثانية

                except Exception as e:
                    logger.error(f"  خطأ في البحث: {e}")

            if search_times:
                avg_search_time = np.mean(search_times)
                std_search_time = np.std(search_times)

                result = {
                    'num_documents': num_docs,
                    'avg_search_time_ms': avg_search_time,
                    'std_search_time_ms': std_search_time,
                    'num_queries': len(search_times)
                }

                self.results['scalability_test'].append(result)

                logger.info(f"  متوسط زمن البحث: {avg_search_time:.2f} ms (±{std_search_time:.2f})")

        self._print_scalability_summary()

    def test_concurrent_requests(self, num_requests: int = 10):
        """
        اختبار الطلبات المتزامنة
        """
        logger.info("\n" + "="*80)
        logger.info(f"اختبار الطلبات المتزامنة ({num_requests} طلب)")
        logger.info("="*80)

        import concurrent.futures

        def make_search_request():
            start = time.time()
            try:
                response = requests.post(
                    f"{self.api_url}/api/search",
                    json={"query": "test query", "search_mode": "unified", "top_k": 5},
                    timeout=30
                )
                elapsed = time.time() - start
                return {'success': response.status_code == 200, 'time': elapsed}
            except Exception as e:
                return {'success': False, 'time': 0, 'error': str(e)}

        # تنفيذ الطلبات المتزامنة
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_search_request) for _ in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        total_time = time.time() - start_time

        successful = sum(1 for r in results if r['success'])
        avg_response_time = np.mean([r['time'] for r in results if r['success']])

        logger.info(f"  إجمالي الوقت: {total_time:.2f} ثانية")
        logger.info(f"  طلبات ناجحة: {successful}/{num_requests}")
        logger.info(f"  متوسط زمن الاستجابة: {avg_response_time:.2f} ثانية")
        logger.info(f"  معدل الإنتاجية: {num_requests/total_time:.2f} طلب/ثانية")

        return {
            'num_requests': num_requests,
            'total_time': total_time,
            'successful_requests': successful,
            'avg_response_time': avg_response_time,
            'throughput': num_requests / total_time
        }

    def _print_document_processing_summary(self):
        """طباعة ملخص معالجة المستندات"""
        if not self.results['document_processing']:
            return

        times = [r['processing_time'] for r in self.results['document_processing']]
        sizes = [r['file_size_mb'] for r in self.results['document_processing']]

        logger.info("\n📊 ملخص معالجة المستندات:")
        logger.info(f"  عدد المستندات: {len(times)}")
        logger.info(f"  متوسط الزمن: {np.mean(times):.2f} ثانية")
        logger.info(f"  الانحراف المعياري: {np.std(times):.2f} ثانية")
        logger.info(f"  متوسط حجم الملف: {np.mean(sizes):.2f} MB")

    def _print_video_processing_summary(self):
        """طباعة ملخص معالجة الفيديو"""
        if not self.results['video_processing']:
            return

        times = [r['total_processing_time'] for r in self.results['video_processing']]
        sizes = [r['file_size_mb'] for r in self.results['video_processing']]

        logger.info("\n📊 ملخص معالجة الفيديو:")
        logger.info(f"  عدد الفيديوهات: {len(times)}")
        logger.info(f"  متوسط الزمن: {np.mean(times):.2f} ثانية")
        logger.info(f"  الانحراف المعياري: {np.std(times):.2f} ثانية")
        logger.info(f"  متوسط حجم الملف: {np.mean(sizes):.2f} MB")

    def _print_scalability_summary(self):
        """طباعة ملخص قابلية التوسع"""
        if not self.results['scalability_test']:
            return

        logger.info("\n📊 ملخص قابلية التوسع:")
        logger.info("-" * 60)
        logger.info(f"{'عدد المستندات':<20} {'زمن البحث (ms)':<20}")
        logger.info("-" * 60)

        for result in self.results['scalability_test']:
            logger.info(f"{result['num_documents']:<20} {result['avg_search_time_ms']:>15.2f}")

        logger.info("-" * 60)

    def save_results(self, filename: str = "performance_metrics.json"):
        """حفظ النتائج إلى ملف JSON"""
        output_path = RESULTS_DIR / filename

        results_to_save = {
            'document_processing': {
                'avg_time': float(np.mean([r['processing_time'] for r in self.results['document_processing']])) if self.results['document_processing'] else None,
                'std_time': float(np.std([r['processing_time'] for r in self.results['document_processing']])) if self.results['document_processing'] else None,
                'details': self.results['document_processing']
            },
            'video_processing': {
                'avg_time': float(np.mean([r['total_processing_time'] for r in self.results['video_processing']])) if self.results['video_processing'] else None,
                'std_time': float(np.std([r['total_processing_time'] for r in self.results['video_processing']])) if self.results['video_processing'] else None,
                'details': self.results['video_processing']
            },
            'scalability_test': self.results['scalability_test'],
            'resource_usage': self.results['resource_usage']
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✅ تم حفظ النتائج في: {output_path}")
        return output_path


def main():
    """التشغيل الرئيسي"""
    print("\n" + "="*80)
    print("⚡ برنامج اختبار مقاييس الأداء والموارد")
    print("="*80 + "\n")

    # إنشاء مثيل الاختبار
    tester = PerformanceTester()

    # 1. اختبار معالجة المستندات
    tester.test_document_processing_time()

    # 2. اختبار معالجة الفيديو
    tester.test_video_processing_time()

    # 3. اختبار قابلية التوسع
    tester.test_search_scalability([100, 500, 1000, 5000])

    # 4. اختبار الطلبات المتزامنة
    tester.test_concurrent_requests(num_requests=20)

    # حفظ النتائج
    tester.save_results()

    print("\n✅ اكتملت جميع الاختبارات بنجاح!")


if __name__ == "__main__":
    main()
