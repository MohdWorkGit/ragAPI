"""
تشغيل جميع اختبارات النظام وتوليد تقرير شامل
Run All System Tests and Generate Comprehensive Report
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# إضافة المسار الحالي
sys.path.insert(0, str(Path(__file__).parent))

# استيراد برامج الاختبار
from test_search_performance import SearchPerformanceTester
from test_video_analysis import VideoAnalysisTester
from test_writer_extraction import WriterExtractionTester
from test_performance_metrics import PerformanceTester
from generate_report import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """طباعة عنوان مزخرف"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_all_tests(skip_tests: list = None):
    """
    تشغيل جميع الاختبارات

    Args:
        skip_tests: قائمة بأسماء الاختبارات المراد تخطيها
                   مثل: ['search', 'video', 'writer', 'performance']
    """
    if skip_tests is None:
        skip_tests = []

    start_time = time.time()
    results = {}

    print("\n" + "🚀"*40)
    print_header("بدء تشغيل جميع اختبارات نظام RAG API")
    print(f"⏰ الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # =================================================================
    # 1. اختبار البحث الدلالي
    # =================================================================
    if 'search' not in skip_tests:
        print_header("1️⃣  اختبار البحث الدلالي (Search Performance)")
        try:
            search_tester = SearchPerformanceTester()
            search_tester.load_test_queries("test_data/search_queries.json")
            search_results = search_tester.run_tests(k_values=[1, 3, 5, 10])
            search_tester.save_results()
            search_tester.generate_markdown_table()
            results['search'] = 'SUCCESS ✅'
            logger.info("✅ اكتمل اختبار البحث الدلالي بنجاح")
        except Exception as e:
            results['search'] = f'FAILED ❌: {str(e)}'
            logger.error(f"❌ فشل اختبار البحث الدلالي: {e}")
    else:
        results['search'] = 'SKIPPED ⏭️'
        logger.info("⏭️  تم تخطي اختبار البحث الدلالي")

    # =================================================================
    # 2. اختبار تحليل الفيديو
    # =================================================================
    if 'video' not in skip_tests:
        print_header("2️⃣  اختبار تحليل الفيديو (Video Analysis)")
        try:
            video_tester = VideoAnalysisTester()
            video_tester.load_test_videos("test_data/video_test_cases.json")
            video_results = video_tester.run_tests()
            video_tester.save_results()
            results['video'] = 'SUCCESS ✅'
            logger.info("✅ اكتمل اختبار تحليل الفيديو بنجاح")
        except Exception as e:
            results['video'] = f'FAILED ❌: {str(e)}'
            logger.error(f"❌ فشل اختبار تحليل الفيديو: {e}")
    else:
        results['video'] = 'SKIPPED ⏭️'
        logger.info("⏭️  تم تخطي اختبار تحليل الفيديو")

    # =================================================================
    # 3. اختبار استخراج الكتّاب
    # =================================================================
    if 'writer' not in skip_tests:
        print_header("3️⃣  اختبار استخراج الكتّاب (Writer Extraction)")
        try:
            writer_tester = WriterExtractionTester()
            writer_tester.load_test_documents("test_data/writer_test_documents.json")
            writer_results = writer_tester.run_tests()
            writer_tester.save_results()
            results['writer'] = 'SUCCESS ✅'
            logger.info("✅ اكتمل اختبار استخراج الكتّاب بنجاح")
        except Exception as e:
            results['writer'] = f'FAILED ❌: {str(e)}'
            logger.error(f"❌ فشل اختبار استخراج الكتّاب: {e}")
    else:
        results['writer'] = 'SKIPPED ⏭️'
        logger.info("⏭️  تم تخطي اختبار استخراج الكتّاب")

    # =================================================================
    # 4. اختبار مقاييس الأداء
    # =================================================================
    if 'performance' not in skip_tests:
        print_header("4️⃣  اختبار مقاييس الأداء (Performance Metrics)")
        try:
            perf_tester = PerformanceTester()
            perf_tester.test_document_processing_time()
            perf_tester.test_video_processing_time()
            perf_tester.test_search_scalability([100, 500, 1000])
            perf_tester.save_results()
            results['performance'] = 'SUCCESS ✅'
            logger.info("✅ اكتمل اختبار مقاييس الأداء بنجاح")
        except Exception as e:
            results['performance'] = f'FAILED ❌: {str(e)}'
            logger.error(f"❌ فشل اختبار مقاييس الأداء: {e}")
    else:
        results['performance'] = 'SKIPPED ⏭️'
        logger.info("⏭️  تم تخطي اختبار مقاييس الأداء")

    # =================================================================
    # 5. توليد التقارير
    # =================================================================
    print_header("5️⃣  توليد التقارير الشاملة (Report Generation)")
    try:
        report_gen = ReportGenerator()
        report_files = report_gen.generate_all_reports()
        results['report'] = 'SUCCESS ✅'
        logger.info("✅ اكتمل توليد التقارير بنجاح")
    except Exception as e:
        results['report'] = f'FAILED ❌: {str(e)}'
        logger.error(f"❌ فشل توليد التقارير: {e}")

    # =================================================================
    # النتيجة النهائية
    # =================================================================
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("  📊 ملخص النتائج")
    print("="*80 + "\n")

    print("┌" + "─"*78 + "┐")
    print(f"│ {'الاختبار':<40} {'النتيجة':<37} │")
    print("├" + "─"*78 + "┤")

    for test_name, status in results.items():
        display_names = {
            'search': 'البحث الدلالي (Search Performance)',
            'video': 'تحليل الفيديو (Video Analysis)',
            'writer': 'استخراج الكتّاب (Writer Extraction)',
            'performance': 'مقاييس الأداء (Performance Metrics)',
            'report': 'توليد التقارير (Report Generation)'
        }

        name = display_names.get(test_name, test_name)
        # تقليم النص إذا كان طويلاً
        if len(status) > 35:
            status = status[:32] + "..."

        print(f"│ {name:<40} {status:<37} │")

    print("└" + "─"*78 + "┘")

    print(f"\n⏱️  إجمالي الوقت المستغرق: {total_time:.2f} ثانية ({total_time/60:.2f} دقيقة)\n")

    # عد النجاحات والفشل
    success_count = sum(1 for s in results.values() if 'SUCCESS' in s)
    failed_count = sum(1 for s in results.values() if 'FAILED' in s)
    skipped_count = sum(1 for s in results.values() if 'SKIPPED' in s)

    print(f"✅ ناجح: {success_count}")
    print(f"❌ فاشل: {failed_count}")
    print(f"⏭️  متخطى: {skipped_count}")
    print(f"📁 الإجمالي: {len(results)}")

    print("\n" + "="*80)

    if failed_count == 0:
        print("🎉 تمت جميع الاختبارات بنجاح!")
    else:
        print(f"⚠️  فشلت {failed_count} اختبار(ات). راجع السجلات أعلاه.")

    print("="*80 + "\n")

    # طباعة مواقع الملفات
    print("📂 مواقع الملفات المُنشأة:")
    print(f"  📊 النتائج: test_results/")
    print(f"  📄 التقارير: reports/")
    print()

    return results, total_time


def main():
    """التشغيل الرئيسي"""
    import argparse

    parser = argparse.ArgumentParser(
        description='تشغيل جميع اختبارات نظام RAG API'
    )
    parser.add_argument(
        '--skip',
        nargs='+',
        choices=['search', 'video', 'writer', 'performance'],
        help='الاختبارات المراد تخطيها'
    )

    args = parser.parse_args()

    # تشغيل جميع الاختبارات
    results, total_time = run_all_tests(skip_tests=args.skip or [])

    # الخروج برمز الخطأ المناسب
    failed = sum(1 for s in results.values() if 'FAILED' in s)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
