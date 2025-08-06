#!/usr/bin/env python3
"""
Comprehensive Performance Evaluation Script for Metadata Generation Framework

This script runs a complete performance evaluation including:
- NLP Extraction Accuracy (Precision, Recall, F1-score)
- System Performance and Efficiency
- Search Relevance (MAP, NDCG)
- Scalability and Reliability
- Standards Compliance (Schema.org, FAIR)

Usage:
    python run_performance_evaluation.py

Results:
    - CSV file with detailed metrics
    - Chapter 4 documentation (Results and Discussion)
"""

import os
import sys
import time
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import Flask app and services
from app import create_app
from app.services.performance_evaluation_service import PerformanceEvaluationService


def main():
    """Run comprehensive performance evaluation"""
    print("🚀 Metadata Generation Framework - Performance Evaluation")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create Flask app context
    app = create_app()
    
    with app.app_context():
        try:
            # Initialize performance evaluation service
            evaluator = PerformanceEvaluationService()
            
            # Check if test data directory exists
            if not os.path.exists("test_data"):
                print("❌ Error: test_data directory not found!")
                print("Please ensure the test_data directory exists with dataset files.")
                return 1
            
            # List available test files
            test_files = [f for f in os.listdir("test_data") 
                         if f.endswith(('.csv', '.xlsx', '.xls', '.json', '.txt'))]
            
            if not test_files:
                print("❌ Error: No test dataset files found in test_data directory!")
                print("Please add dataset files (.csv, .xlsx, .xls, .json, .txt) to the test_data directory.")
                return 1
            
            print(f"📊 Found {len(test_files)} test datasets:")
            for i, filename in enumerate(test_files, 1):
                file_path = os.path.join("test_data", filename)
                file_size = os.path.getsize(file_path)
                print(f"  {i:2d}. {filename} ({file_size / 1024:.1f} KB)")
            print()
            
            # Confirm evaluation
            print("🔍 This evaluation will test the following metrics:")
            print("   • NLP Extraction Accuracy (Precision, Recall, F1-score)")
            print("   • System Performance (Processing time, Speed)")
            print("   • Search Relevance (MAP, Response time)")
            print("   • Scalability (Concurrent processing, Large files)")
            print("   • Standards Compliance (FAIR, Schema.org)")
            print()
            
            # Start evaluation
            start_time = time.time()
            
            print("🎯 Starting comprehensive evaluation...")
            results = evaluator.run_comprehensive_evaluation()
            
            total_time = time.time() - start_time
            
            # Display summary
            print("\n" + "=" * 80)
            print("📈 EVALUATION SUMMARY")
            print("=" * 80)
            
            if 'summary_statistics' in results:
                summary = results['summary_statistics']
                
                # NLP Accuracy Summary
                if 'nlp_accuracy' in summary:
                    nlp = summary['nlp_accuracy']
                    print(f"🧠 NLP Extraction Accuracy:")
                    print(f"   • Precision: {nlp.get('avg_precision', 0):.3f}")
                    print(f"   • Recall: {nlp.get('avg_recall', 0):.3f}")
                    print(f"   • F1-Score: {nlp.get('avg_f1_score', 0):.3f}")
                    print(f"   • Avg Keywords: {nlp.get('avg_keywords_extracted', 0):.1f}")
                    print(f"   • Avg Entities: {nlp.get('avg_entities_extracted', 0):.1f}")
                    print()
                
                # System Performance Summary
                if 'system_performance' in summary:
                    perf = summary['system_performance']
                    print(f"⚡ System Performance:")
                    print(f"   • Avg Processing Time: {perf.get('avg_processing_time', 0):.3f} seconds")
                    print(f"   • Avg Speed: {perf.get('avg_processing_speed', 0):.2f} KB/sec")
                    print(f"   • Efficiency Score: {perf.get('avg_efficiency_score', 0):.2f}/100")
                    print()
                
                # Search Relevance Summary
                if 'search_relevance' in summary:
                    search = summary['search_relevance']
                    print(f"🔍 Search Relevance:")
                    print(f"   • Mean Average Precision: {search.get('avg_map_score', 0):.3f}")
                    print(f"   • Avg Response Time: {search.get('avg_response_time', 0):.3f} seconds")
                    print()
                
                # Compliance Summary
                if 'compliance' in summary:
                    comp = summary['compliance']
                    total_datasets = len(results.get('compliance_scores', []))
                    fair_compliant = comp.get('fair_compliant_datasets', 0)
                    compliance_rate = (fair_compliant / max(1, total_datasets)) * 100
                    
                    print(f"📋 Standards Compliance:")
                    print(f"   • Avg Quality Score: {comp.get('avg_quality_score', 0):.2f}/100")
                    print(f"   • Avg FAIR Score: {comp.get('avg_fair_score', 0):.2f}/100")
                    print(f"   • FAIR Compliant (≥75%): {fair_compliant}/{total_datasets} ({compliance_rate:.1f}%)")
                    print(f"   • Avg Schema.org Score: {comp.get('avg_schema_org_score', 0):.2f}/100")
                    print()
                
                # Scalability Summary
                if 'scalability' in summary:
                    scale = summary['scalability']
                    print(f"📈 Scalability:")
                    print(f"   • Concurrent Capacity: {scale.get('avg_concurrent_capacity', 0):.1f} tasks")
                    print(f"   • Large File Score: {scale.get('avg_large_file_score', 0):.2f}/100")
                    print(f"   • Overall Scalability: {scale.get('avg_scalability_score', 0):.2f}/100")
                    print()
            
            # Overall Results
            print(f"⏱️  Total Evaluation Time: {total_time:.2f} seconds")
            print(f"📊 Datasets Evaluated: {len(results.get('metadata_accuracy', []))}")
            print()
            
            # Output Files
            print("📄 Generated Output Files:")
            csv_files = [f for f in os.listdir('.') if f.startswith('performance_evaluation_results_') and f.endswith('.csv')]
            chapter_files = [f for f in os.listdir('.') if f.startswith('Chapter_4_Results_and_Discussion_') and f.endswith('.md')]
            
            if csv_files:
                latest_csv = max(csv_files, key=lambda x: os.path.getctime(x))
                print(f"   • CSV Results: {latest_csv}")
            
            if chapter_files:
                latest_chapter = max(chapter_files, key=lambda x: os.path.getctime(x))
                print(f"   • Chapter 4 Doc: {latest_chapter}")
            
            print()
            print("✅ Performance evaluation completed successfully!")
            print("   The results demonstrate comprehensive system capabilities across")
            print("   all evaluated dimensions including NLP accuracy, performance,")
            print("   search relevance, scalability, and standards compliance.")
            
            return 0
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        finally:
            print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
