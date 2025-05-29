#!/usr/bin/env python3
"""
Test script to verify the record count fix for large datasets.
This script creates a test CSV file with more than 10,000 rows and tests
the dataset processing to ensure the correct record count is returned.
"""

import os
import sys
import csv
import tempfile
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from services.dataset_service import DatasetService

def create_test_csv(num_rows=15000):
    """Create a test CSV file with the specified number of rows."""
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    
    # Write header
    fieldnames = ['id', 'name', 'age', 'city', 'salary', 'department']
    writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
    writer.writeheader()
    
    # Write data rows
    departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance']
    cities = ['New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle']
    
    for i in range(1, num_rows + 1):
        writer.writerow({
            'id': i,
            'name': f'Person_{i}',
            'age': 25 + (i % 40),
            'city': cities[i % len(cities)],
            'salary': 50000 + (i % 50000),
            'department': departments[i % len(departments)]
        })
    
    temp_file.close()
    return temp_file.name

def test_record_count():
    """Test the record count functionality."""
    print("Testing record count fix...")
    
    # Create test CSV with 15,000 rows
    test_rows = 15000
    print(f"Creating test CSV with {test_rows} rows...")
    csv_file = create_test_csv(test_rows)
    
    try:
        # Initialize dataset service
        dataset_service = DatasetService(upload_folder=tempfile.gettempdir())
        
        # Process the CSV file
        print("Processing CSV file...")
        file_info = {
            'path': csv_file,
            'format': 'csv',
            'size': f"{os.path.getsize(csv_file) / 1024 / 1024:.2f} MB"
        }
        
        result = dataset_service.process_dataset(file_info, 'csv')
        
        if result:
            print(f"Processing successful!")
            print(f"Format: {result.get('format', 'Unknown')}")
            print(f"Columns: {len(result.get('columns', []))}")
            print(f"Record count: {result.get('record_count', 0)}")
            print(f"Sample data rows: {len(result.get('sample_data', []))}")
            
            # Verify the record count
            expected_count = test_rows
            actual_count = result.get('record_count', 0)
            
            if actual_count == expected_count:
                print(f"✅ SUCCESS: Record count is correct ({actual_count})")
                return True
            else:
                print(f"❌ FAILURE: Expected {expected_count} records, got {actual_count}")
                return False
        else:
            print("❌ FAILURE: Processing returned None")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False
    finally:
        # Clean up
        if os.path.exists(csv_file):
            os.unlink(csv_file)

def test_json_record_count():
    """Test JSON record count functionality."""
    print("\nTesting JSON record count...")
    
    # Create test JSON with array of objects
    test_rows = 12000
    print(f"Creating test JSON with {test_rows} records...")
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    
    # Write JSON array
    temp_file.write('[\n')
    for i in range(1, test_rows + 1):
        record = {
            'id': i,
            'name': f'Item_{i}',
            'value': i * 10,
            'category': f'Category_{i % 5}'
        }
        
        import json
        temp_file.write(json.dumps(record))
        if i < test_rows:
            temp_file.write(',\n')
        else:
            temp_file.write('\n')
    
    temp_file.write(']')
    temp_file.close()
    
    try:
        # Initialize dataset service
        dataset_service = DatasetService(upload_folder=tempfile.gettempdir())
        
        # Process the JSON file
        print("Processing JSON file...")
        file_info = {
            'path': temp_file.name,
            'format': 'json',
            'size': f"{os.path.getsize(temp_file.name) / 1024 / 1024:.2f} MB"
        }
        
        result = dataset_service.process_dataset(file_info, 'json')
        
        if result:
            print(f"Processing successful!")
            print(f"Format: {result.get('format', 'Unknown')}")
            print(f"Record count: {result.get('record_count', 0)}")
            print(f"Is array: {result.get('is_array', False)}")
            print(f"Sample data rows: {len(result.get('sample_data', []))}")
            
            # Verify the record count
            expected_count = test_rows
            actual_count = result.get('record_count', 0)
            
            if actual_count == expected_count:
                print(f"✅ SUCCESS: JSON record count is correct ({actual_count})")
                return True
            else:
                print(f"❌ FAILURE: Expected {expected_count} records, got {actual_count}")
                return False
        else:
            print("❌ FAILURE: Processing returned None")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    print("=" * 60)
    print("RECORD COUNT FIX TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    
    # Test CSV record count
    csv_success = test_record_count()
    
    # Test JSON record count
    json_success = test_json_record_count()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"CSV Test: {'PASSED' if csv_success else 'FAILED'}")
    print(f"JSON Test: {'PASSED' if json_success else 'FAILED'}")
    
    if csv_success and json_success:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)
