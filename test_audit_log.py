import unittest
import time
import os
from audit_manager import audit_logger, DB_NAME, COLLECTION_NAME

class TestAuditLog(unittest.TestCase):
    def test_chain_integrity(self):
        print("\n[Test] Logging events...")
        # Log a few events
        audit_logger.log_event("TEST_EVENT", {"id": 1})
        audit_logger.log_event("TEST_EVENT", {"id": 2})
        audit_logger.log_event("TEST_EVENT", {"id": 3})
        
        print("[Test] Verifying chain...")
        valid, msg = audit_logger.verify_chain()
        print(f"Result: {msg}")
        self.assertTrue(valid, f"Chain should be valid. Error: {msg}")

    def test_tamper_detection(self):
        print("\n[Test] Simulating tampering...")
        # Get last inserted doc
        last_doc = audit_logger.col.find_one(sort=[("timestamp", -1)])
        original_data = last_doc["data"]
        
        # Tamper with data
        audit_logger.col.update_one(
            {"_id": last_doc["_id"]},
            {"$set": {"data": {"id": 999, "tampered": True}}}
        )
        
        print("[Test] Verifying chain (expect failure)...")
        valid, msg = audit_logger.verify_chain()
        print(f"Result: {msg}")
        self.assertFalse(valid, "Chain should be invalid after tampering")
        
        # Restore data to fix chain for other tests? 
        # Actually, for unit test isolation, we might leave it broken or fix it.
        # Let's fix it to be nice.
        audit_logger.col.update_one(
            {"_id": last_doc["_id"]},
            {"$set": {"data": original_data}}
        )

    def test_file_signing(self):
        print("\n[Test] File signing...")
        dummy_file = "test_file.txt"
        with open(dummy_file, "w") as f:
            f.write("Integrity Check")
            
        audit_logger.sign_file(dummy_file)
        
        # Verify the log exists
        log = audit_logger.col.find_one(
            {"event_type": "FILE_INTEGRITY", "data.filepath": dummy_file},
            sort=[("timestamp", -1)]
        )
        self.assertIsNotNone(log)
        print(f"[Test] File log found: {log['data']['file_hash']}")
        
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

if __name__ == '__main__':
    unittest.main()
