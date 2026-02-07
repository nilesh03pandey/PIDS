import os
import hashlib
import hmac
import time
import json
from pymongo import MongoClient
import datetime

# Configuration
DB_URI = "mongodb://localhost:27017/"
DB_NAME = "audit_ledger"
COLLECTION_NAME = "chain"
SECRET_KEY_FILE = "secret.key"

class AuditLogger:
    def __init__(self):
        self.client = MongoClient(DB_URI)
        self.db = self.client[DB_NAME]
        self.col = self.db[COLLECTION_NAME]
        self.secret_key = self._load_or_generate_key()
        
        # Ensure Genesis block exists
        if self.col.count_documents({}) == 0:
            self._create_genesis_block()

    def _load_or_generate_key(self):
        if os.path.exists(SECRET_KEY_FILE):
            with open(SECRET_KEY_FILE, "rb") as f:
                return f.read()
        else:
            key = os.urandom(32)
            with open(SECRET_KEY_FILE, "wb") as f:
                f.write(key)
            print("[Audit] Generated new HMAC secret key.")
            return key

    def _create_genesis_block(self):
        genesis_data = {
            "timestamp": time.time(),
            "event_type": "GENESIS",
            "data": "Genesis Block",
            "user": "SYSTEM",
            "previous_hash": "0" * 64
        }
        self._log_entry(genesis_data)
        print("[Audit] Genesis block created.")

    def _compute_hash(self, content):
        return hashlib.sha256(content.encode()).hexdigest()

    def _compute_hmac(self, content):
        return hmac.new(self.secret_key, content.encode(), hashlib.sha256).hexdigest()

    def _get_last_hash(self):
        last_entry = self.col.find_one(sort=[("timestamp", -1)])
        if last_entry:
            return last_entry["hash"]
        return "0" * 64

    def _log_entry(self, entry_data):
        # entry_data must contain: timestamp, event_type, data, user, previous_hash
        # We need to construct a canonical string for hashing to ensure consistency
        # JSON dump with sort_keys=True is a good way
        
        # Construct payload string for hashing (exclude fields that are generated from the payload)
        payload = f"{entry_data['previous_hash']}|{entry_data['timestamp']}|{entry_data['event_type']}|{str(entry_data['data'])}|{entry_data['user']}"
        
        current_hash = self._compute_hash(payload)
        signature = self._compute_hmac(current_hash)
        
        full_entry = entry_data.copy()
        full_entry["hash"] = current_hash
        full_entry["signature"] = signature
        
        self.col.insert_one(full_entry)
        return full_entry

    def log_event(self, event_type, data, user="System"):
        """
        Public method to log an event.
        """
        prev_hash = self._get_last_hash()
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data,
            "user": user,
            "previous_hash": prev_hash
        }
        logged = self._log_entry(entry)
        # print(f"🔒 [Audit] Logged {event_type} (Hash: {logged['hash'][:8]}...)")
        return logged["hash"]

    def sign_file(self, filepath, user="System"):
        """
        Reads a file, computes its SHA256, and logs it.
        """
        if not os.path.exists(filepath):
            print(f"[Audit] File not found: {filepath}")
            return None
            
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        file_hash = sha256_hash.hexdigest()
        
        data = {
            "action": "FILE_SIGNING",
            "filepath": filepath,
            "file_hash": file_hash
        }
        
        return self.log_event("FILE_INTEGRITY", data, user)

    def verify_chain(self):
        """
        Verifies the integrity of the entire chain.
        Returns (True, "OK") or (False, "Error details")
        """
        chain = list(self.col.find().sort("timestamp", 1))
        if not chain:
            return False, "Chain empty"
            
        prev_hash = "0" * 64
        
        for i, block in enumerate(chain):
            # 1. Check Linkage
            if block["previous_hash"] != prev_hash:
                return False, f"Broken link at block {i} (ID: {block['_id']}). Expected Prev: {prev_hash}, Got: {block['previous_hash']}"
            
            # 2. Re-compute Hash
            payload = f"{block['previous_hash']}|{block['timestamp']}|{block['event_type']}|{str(block['data'])}|{block['user']}"
            computed_hash = self._compute_hash(payload)
            
            if computed_hash != block["hash"]:
                return False, f"Hash mismatch at block {i}. Stored: {block['hash']}, Computed: {computed_hash}"
                
            # 3. Verify Signature
            computed_sig = self._compute_hmac(computed_hash)
            if computed_sig != block["signature"]:
                return False, f"Invalid signature at block {i}."
                
            prev_hash = block["hash"]
            
        return True, f"Chain verified. Length: {len(chain)}"

# Global instance
audit_logger = AuditLogger()
